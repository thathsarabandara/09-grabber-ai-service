from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models import GestureMapping, GestureSetting, CustomGestureTemplate
from app.core.state import gesture_controller

router = APIRouter()

# Canonical robot movements – these are the ONLY permitted actions
VALID_ACTIONS = {
    "BASE LEFT", "BASE RIGHT",
    "SHOULDER LEFT", "SHOULDER RIGHT",
    "ELBOW LEFT", "ELBOW RIGHT",
    "OPEN GRIP", "CLOSE GRIP",
}

# ── DB helpers ─────────────────────────────────────────────────────────────

def get_gesture_settings_db(db: Session):
    setting = db.query(GestureSetting).first()
    if not setting:
        setting = GestureSetting(control_enabled=False, safety_enabled=True)
        db.add(setting)
        db.commit()
        db.refresh(setting)
    return setting

def get_gesture_mappings_db(db: Session):
    mappings = db.query(GestureMapping).all()
    return {m.gesture_name: m.action for m in mappings}

# ── Settings & mappings ────────────────────────────────────────────────────

@router.get("/gesture/settings")
def get_settings(db: Session = Depends(get_db)):
    setting  = get_gesture_settings_db(db)
    mappings = get_gesture_mappings_db(db)
    return {
        "control_enabled": setting.control_enabled,
        "safety_enabled":  setting.safety_enabled,
        "mappings":        mappings
    }

@router.post("/gesture/settings")
def update_settings(payload: dict, db: Session = Depends(get_db)):
    setting = get_gesture_settings_db(db)
    if "control_enabled" in payload:
        setting.control_enabled = bool(payload["control_enabled"])
    if "safety_enabled" in payload:
        setting.safety_enabled = bool(payload["safety_enabled"])
    db.commit()
    db.refresh(setting)
    return {
        "control_enabled": setting.control_enabled,
        "safety_enabled":  setting.safety_enabled,
        "mappings":        get_gesture_mappings_db(db)
    }

@router.post("/gesture/mappings")
def add_or_update_mapping(payload: dict, db: Session = Depends(get_db)):
    gesture_name = payload.get("gesture_name")
    action       = payload.get("action", "BASE LEFT")
    if not gesture_name:
        raise HTTPException(status_code=400, detail="gesture_name is required")
    if action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{action}'. Must be one of: {sorted(VALID_ACTIONS)}"
        )

    mapping = db.query(GestureMapping).filter(GestureMapping.gesture_name == gesture_name).first()
    if mapping:
        mapping.action = action
    else:
        mapping = GestureMapping(gesture_name=gesture_name, action=action)
        db.add(mapping)
    db.commit()
    return {"status": "success", "mappings": get_gesture_mappings_db(db)}

@router.delete("/gesture/mappings/{gesture_name}")
def delete_mapping(gesture_name: str, db: Session = Depends(get_db)):
    mapping = db.query(GestureMapping).filter(GestureMapping.gesture_name == gesture_name).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    
    # Remove from ChromaDB too
    gesture_controller.remove_gesture(gesture_name)

    db.delete(mapping)
    db.commit()
    return {"status": "success", "mappings": get_gesture_mappings_db(db)}

# ── Custom gesture registration ────────────────────────────────────────────

@router.post("/gesture/custom/register")
async def register_custom_gesture(
    frame: UploadFile = File(..., description="JPEG frame containing the hand pose"),
    gesture_name: str = Form(..., description="Name for this custom gesture"),
    action: str       = Form("UNASSIGNED", description="Robot action to map to this gesture"),
    db: Session       = Depends(get_db)
):
    """
    Register a custom gesture from a JPEG frame.

    - MediaPipe detects the hand and extracts 21 landmarks
    - Landmarks are normalised and stored as a 63-dim vector in ChromaDB
    - A CustomGestureTemplate record is saved in MySQL for management
    - A GestureMapping entry is created/updated linking the gesture to a robot action

    Capture 5-10 samples from slightly different angles for best accuracy.
    """
    jpeg_bytes = await frame.read()
    try:
        result = gesture_controller.register_from_frame(jpeg_bytes, gesture_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    mapping = db.query(GestureMapping).filter(GestureMapping.gesture_name == gesture_name).first()
    if not mapping:
        mapping = GestureMapping(gesture_name=gesture_name, action=action)
        db.add(mapping)
        db.flush()
    elif action != "UNASSIGNED":
        mapping.action = action

    # Persist tracking record in MySQL referencing the mapping
    db.add(CustomGestureTemplate(
        gesture_name=gesture_name,
        vector_id=result["vector_id"]
    ))

    db.commit()

    return {
        "status":       "success",
        "gesture_name": gesture_name,
        "action":       action,
        "vector_id":    result["vector_id"],
        "sample_count": result["sample_count"],
        "message":      f"Sample {result['sample_count']} registered. Capture more samples for better accuracy."
    }

@router.get("/gesture/custom")
def list_custom_gestures(db: Session = Depends(get_db)):
    """List all registered custom gestures with sample counts and mapped actions."""
    vector_counts = gesture_controller.list_gestures()   # from ChromaDB
    mappings      = get_gesture_mappings_db(db)

    return {
        "gestures": [
            {
                "gesture_name": name,
                "sample_count": count,
                "action":       mappings.get(name, "UNASSIGNED")
            }
            for name, count in vector_counts.items()
        ],
        "total_gestures": len(vector_counts)
    }

@router.delete("/gesture/custom/{gesture_name}")
def delete_custom_gesture(gesture_name: str, db: Session = Depends(get_db)):
    """
    Delete all ChromaDB vectors, MySQL tracking records, and the action mapping
    for a custom gesture.
    """
    deleted_count = gesture_controller.remove_gesture(gesture_name)

    # Remove MySQL tracking records
    db.query(CustomGestureTemplate).filter(
        CustomGestureTemplate.gesture_name == gesture_name
    ).delete()

    # Remove action mapping
    db.query(GestureMapping).filter(
        GestureMapping.gesture_name == gesture_name
    ).delete()

    db.commit()

    return {
        "status":         "success",
        "gesture_name":   gesture_name,
        "vectors_deleted": deleted_count,
        "message":        f"Gesture '{gesture_name}' and all {deleted_count} sample(s) removed."
    }

@router.post("/gesture/train")
def train_gesture():
    """Returns MediaPipe and ChromaDB readiness status."""
    return gesture_controller.train_wizard()

@router.post("/gesture/recognize")
async def recognize_gesture(
    frame: UploadFile = File(...),
    robot_id: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Process a live webcam frame, detect gestures, draw annotations,
    and dispatch commands if control is enabled.
    """
    import cv2
    import numpy as np
    import base64
    from app.api.routes.stream import _get_gesture_config
    
    jpeg_bytes = await frame.read()
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    mappings, control_enabled, safety_enabled = _get_gesture_config()
    
    # Process frame — use robot_id as session key for per-stream motion tracking
    session_id = robot_id or "default"
    annotated_frame, detected_action = gesture_controller.process_frame(img, mappings, session_id=session_id)
    
    # Dispatch control if needed
    if control_enabled and robot_id and detected_action:
        gesture_controller.publish_robot_command(robot_id, detected_action)

    # Encode back to JPEG
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    b64_image = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": "success",
        "detected_action": detected_action,
        "image": f"data:image/jpeg;base64,{b64_image}"
    }
