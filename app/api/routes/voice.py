from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models import VoiceSetting, VoiceCommand

router = APIRouter()

def get_voice_settings_db(db: Session):
    setting = db.query(VoiceSetting).first()
    if not setting:
        setting = VoiceSetting(wake_word="Hey Robot", language="English (US)")
        db.add(setting)
        db.commit()
        db.refresh(setting)
    return setting

def get_voice_commands_db(db: Session):
    commands = db.query(VoiceCommand).all()
    if not commands:
        defaults = [
            {"phrase": "Open Gripper",   "action": "RELEASE",       "target": "gripper"},
            {"phrase": "Close Gripper",  "action": "GRAB",          "target": "gripper"},
            {"phrase": "Move Left",      "action": "MOVE_LEFT",     "target": "arm"},
            {"phrase": "Move Right",     "action": "MOVE_RIGHT",    "target": "arm"},
            {"phrase": "Home Position",  "action": "MOVE_HOME",     "target": "arm"},
            {"phrase": "Emergency Stop", "action": "STOP_ALL",      "target": "all"},
            {"phrase": "Pick Bottle",    "action": "Pick And Place", "target": "Bottle"},
            {"phrase": "Tidy Desk",      "action": "Smart Sorting", "target": "All Items"},
        ]
        for item in defaults:
            db.add(VoiceCommand(phrase=item["phrase"], action=item["action"], target=item["target"]))
        db.commit()
        commands = db.query(VoiceCommand).all()
    return commands

@router.get("/voice/commands")
def get_commands(db: Session = Depends(get_db)):
    setting = get_voice_settings_db(db)
    commands = get_voice_commands_db(db)
    return {
        "wake_word": setting.wake_word,
        "language": setting.language,
        "commands": [
            {"id": c.id, "phrase": c.phrase, "action": c.action, "target": c.target}
            for c in commands
        ]
    }

@router.post("/voice/settings")
def update_settings(payload: dict, db: Session = Depends(get_db)):
    setting = get_voice_settings_db(db)
    if "wake_word" in payload:
        setting.wake_word = payload["wake_word"]
    if "language" in payload:
        setting.language = payload["language"]
    db.commit()
    db.refresh(setting)
    return {
        "status": "success",
        "wake_word": setting.wake_word,
        "language": setting.language
    }

@router.post("/voice/commands")
def add_command(payload: dict, db: Session = Depends(get_db)):
    phrase = payload.get("phrase")
    action = payload.get("action")
    target = payload.get("target", "any")
    if not phrase or not action:
        raise HTTPException(status_code=400, detail="Phrase and action are required")
    existing = db.query(VoiceCommand).filter(VoiceCommand.phrase == phrase).first()
    if existing:
        existing.action = action
        existing.target = target
    else:
        db.add(VoiceCommand(phrase=phrase, action=action, target=target))
    db.commit()
    commands = get_voice_commands_db(db)
    return {
        "status": "success",
        "commands": [
            {"id": c.id, "phrase": c.phrase, "action": c.action, "target": c.target}
            for c in commands
        ]
    }

@router.delete("/voice/commands/{cmd_id}")
def delete_command(cmd_id: int, db: Session = Depends(get_db)):
    command = db.query(VoiceCommand).filter(VoiceCommand.id == cmd_id).first()
    if not command:
        raise HTTPException(status_code=404, detail="Command rule not found")
    db.delete(command)
    db.commit()
    commands = get_voice_commands_db(db)
    return {
        "status": "success",
        "commands": [
            {"id": c.id, "phrase": c.phrase, "action": c.action, "target": c.target}
            for c in commands
        ]
    }

@router.get("/voice/settings")
def get_settings(db: Session = Depends(get_db)):
    """GET settings endpoint to retrieve settings and command list (requested by dashboard)."""
    return get_commands(db)


def resolve_physical_robot_id(robot_id: str, db: Session) -> str:
    """
    If robot_id is a UUID, resolve it to the physical robot_id (e.g. GRABBER-V1-ESP32).
    If robot_id is not provided, query the first robot in grabber_robot.robots.
    """
    import uuid
    from sqlalchemy import text

    physical_robot_id = robot_id

    # If no robot_id is provided, try to fetch the first one from grabber_robot.robots
    if not physical_robot_id:
        try:
            res = db.execute(text("SELECT robot_id FROM grabber_robot.robots LIMIT 1"))
            row = res.fetchone()
            if row:
                physical_robot_id = row[0]
                print(f"[Voice] Auto-resolved to first robot: {physical_robot_id}", flush=True)
        except Exception as e:
            print(f"[Voice] Failed to fetch first robot from DB: {e}", flush=True)
        return physical_robot_id

    # If it's a UUID, resolve it
    try:
        val = uuid.UUID(str(robot_id))
        try:
            res = db.execute(
                text("SELECT robot_id FROM grabber_robot.robots WHERE id = :hex OR id = :str"),
                {"hex": val.hex, "str": str(val)}
            )
            row = res.fetchone()
            if row:
                physical_robot_id = row[0]
                print(f"[Voice] Resolved UUID {robot_id} to physical robot_id {physical_robot_id}", flush=True)
        except Exception as db_err:
            print(f"[Voice] Failed to resolve robot UUID: {db_err}", flush=True)
    except ValueError:
        pass

    return physical_robot_id


def execute_voice_action(physical_robot_id: str, action: str, target: str):
    if not physical_robot_id:
        print("[Voice] No robot ID available for execution", flush=True)
        return

    import paho.mqtt.client as mqtt
    import json
    from app.core.config import settings

    subtopic = None
    payload = {}

    if action == "RELEASE":
        subtopic = "open-gripper"
    elif action == "GRAB":
        subtopic = "close-gripper"
    elif action == "MOVE_LEFT":
        from app.core.state import gesture_controller
        gesture_controller.publish_robot_command(physical_robot_id, "BASE LEFT")
        return
    elif action == "MOVE_RIGHT":
        from app.core.state import gesture_controller
        gesture_controller.publish_robot_command(physical_robot_id, "BASE RIGHT")
        return
    elif action == "MOVE_FORWARD":
        from app.core.state import gesture_controller
        gesture_controller.publish_robot_command(physical_robot_id, "ELBOW LEFT")
        return
    elif action == "MOVE_BACKWARD":
        from app.core.state import gesture_controller
        gesture_controller.publish_robot_command(physical_robot_id, "ELBOW RIGHT")
        return
    elif action == "MOVE_UP":
        from app.core.state import gesture_controller
        gesture_controller.publish_robot_command(physical_robot_id, "SHOULDER LEFT")
        return
    elif action == "MOVE_DOWN":
        from app.core.state import gesture_controller
        gesture_controller.publish_robot_command(physical_robot_id, "SHOULDER RIGHT")
        return
    elif action == "MOVE_HOME":
        subtopic = "home"
        from app.core.state import gesture_controller
        if hasattr(gesture_controller, "_angle_cache") and physical_robot_id in gesture_controller._angle_cache:
            gesture_controller._angle_cache[physical_robot_id] = {
                0: 90.0,
                1: 100.0,
                2: 60.0,
                3: 90.0,
            }
    elif action == "STOP_ALL":
        subtopic = "estop"
    else:
        print(f"[Voice] Action {action} not mapping to standard MQTT topic. Skipping.", flush=True)
        return

    topic = f"robot/{physical_robot_id}/commands/{subtopic}"
    message = json.dumps(payload)

    try:
        client = mqtt.Client()
        if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
            client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
        client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
        client.publish(topic, message)
        client.disconnect()
        print(f"[Voice] Executed {action} → {topic}: {message}", flush=True)
    except Exception as e:
        print(f"[Voice] MQTT publish failed: {e}", flush=True)


from app.core.state import voice_processor

@router.post("/voice/test")
def test_voice_command(
    robot_id: str = Form(None),
    db: Session = Depends(get_db)
):
    """Simulation endpoint — picks a random command, no audio needed."""
    commands = get_voice_commands_db(db)
    if not commands:
        raise HTTPException(status_code=400, detail="No voice commands configured")
    result = voice_processor.parse_intent(commands)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to simulate intent")
    
    # Execute the simulated action physically
    if result.get("intent") != "NO_MATCH":
        physical_id = resolve_physical_robot_id(robot_id, db)
        matched_phrase = result.get("phrase")
        matched_cmd = next((c for c in commands if c.phrase == matched_phrase), None)
        if matched_cmd:
            execute_voice_action(physical_id, matched_cmd.action, matched_cmd.target)

    return result

@router.post("/voice/transcribe")
async def transcribe_voice(
    audio: UploadFile = File(...),
    robot_id: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Real Whisper transcription endpoint.
    Accepts an audio file (WAV, MP3, WebM, OGG etc.), transcribes it with
    Whisper, then matches the transcript against the configured DB commands.
    """
    commands = get_voice_commands_db(db)
    audio_bytes = await audio.read()
    result = voice_processor.match_intent(audio_bytes, commands)
    if result is None:
        raise HTTPException(status_code=503, detail="Whisper model not available")
    
    # If intent matches a valid command, execute it physically
    if result.get("intent") != "NO_MATCH":
        physical_id = resolve_physical_robot_id(robot_id, db)
        matched_phrase = result.get("phrase")
        matched_cmd = next((c for c in commands if c.phrase == matched_phrase), None)
        if matched_cmd:
            execute_voice_action(physical_id, matched_cmd.action, matched_cmd.target)

    return result

@router.post("/voice/execute")
def execute_command_endpoint(payload: dict, db: Session = Depends(get_db)):
    """Directly execute a voice action physically (for simplified real-time browser controls)."""
    action = payload.get("action")
    target = payload.get("target", "any")
    robot_id = payload.get("robot_id")
    
    physical_id = resolve_physical_robot_id(robot_id, db)
    execute_voice_action(physical_id, action, target)
    return {"status": "executed", "action": action, "target": target}
