import os
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List

from app.core.db import get_db
from app.models import Operator
from app.core.state import face_recognizer

router = APIRouter()

@router.post("/face/register")
async def register_face(
    files: List[UploadFile] = File(...),
    name: str = Form(...),
    access_level: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Accepts operator face snapshots, processes and resizes them,
    registers them in the cv_engine templates, and saves operator records to DB.
    """
    try:
        saved_filenames = []
        for file in files:
            image_bytes = await file.read()
            saved_path = face_recognizer.register_face(image_bytes, name)
            if saved_path:
                # Store only the filename so the frontend can build the URL via the gateway
                saved_filenames.append(os.path.basename(saved_path))
            
        if not saved_filenames:
            raise Exception("No valid faces detected in the provided images")
        
        # Save to DB with just the filename as the face_path reference
        operator = Operator(name=name, face_path=saved_filenames[0], access_level=access_level)
        db.add(operator)
        db.commit()
        db.refresh(operator)
        
        return {
            "status": "success", 
            "id": operator.id, 
            "name": operator.name, 
            "face_path": operator.face_path,  # just the filename
            "access_level": operator.access_level,
            "samples_added": len(saved_filenames)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/face/operators")
def list_operators(db: Session = Depends(get_db)):
    """Retrieve all registered operators."""
    operators = db.query(Operator).all()
    return operators

@router.put("/face/operators/{operator_id}")
def update_operator(operator_id: int, payload: dict, db: Session = Depends(get_db)):
    """Update operator details (name, access_level)."""
    operator = db.query(Operator).filter(Operator.id == operator_id).first()
    if not operator:
        raise HTTPException(status_code=404, detail="Operator not found")
        
    new_name = payload.get("name", operator.name)
    new_access = payload.get("access_level", operator.access_level)
    
    if new_name != operator.name:
        face_recognizer.rename_operator(operator.name, new_name)
        # face_path is just the filename — update only the name prefix in the filename
        safe_old = operator.name.lower().replace(' ', '-')
        safe_new = new_name.lower().replace(' ', '-')
        if operator.face_path and safe_old + '_' in operator.face_path:
            operator.face_path = operator.face_path.replace(safe_old + '_', safe_new + '_', 1)
        operator.name = new_name
        
    operator.access_level = new_access
    db.commit()
    db.refresh(operator)
    return operator

@router.delete("/face/operators/{operator_id}")
def delete_operator(operator_id: int, db: Session = Depends(get_db)):
    """Delete an operator and all their template files."""
    operator = db.query(Operator).filter(Operator.id == operator_id).first()
    if not operator:
        raise HTTPException(status_code=404, detail="Operator not found")
        
    face_recognizer.remove_operator(operator.name)
    db.delete(operator)
    db.commit()
    return {"status": "success", "message": "Operator deleted"}


@router.post("/face/retrain")
def retrain_model(db: Session = Depends(get_db)):
    """
    'Retraining' for this ChromaDB-backed recogniser means verifying how many
    face embeddings are stored and returning readiness stats.
    Since ChromaDB uses real-time vector similarity, there is no offline training
    step — every newly registered face is immediately queryable.
    """
    try:
        count = face_recognizer._collection.count()
        operators = db.query(Operator).all()
        return {
            "status": "ready",
            "embedding_count": count,
            "operator_count": len(operators),
            "message": f"Face recognition model is ready with {count} embedding(s) across {len(operators)} operator(s)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/face/stats")
def face_stats(db: Session = Depends(get_db)):
    """Return quick stats about the face recognition model readiness."""
    try:
        count = face_recognizer._collection.count()
        operators = db.query(Operator).all()
        return {
            "embedding_count": count,
            "operator_count": len(operators),
            "ready": count > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
