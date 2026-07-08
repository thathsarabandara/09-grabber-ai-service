from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.state import task_metadata
from app.models.task_state import TaskState

router = APIRouter()

TASK_IDS = list(task_metadata.keys())


def _ensure_tasks_seeded(db: Session):
    """Seed the task_states table with all known tasks if they don't exist yet."""
    for task_id in TASK_IDS:
        existing = db.query(TaskState).filter(TaskState.task_id == task_id).first()
        if not existing:
            db.add(TaskState(task_id=task_id, status="idle"))
    db.commit()


@router.get("/tasks")
def list_tasks(db: Session = Depends(get_db)):
    """Retrieve all AI tasks and their current active/idle statuses from the DB."""
    _ensure_tasks_seeded(db)
    rows = db.query(TaskState).all()
    # Build a quick lookup
    status_map = {row.task_id: row.status for row in rows}

    response = []
    for task_id, meta in task_metadata.items():
        response.append({
            "id": task_id,
            "title": meta["title"],
            "status": status_map.get(task_id, "idle"),
            "accuracy": meta["accuracy"],
            "latency": meta["latency"]
        })
    return response


@router.post("/tasks/start")
def start_task(payload: dict, db: Session = Depends(get_db)):
    task_id = payload.get("task_id")
    if task_id not in TASK_IDS:
        raise HTTPException(status_code=400, detail="Invalid task ID")

    _ensure_tasks_seeded(db)
    row = db.query(TaskState).filter(TaskState.task_id == task_id).first()
    row.status = "active"
    db.commit()
    return {"status": "success", "task_id": task_id, "state": "active"}


@router.post("/tasks/stop")
def stop_task(payload: dict, db: Session = Depends(get_db)):
    task_id = payload.get("task_id")
    if task_id not in TASK_IDS:
        raise HTTPException(status_code=400, detail="Invalid task ID")

    _ensure_tasks_seeded(db)
    row = db.query(TaskState).filter(TaskState.task_id == task_id).first()
    row.status = "idle"
    db.commit()
    return {"status": "success", "task_id": task_id, "state": "idle"}


@router.post("/tasks/start-all")
def start_all_tasks(db: Session = Depends(get_db)):
    _ensure_tasks_seeded(db)
    db.query(TaskState).update({TaskState.status: "active"})
    db.commit()
    return {"status": "success", "state": "all_active"}


@router.post("/tasks/stop-all")
def stop_all_tasks(db: Session = Depends(get_db)):
    _ensure_tasks_seeded(db)
    db.query(TaskState).update({TaskState.status: "idle"})
    db.commit()
    return {"status": "success", "state": "all_idle"}
