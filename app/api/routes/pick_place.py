from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models import PickPlaceSetting, GraspForce

router = APIRouter()

def get_pick_place_settings_db(db: Session):
    setting = db.query(PickPlaceSetting).first()
    if not setting:
        setting = PickPlaceSetting(
            selection_rule="Highest Priority",
            pick_strategy="Top-Down Vertical",
            pick_min_x=-100.0,
            pick_max_x=100.0,
            drop_min_x=150.0,
            drop_max_x=250.0
        )
        db.add(setting)
        db.commit()
        db.refresh(setting)
    return setting

def get_grasp_forces_db(db: Session):
    forces = db.query(GraspForce).all()
    if not forces:
        defaults = {
            "Bottle": "Medium",
            "Cube": "High",
            "Cup": "Low",
            "Phone": "Medium"
        }
        for name, force in defaults.items():
            db.add(GraspForce(class_name=name, force_level=force))
        db.commit()
        forces = db.query(GraspForce).all()
    return {f.class_name: f.force_level for f in forces}

@router.get("/pick-place/settings")
def get_settings(db: Session = Depends(get_db)):
    setting = get_pick_place_settings_db(db)
    forces = get_grasp_forces_db(db)
    return {
        "selection_rule": setting.selection_rule,
        "pick_strategy": setting.pick_strategy,
        "workspace_coords": {
            "pickMinX": setting.pick_min_x,
            "pickMaxX": setting.pick_max_x,
            "dropMinX": setting.drop_min_x,
            "dropMaxX": setting.drop_max_x
        },
        "grasp_forces": forces
    }

@router.post("/pick-place/settings")
def update_settings(payload: dict, db: Session = Depends(get_db)):
    setting = get_pick_place_settings_db(db)
    
    if "selection_rule" in payload:
        setting.selection_rule = payload["selection_rule"]
    if "pick_strategy" in payload:
        setting.pick_strategy = payload["pick_strategy"]
        
    coords = payload.get("workspace_coords", {})
    if "pickMinX" in coords:
        setting.pick_min_x = float(coords["pickMinX"])
    if "pickMaxX" in coords:
        setting.pick_max_x = float(coords["pickMaxX"])
    if "dropMinX" in coords:
        setting.drop_min_x = float(coords["dropMinX"])
    if "dropMaxX" in coords:
        setting.drop_max_x = float(coords["dropMaxX"])
        
    forces = payload.get("grasp_forces", {})
    for name, force in forces.items():
        existing = db.query(GraspForce).filter(GraspForce.class_name == name).first()
        if existing:
            existing.force_level = force
        else:
            db.add(GraspForce(class_name=name, force_level=force))
            
    db.commit()
    db.refresh(setting)
    
    return {
        "status": "success",
        "selection_rule": setting.selection_rule,
        "pick_strategy": setting.pick_strategy,
        "workspace_coords": {
            "pickMinX": setting.pick_min_x,
            "pickMaxX": setting.pick_max_x,
            "dropMinX": setting.drop_min_x,
            "dropMaxX": setting.drop_max_x
        },
        "grasp_forces": get_grasp_forces_db(db)
    }

from app.core.state import pick_place_simulator

@router.post("/pick-place/simulate")
def simulate_path(db: Session = Depends(get_db)):
    setting = get_pick_place_settings_db(db)
    forces = get_grasp_forces_db(db)
    
    cube_force = forces.get("Cube", "Medium")
    
    res = pick_place_simulator.run_simulation(
        strategy=setting.pick_strategy,
        selection_rule=setting.selection_rule,
        pick_min_x=setting.pick_min_x,
        pick_max_x=setting.pick_max_x,
        drop_min_x=setting.drop_min_x,
        drop_max_x=setting.drop_max_x,
        cube_force=cube_force
    )
    return res
