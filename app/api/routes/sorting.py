from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models import SortingCategory, SortingRule
import random

router = APIRouter()

def get_sorting_categories_db(db: Session):
    categories = db.query(SortingCategory).all()
    if not categories:
        defaults = ['Plastic', 'Metal', 'Electronics', 'Tools', 'Stationery']
        for name in defaults:
            db.add(SortingCategory(name=name))
        db.commit()
        categories = db.query(SortingCategory).all()
    return [c.name for c in categories]

def get_sorting_rules_db(db: Session):
    rules = db.query(SortingRule).all()
    if not rules:
        defaults = [
            {"object": "Bottle", "bin": "Plastic"},
            {"object": "Phone", "bin": "Electronics"},
            {"object": "Cube", "bin": "Metal"}
        ]
        for item in defaults:
            db.add(SortingRule(object_name=item["object"], bin_name=item["bin"]))
        db.commit()
        rules = db.query(SortingRule).all()
    return rules

@router.get("/sorting/settings")
def get_settings(db: Session = Depends(get_db)):
    categories = get_sorting_categories_db(db)
    rules = get_sorting_rules_db(db)
    return {
        "categories": categories,
        "rules": [
            {"id": r.id, "object": r.object_name, "bin": r.bin_name}
            for r in rules
        ]
    }

@router.post("/sorting/categories")
def add_category(payload: dict, db: Session = Depends(get_db)):
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Category name is required")
    name = name.strip().capitalize()
    
    existing = db.query(SortingCategory).filter(SortingCategory.name == name).first()
    if not existing:
        db.add(SortingCategory(name=name))
        db.commit()
        
    return get_settings(db)

@router.delete("/sorting/categories/{name}")
def delete_category(name: str, db: Session = Depends(get_db)):
    category = db.query(SortingCategory).filter(SortingCategory.name == name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
        
    # Delete cascade rules referencing this category bin
    db.query(SortingRule).filter(SortingRule.bin_name == name).delete()
    db.delete(category)
    db.commit()
    
    return get_settings(db)

@router.post("/sorting/rules")
def add_rule(payload: dict, db: Session = Depends(get_db)):
    object_name = payload.get("object_name")
    bin_name = payload.get("bin_name")
    if not object_name or not bin_name:
        raise HTTPException(status_code=400, detail="Object name and bin name are required")
        
    object_name = object_name.strip().capitalize()
    
    # Check if category exists
    category = db.query(SortingCategory).filter(SortingCategory.name == bin_name).first()
    if not category:
        raise HTTPException(status_code=400, detail=f"Bin category '{bin_name}' does not exist")
        
    existing = db.query(SortingRule).filter(SortingRule.object_name == object_name).first()
    if existing:
        existing.bin_name = bin_name
    else:
        db.add(SortingRule(object_name=object_name, bin_name=bin_name))
    db.commit()
    
    return get_settings(db)

@router.delete("/sorting/rules/{rule_id}")
def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.query(SortingRule).filter(SortingRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Sorting rule not found")
    db.delete(rule)
    db.commit()
    
    return get_settings(db)

from app.core.state import sorting_assistant

@router.post("/sorting/simulate")
def simulate_sorting(db: Session = Depends(get_db)):
    rules = get_sorting_rules_db(db)
    if not rules:
        raise HTTPException(status_code=400, detail="No sorting rules configured")
        
    result = sorting_assistant.run_sorting_step(rules)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to run sorting step")
    return result
