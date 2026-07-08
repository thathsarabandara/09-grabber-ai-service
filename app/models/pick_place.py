from sqlalchemy import Column, Integer, String, Float
from app.core.db import Base

class PickPlaceSetting(Base):
    __tablename__ = "pick_place_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    selection_rule = Column(String(100), default="Highest Priority")
    pick_strategy = Column(String(100), default="Top-Down Vertical")
    pick_min_x = Column(Float, default=-100.0)
    pick_max_x = Column(Float, default=100.0)
    drop_min_x = Column(Float, default=150.0)
    drop_max_x = Column(Float, default=250.0)

class GraspForce(Base):
    __tablename__ = "grasp_forces"
    
    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String(100), unique=True, nullable=False)
    force_level = Column(String(50), default="Medium")
