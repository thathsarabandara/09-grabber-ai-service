from sqlalchemy import Column, Integer, String, Boolean
from app.core.db import Base

class GestureMapping(Base):
    __tablename__ = "gesture_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    gesture_name = Column(String(100), unique=True, nullable=False)
    action = Column(String(100), nullable=False, default="UNASSIGNED")

class GestureSetting(Base):
    __tablename__ = "gesture_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    control_enabled = Column(Boolean, default=False)
    safety_enabled = Column(Boolean, default=True)
