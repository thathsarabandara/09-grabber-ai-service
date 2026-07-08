from sqlalchemy import Column, Integer, String, DateTime
from app.core.db import Base
import datetime

class Operator(Base):
    __tablename__ = "operators"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    face_path = Column(String(255), nullable=False)
    access_level = Column(String(100), nullable=False, default="Level 1 - Basic Operator")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
