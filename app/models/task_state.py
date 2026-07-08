from sqlalchemy import Column, Integer, String
from app.core.db import Base

class TaskState(Base):
    __tablename__ = "task_states"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(50), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="idle")
