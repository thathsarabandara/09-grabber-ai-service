from sqlalchemy import Column, Integer, String
from app.core.db import Base

class SortingCategory(Base):
    __tablename__ = "sorting_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

class SortingRule(Base):
    __tablename__ = "sorting_rules"
    
    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String(100), unique=True, nullable=False)
    bin_name = Column(String(100), nullable=False)
