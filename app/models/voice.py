from sqlalchemy import Column, Integer, String
from app.core.db import Base

class VoiceCommand(Base):
    __tablename__ = "voice_commands"
    
    id = Column(Integer, primary_key=True, index=True)
    phrase = Column(String(255), unique=True, nullable=False)
    action = Column(String(100), nullable=False)
    target = Column(String(100), nullable=True)

class VoiceSetting(Base):
    __tablename__ = "voice_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    wake_word = Column(String(100), default="Hey Robot")
    language = Column(String(100), default="English (US)")
