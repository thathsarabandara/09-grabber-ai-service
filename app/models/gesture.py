from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.db import Base
import datetime


class GestureMapping(Base):
    """Maps a gesture name to a robot action command."""
    __tablename__ = "gesture_mappings"

    id           = Column(Integer, primary_key=True, index=True)
    gesture_name = Column(String(100), unique=True, nullable=False)
    action       = Column(String(100), nullable=False, default="UNASSIGNED")

    # Relationship to custom gesture templates
    templates    = relationship(
        "CustomGestureTemplate",
        back_populates="mapping",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class GestureSetting(Base):
    """Global gesture control settings."""
    __tablename__ = "gesture_settings"

    id              = Column(Integer, primary_key=True, index=True)
    control_enabled = Column(Boolean, default=False)
    safety_enabled  = Column(Boolean, default=True)


class CustomGestureTemplate(Base):
    """
    Tracks custom gesture registration records.
    The actual 63-dim landmark vector is stored in ChromaDB (GESTURE_DB_DIR).
    This table records the name, sample count, and creation time so the
    REST API can list/manage gestures without querying ChromaDB directly.
    """
    __tablename__ = "custom_gesture_templates"

    id           = Column(Integer, primary_key=True, index=True)
    gesture_name = Column(
        String(100),
        ForeignKey("gesture_mappings.gesture_name", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    vector_id    = Column(String(20), nullable=False, unique=True)
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)
    mapping      = relationship("GestureMapping", back_populates="templates")

