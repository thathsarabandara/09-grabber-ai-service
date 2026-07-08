from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    import app.models.operator  # noqa: F401
    import app.models.gesture   # noqa: F401
    import app.models.voice     # noqa: F401
    import app.models.pick_place  # noqa: F401
    import app.models.sorting   # noqa: F401
    import app.models.task_state  # noqa: F401
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
