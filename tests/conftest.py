import os
import sys
import pytest
from unittest.mock import MagicMock

# 1. Stub heavy machine learning/computer vision libraries in sys.modules
mock_chroma = MagicMock()
sys.modules["chromadb"] = mock_chroma
sys.modules["chromadb.config"] = mock_chroma.config
sys.modules["cv2"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["mediapipe"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["ffmpeg"] = MagicMock()

# Stub paho-mqtt globally
mock_paho = MagicMock()
sys.modules["paho"] = mock_paho
sys.modules["paho.mqtt"] = mock_paho.mqtt
sys.modules["paho.mqtt.client"] = mock_paho.mqtt.client

# Set test environment database URL before importing app
os.environ["DATABASE_URL"] = "sqlite:///./test_temp.db"

# Now import SQLAlchemy, SQLModel, FastAPI components
from fastapi.testclient import TestClient
from app.main import app
from app.core.db import get_db, engine, Base
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Sync database initialization: create tables once at start of session
    Base.metadata.create_all(bind=engine)
    yield
    # No drop_all here. The file will be cleaned up in pytest_sessionfinish.

@pytest.fixture(name="db")
def db_fixture():
    # Transactional session for tests
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(name="client")
def client_fixture(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

def pytest_sessionfinish(session, exitstatus):
    db_file = "test_temp.db"
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass
