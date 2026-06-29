import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Grabber AI Service"
    DATABASE_URL: str = "mysql+pymysql://user:password@localhost/dbname"
    CHROMA_DB_DIR: str = "/data/vectordb/faces"
    UPLOAD_DIR: str = "/data/uploads/operators"
    GESTURE_DB_DIR: str = "/data/vectordb/gestures"
    WHISPER_MODEL: str = "base"
    MEDIAPIPE_DETECTION_CONFIDENCE: float = 0.7
    MEDIAPIPE_TRACKING_CONFIDENCE: float = 0.6

    MQTT_BROKER: str = "grabber-mqtt-server"
    MQTT_PORT: int = 1883
    MQTT_USERNAME: str = "thathsara"
    MQTT_PASSWORD: str = "BandaPutha"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

_in_docker = os.path.exists("/.dockerenv")

if not _in_docker and "@db/" in settings.DATABASE_URL:
    settings.DATABASE_URL = settings.DATABASE_URL.replace("@db/", "@127.0.0.1/")

if not _in_docker:
    _base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    settings.CHROMA_DB_DIR = os.path.join(_base, "vectordb", "faces")
    settings.UPLOAD_DIR = os.path.join(_base, "uploads", "operators")
    settings.GESTURE_DB_DIR = os.path.join(_base, "vectordb", "gestures")
