from fastapi import APIRouter
import socket
from urllib.parse import urlparse

from app.core.state import object_detector

router = APIRouter()


def check_camera_connected(camera_url: str) -> str:
    """TCP-connect to the camera host/port to verify reachability."""
    try:
        parsed = urlparse(camera_url)
        host   = parsed.hostname or "192.168.1.105"
        port   = parsed.port or 81
        with socket.create_connection((host, port), timeout=2):
            return "connected"
    except Exception:
        return "disconnected"


@router.get("/status")
def get_status(camera_url: str = None):
    """
    Returns current AI engine status:
      - YOLO model name and live FPS/detection metrics
      - Camera reachability check
    """
    target_url = camera_url or "http://192.168.1.105:81/stream"
    cam_status = check_camera_connected(target_url)

    model_name = "YOLO"
    if object_detector.model_path:
        base = object_detector.model_path.split("/")[-1]
        model_name = base.replace(".pt", "").replace(".onnx", "").upper()

    return {
        "model_status":    model_name,
        "detection_fps":   object_detector.last_fps,
        "detected_today":  object_detector.detected_count,
        "precision":       object_detector.precision,
        "camera_status":   cam_status,
    }
