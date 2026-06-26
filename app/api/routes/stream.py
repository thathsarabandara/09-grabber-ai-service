from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from typing import List

from app.core.db import SessionLocal
from app.models.task_state import TaskState
from app.core.state import face_recognizer, object_detector

router = APIRouter()


def _get_active_tasks() -> set:
    """Open a short-lived DB session to return the set of currently active task IDs."""
    db = SessionLocal()
    try:
        rows = db.query(TaskState).filter(TaskState.status == "active").all()
        return {row.task_id for row in rows}
    finally:
        db.close()


async def mjpeg_stream_generator(
    camera_url: str,
    conf_threshold: float = 0.60,
    allowed_classes: List[str] = None,
    allowed_zones: List[str] = None
):
    """
    Fetches the raw camera stream, decodes each frame, overlays active computer
    vision outputs (object detection labels, face recognition boxes),
    re-encodes to JPEG, and streams to the client.

    Task states are re-read from the DB every 30 frames (~1s at 30fps) so
    activating/deactivating from the Control Center takes effect quickly without
    hammering the database on every single frame.
    """
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            async with client.stream("GET", camera_url) as response:
                if response.status_code != 200:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nFailed to connect to camera\r\n"
                    return

                buffer = b""
                frame_counter = 0
                active_tasks: set = set()

                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start) if start != -1 else -1
                        if start != -1 and end != -1:
                            jpg_bytes = buffer[start:end + 2]
                            buffer = buffer[end + 2:]

                            # Re-query DB every 30 frames to pick up task state changes
                            if frame_counter % 30 == 0:
                                active_tasks = _get_active_tasks()
                            frame_counter += 1

                            # Decode JPEG frame to OpenCV BGR image
                            nparr = np.frombuffer(jpg_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if frame is not None:
                                # Object detection overlay
                                if "obj-detect" in active_tasks:
                                    frame = object_detector.process_frame(
                                        frame,
                                        conf_threshold=conf_threshold,
                                        allowed_classes=allowed_classes,
                                        allowed_zones=allowed_zones
                                    )

                                # Face recognition overlay
                                if "face-rec" in active_tasks:
                                    frame = face_recognizer.process_frame(frame)

                                # Re-encode back to JPEG
                                _, encoded_img = cv2.imencode(".jpg", frame)
                                frame_bytes = encoded_img.tobytes()
                            else:
                                frame_bytes = jpg_bytes

                            # Yield MJPEG chunk
                            yield (
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n'
                                b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                                frame_bytes + b'\r\n'
                            )
                        else:
                            break
        except Exception as e:
            yield f"--frame\r\nContent-Type: text/plain\r\n\r\nStream error: {str(e)}\r\n".encode()


@router.get("/stream")
def get_annotated_stream(
    camera_url: str,
    conf_threshold: float = 0.60,
    classes: str = None,
    zones: str = None
):
    """Serves the real-time annotated image stream from the ESP32 camera feed."""
    allowed_classes = [c.strip() for c in classes.split(",")] if classes else None
    allowed_zones = [z.strip() for z in zones.split(",")] if zones else None

    return StreamingResponse(
        mjpeg_stream_generator(
            camera_url=camera_url,
            conf_threshold=conf_threshold,
            allowed_classes=allowed_classes,
            allowed_zones=allowed_zones
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
