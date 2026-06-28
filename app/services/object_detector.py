import cv2
import numpy as np
import os
import time

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.last_fps = 0.0
        self.detected_count = 0
        self.precision = 98.5
        self._load_model()

    def _load_model(self):
        """Attempt to load the YOLO model from the standard paths."""
        candidate_paths = [
            "/app/yolo11x.pt",   # full model (mounted via docker-compose .:/app)
            "/app/yolo11l.pt",   # large model fallback
            "yolo11x.pt",        # relative path fallback
            "yolo11l.pt",
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(path)
                    self.model_path = path
                    print(f"[ObjectDetector] Loaded YOLO model from: {path}")
                    return
                except Exception as e:
                    print(f"[ObjectDetector] Failed to load model at {path}: {e}")
                    continue

        # Fallback: let Ultralytics auto-download yolo11n (smallest)
        try:
            from ultralytics import YOLO
            print("[ObjectDetector] No local model found — downloading yolo11n.pt as fallback…")
            self.model = YOLO("yolo11n.pt")
            self.model_path = "yolo11n.pt"
        except Exception as e:
            print(f"[ObjectDetector] Failed to load any YOLO model: {e}")
            self.model = None

    def process_frame(
        self,
        frame,
        conf_threshold: float = 0.60,
        allowed_classes: list = None,
        allowed_zones: list = None
    ):
        """
        Runs YOLO inference on the frame and draws bounding boxes with labels.
        Returns the annotated frame. Falls back to the raw frame if model is unavailable.
        """
        if self.model is None or frame is None:
            return frame

        t0 = time.time()
        results = self.model(frame, conf=conf_threshold, verbose=False)
        elapsed = time.time() - t0
        self.last_fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0

        h, w = frame.shape[:2]
        detected_this_frame = 0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, "Unknown")

                # Filter by allowed class names if specified
                if allowed_classes and label not in allowed_classes:
                    continue

                # Zone filtering: divide frame into left / center / right thirds
                if allowed_zones:
                    x1, _, x2, _ = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    if cx < w / 3:
                        zone = "left"
                    elif cx < 2 * w / 3:
                        zone = "center"
                    else:
                        zone = "right"
                    if zone not in allowed_zones:
                        continue

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = (59, 130, 246)  # Blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                caption = f"{label} {int(conf * 100)}%"
                cv2.putText(
                    frame, caption,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                detected_this_frame += 1

        self.detected_count += detected_this_frame
        return frame
