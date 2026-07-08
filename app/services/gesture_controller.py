"""
GestureController (Dual-Mode) — MediaPipe Tasks API
=====================================================
Uses the new mediapipe 0.10+ Tasks-based HandLandmarker API.

Two detection modes running simultaneously on every frame:

Mode 1 — Motion Tracking (built-in, always active, no registration)
────────────────────────────────────────────────────────────────────
  Hand moves Left/Right         → BASE LEFT / BASE RIGHT
  Hand moves Up/Down            → SHOULDER LEFT / SHOULDER RIGHT
  Hand moves Forward/Backward   → ELBOW LEFT / ELBOW RIGHT
  Open Palm  (≥4 fingers)       → OPEN GRIP
  Closed Fist (≤1 finger)       → CLOSE GRIP

Mode 2 — Custom Pose Registration (optional, ChromaDB-backed)
──────────────────────────────────────────────────────────────
  Users capture static finger poses → 63-dim landmark vector → ChromaDB.
  Registered poses take priority over built-in motion detection.

Priority: Custom Pose > Static Finger Pose > Motion Direction
"""

import cv2
import numpy as np
import os
import uuid
import threading
from collections import deque
from typing import Optional

from app.core.config import settings


# ── Tuning ────────────────────────────────────────────────────────────────────

MATCH_THRESHOLD     = 0.25   # ChromaDB cosine distance threshold
MIN_DISP            = 0.03   # Min normalised wrist displacement for motion
N_HISTORY           = 6      # Frames in rolling motion window
DEPTH_DISP          = 0.025  # Palm-size change threshold for elbow
FINGER_OPEN_THRESH  = 4      # ≥N extended fingers → OPEN GRIP
FINGER_CLOSE_THRESH = 1      # ≤N extended fingers → CLOSE GRIP

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          "models", "hand_landmarker.task")


# ── Per-session motion tracker ─────────────────────────────────────────────────

class _MotionTracker:
    def __init__(self):
        self.history: deque = deque(maxlen=N_HISTORY)

    def push(self, wrist_x: float, wrist_y: float, palm_size: float):
        self.history.append((wrist_x, wrist_y, palm_size))

    def detect_motion(self) -> Optional[str]:
        if len(self.history) < N_HISTORY:
            return None
        x0, y0, s0 = self.history[0]
        x1, y1, s1 = self.history[-1]
        dx, dy, ds = x1 - x0, y1 - y0, s1 - s0

        # Dominant axis motion classification
        mag_x = abs(dx)
        mag_y = abs(dy)
        mag_s = abs(ds)

        # Check if depth change (moving toward/away from camera) is the dominant motion
        if mag_s >= DEPTH_DISP and mag_s > mag_x and mag_s > mag_y:
            return "ELBOW LEFT" if ds > 0 else "ELBOW RIGHT"

        if mag_x >= mag_y:
            if mag_x >= MIN_DISP:
                return "BASE LEFT" if dx < 0 else "BASE RIGHT"
        else:
            if mag_y >= MIN_DISP:
                return "SHOULDER LEFT" if dy < 0 else "SHOULDER RIGHT"
        return None

    def reset(self):
        self.history.clear()


# ── Landmark wrapper (adapts Tasks API output to legacy interface) ─────────────

class _LandmarkAdapter:
    """Wraps a mediapipe NormalizedLandmark to match the old lm.x/lm.y/lm.z API."""
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z


# ── Main controller ────────────────────────────────────────────────────────────

class GestureController:

    def __init__(self):
        self._landmarker  = None
        self._chroma      = None
        self._collection  = None
        self._trackers: dict[str, _MotionTracker] = {}
        self._lock = threading.Lock()
        self._load_mediapipe()
        self._load_vectordb()

    # ── Init ────────────────────────────────────────────────────────────────

    def _load_mediapipe(self):
        try:
            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            if not os.path.exists(MODEL_PATH):
                print(f"[GestureController] Model file not found: {MODEL_PATH}")
                return

            base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=settings.MEDIAPIPE_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=settings.MEDIAPIPE_TRACKING_CONFIDENCE,
                min_tracking_confidence=settings.MEDIAPIPE_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.IMAGE
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            print("[GestureController] MediaPipe HandLandmarker loaded (dual-mode, Tasks API).")
        except Exception as e:
            print(f"[GestureController] MediaPipe load failed: {e}")
            import traceback
            traceback.print_exc()

    def _load_vectordb(self):
        try:
            import chromadb
            db_dir = settings.GESTURE_DB_DIR
            os.makedirs(db_dir, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=db_dir)
            self._collection = self._chroma.get_or_create_collection(
                name="gesture_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            count = self._collection.count()
            print(f"[GestureController] ChromaDB ready — {count} custom pose sample(s).")
        except Exception as e:
            print(f"[GestureController] ChromaDB load failed: {e}")

    def _get_tracker(self, session_id: str) -> _MotionTracker:
        with self._lock:
            if session_id not in self._trackers:
                self._trackers[session_id] = _MotionTracker()
            return self._trackers[session_id]

    # ── Run detection ───────────────────────────────────────────────────────

    def _detect_hand(self, frame_bgr):
        """
        Run MediaPipe HandLandmarker on a BGR frame.
        Returns list of _LandmarkAdapter objects (21 landmarks) or None.
        """
        if self._landmarker is None:
            return None
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        # Wrap to match legacy lm.x/lm.y/lm.z API
        return [_LandmarkAdapter(lm) for lm in result.hand_landmarks[0]]

    # ── Landmark helpers ────────────────────────────────────────────────────

    def _embed(self, landmarks) -> list:
        """63-dim normalised pose vector (translation + scale invariant)."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        pts -= pts[0]
        palm_size = np.linalg.norm(pts[9])
        if palm_size > 1e-6:
            pts /= palm_size
        flat = pts.flatten()
        norm = np.linalg.norm(flat)
        if norm > 1e-6:
            flat /= norm
        return flat.tolist()

    @staticmethod
    def _count_extended_fingers(landmarks) -> int:
        extended = 0
        def dist(a, b):
            return ((landmarks[a].x - landmarks[b].x)**2 + (landmarks[a].y - landmarks[b].y)**2)**0.5
        # Thumb: MCP (2) to TIP (4) vs MCP (2) to IP (3)
        if dist(4, 2) > dist(3, 2) * 1.3:
            extended += 1
        # Other fingers: MCP to TIP vs MCP to PIP
        fingers = [
            (8, 6, 5),   # Index: TIP, PIP, MCP
            (12, 10, 9),  # Middle: TIP, PIP, MCP
            (16, 14, 13), # Ring: TIP, PIP, MCP
            (20, 18, 17)  # Pinky: TIP, PIP, MCP
        ]
        for tip, pip, mcp in fingers:
            if dist(tip, mcp) > dist(pip, mcp) * 1.3:
                extended += 1
        return extended

    def _query_custom_pose(self, vector: list) -> Optional[str]:
        if self._collection is None or self._collection.count() == 0:
            return None
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=1,
            include=["metadatas", "distances"]
        )
        if result["ids"] and result["ids"][0]:
            distance = result["distances"][0][0]
            if distance < MATCH_THRESHOLD:
                return result["metadatas"][0][0]["gesture_name"]
        return None

    # ── Draw hand skeleton manually ─────────────────────────────────────────

    @staticmethod
    def _draw_landmarks(frame, landmarks):
        """Draw the 21 hand landmarks and connections on frame."""
        h, w = frame.shape[:2]

        # MediaPipe hand connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),       # thumb
            (0,5),(5,6),(6,7),(7,8),       # index
            (5,9),(9,10),(10,11),(11,12),  # middle
            (9,13),(13,14),(14,15),(15,16),# ring
            (13,17),(17,18),(18,19),(19,20),# pinky
            (0,17)                         # palm base
        ]

        # Draw connections
        for start, end in connections:
            pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
            pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 128), 2)

        # Draw landmark dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 2, (0, 128, 255), -1)

    # ── Registration (Mode 2) ───────────────────────────────────────────────

    def register_from_frame(self, jpeg_bytes: bytes, gesture_name: str) -> dict:
        if self._collection is None:
            raise Exception("Gesture vector database not available")
        if self._landmarker is None:
            raise Exception("MediaPipe not available")

        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise Exception("Invalid image data")

        landmarks = self._detect_hand(frame)
        if landmarks is None:
            raise Exception("No hand detected in the provided frame")

        vector    = self._embed(landmarks)
        vector_id = uuid.uuid4().hex[:10]

        self._collection.add(
            ids=[vector_id],
            embeddings=[vector],
            metadatas=[{"gesture_name": gesture_name}]
        )

        existing     = self._collection.get(where={"gesture_name": gesture_name})
        sample_count = len(existing["ids"])

        return {
            "gesture_name": gesture_name,
            "vector_id":    vector_id,
            "sample_count": sample_count,
        }

    def remove_gesture(self, gesture_name: str) -> int:
        if self._collection is None:
            return 0
        rows = self._collection.get(where={"gesture_name": gesture_name})
        if rows["ids"]:
            self._collection.delete(ids=rows["ids"])
            return len(rows["ids"])
        return 0

    def list_gestures(self) -> dict:
        if self._collection is None or self._collection.count() == 0:
            return {}
        rows   = self._collection.get(include=["metadatas"])
        counts = {}
        for meta in rows["metadatas"]:
            name         = meta["gesture_name"]
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ── Live frame processing ───────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        gesture_mappings: dict,
        session_id: str = "default"
    ):
        """
        Dual-mode detection on a single frame.
        Priority: Custom Pose > Motion Direction > Static Finger.
        Returns: (annotated_frame, robot_action | None)
        """
        if self._landmarker is None:
            return frame, None

        h, w     = frame.shape[:2]
        tracker  = self._get_tracker(session_id)
        landmarks = self._detect_hand(frame)

        detected_action  = None
        detection_source = None

        if landmarks:
            # Draw skeleton
            self._draw_landmarks(frame, landmarks)

            wrist_x  = landmarks[0].x
            wrist_y  = landmarks[0].y
            xs       = [landmarks[i].x for i in range(9)]
            ys       = [landmarks[i].y for i in range(9)]
            palm_diag = ((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2) ** 0.5
            tracker.push(wrist_x, wrist_y, palm_diag)

            # Priority 1: Custom ChromaDB pose
            vector      = self._embed(landmarks)
            custom_pose = self._query_custom_pose(vector)
            if custom_pose:
                mapped = gesture_mappings.get(custom_pose)
                if mapped:
                    detected_action  = mapped
                    detection_source = f"Custom: {custom_pose}"

            # Priority 2: Motion direction
            if not detected_action:
                motion = tracker.detect_motion()
                if motion:
                    detected_action  = motion
                    detection_source = "Motion"

            # Priority 3: Static finger pose (Open Palm / Closed Fist)
            if not detected_action:
                extended = self._count_extended_fingers(landmarks)
                if extended >= FINGER_OPEN_THRESH:
                    detected_action  = "OPEN GRIP"
                    detection_source = f"Palm ({extended} fingers)"
                elif extended <= FINGER_CLOSE_THRESH:
                    detected_action  = "CLOSE GRIP"
                    detection_source = f"Fist ({extended} fingers)"

            # Annotate
            lx = int(wrist_x * w)
            ly = int(wrist_y * h)

            if detected_action:
                color = (16, 185, 129)
                label = f"{detected_action}"
                src_label = detection_source or ""
            else:
                color = (180, 180, 180)
                label = "Detecting..."
                src_label = ""

            cv2.putText(frame, label,
                (max(lx - 80, 0), max(ly - 22, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if src_label:
                cv2.putText(frame, src_label,
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # Motion trail
            history = list(tracker.history)
            for i in range(1, len(history)):
                p1 = (int(history[i-1][0] * w), int(history[i-1][1] * h))
                p2 = (int(history[i][0]   * w), int(history[i][1]   * h))
                alpha = int(80 + 175 * i / len(history))
                cv2.line(frame, p1, p2, (alpha, 100, 255), 2)
        else:
            tracker.reset()
            cv2.putText(frame, "No hand detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        return frame, detected_action

    # ── MQTT dispatch ───────────────────────────────────────────────────────

    def publish_robot_command(self, robot_id: str, action: str):
        if not robot_id or not action:
            return

        # Resolve database UUID robot_id to physical robot_id (e.g. GRABBER-V1-ESP32)
        physical_robot_id = robot_id
        import uuid
        try:
            val = uuid.UUID(str(robot_id))
            from app.core.db import SessionLocal
            from sqlalchemy import text
            db = SessionLocal()
            try:
                res = db.execute(
                    text("SELECT robot_id FROM grabber_robot.robots WHERE id = :hex OR id = :str"),
                    {"hex": val.hex, "str": str(val)}
                )
                row = res.fetchone()
                if row:
                    physical_robot_id = row[0]
            except Exception as db_err:
                print(f"[GestureController] Failed to resolve robot UUID: {db_err}", flush=True)
            finally:
                db.close()
        except ValueError:
            pass

        import paho.mqtt.client as mqtt
        import json

        # Physical limits from firmware Config.h:
        # Base (0): 1 to 180 (home 90)
        # Shoulder (1): 50 to 150 (home 100)
        # Elbow (2): 20 to 100 (home 60)
        # Gripper (3): 70 to 100 (home 90)
        LIMITS = {
            0: (1, 180),
            1: (50, 150),
            2: (20, 100),
            3: (70, 100),
        }

        if not hasattr(self, "_angle_cache"):
            self._angle_cache = {}

        if physical_robot_id not in self._angle_cache:
            self._angle_cache[physical_robot_id] = {
                0: 90.0,
                1: 100.0,
                2: 60.0,
                3: 90.0,
            }

        # Check gripper state commands
        if action == "OPEN GRIP":
            self._angle_cache[physical_robot_id][3] = 100.0
            subtopic = "open-gripper"
            payload = {}
        elif action == "CLOSE GRIP":
            self._angle_cache[physical_robot_id][3] = 70.0
            subtopic = "close-gripper"
            payload = {}
        else:
            # Joint index and direction steps
            action_map = {
                "BASE LEFT":      (0, -5.0),
                "BASE RIGHT":     (0, 5.0),
                "SHOULDER LEFT":  (1, 5.0),   # Hand moves Up -> Shoulder Up
                "SHOULDER RIGHT": (1, -5.0),  # Hand moves Down -> Shoulder Down
                "ELBOW LEFT":     (2, 5.0),   # Hand moves Forward -> Elbow Extend
                "ELBOW RIGHT":    (2, -5.0),  # Hand moves Backward -> Elbow Retract
            }

            if action not in action_map:
                return

            servo_idx, step = action_map[action]
            curr_angle = self._angle_cache[physical_robot_id].get(servo_idx, 90.0)
            new_angle = curr_angle + step

            # Clamp target angle within safe limits
            min_val, max_val = LIMITS[servo_idx]
            new_angle = max(min_val, min(max_val, new_angle))
            self._angle_cache[physical_robot_id][servo_idx] = new_angle

            subtopic = "move"
            payload = {
                "servo": servo_idx,
                "angle": float(new_angle)
            }

        topic   = f"robot/{physical_robot_id}/commands/{subtopic}"
        message = json.dumps(payload)

        try:
            client = mqtt.Client()
            if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
                client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
            client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, 60)
            client.publish(topic, message)
            client.disconnect()
            print(f"[GestureController] {action} → {topic}: {message}", flush=True)
        except Exception as e:
            print(f"[GestureController] MQTT publish failed: {e}", flush=True)

    # ── Status ──────────────────────────────────────────────────────────────

    def classify_frame(self, jpeg_bytes: bytes) -> Optional[str]:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        landmarks = self._detect_hand(frame)
        if landmarks is None:
            return None
        return self._query_custom_pose(self._embed(landmarks))

    def train_wizard(self) -> dict:
        gestures = self.list_gestures()
        return {
            "status":  "success" if self._landmarker else "error",
            "message": (
                f"Dual-mode detection active. "
                f"{len(gestures)} custom pose(s) registered."
                if self._landmarker else
                "MediaPipe HandLandmarker not loaded."
            ),
            "mediapipe_ready":      self._landmarker is not None,
            "vectordb_ready":       self._collection is not None,
            "detection_confidence": settings.MEDIAPIPE_DETECTION_CONFIDENCE,
            "tracking_confidence":  settings.MEDIAPIPE_TRACKING_CONFIDENCE,
            "registered_gestures":  gestures,
            "motion_detection":     True,
        }
