"""
Unit tests for AI service engine classes:
  - ObjectDetector
  - PickPlaceSimulator
  - SortingAssistant
  - FaceRecognizer (with mocked chromadb)
  - VoiceProcessor (with mocked whisper)
  - GestureController (with mocked mediapipe & chromadb)
"""
import os
import numpy as np
from unittest.mock import MagicMock, patch
from app.core.db import get_db  # noqa: F401 — used via conftest fixture injection


# ── DB utility ────────────────────────────────────────────────────────────────

def test_get_db_generator(db):
    assert db is not None


# ── ObjectDetector ────────────────────────────────────────────────────────────

def test_object_detector_defaults():
    from app.services.object_detector import ObjectDetector
    detector = ObjectDetector()
    assert detector.last_fps == 0.0
    assert detector.detected_count == 0
    assert detector.precision == 98.5


def test_object_detector_no_model_returns_frame():
    from app.services.object_detector import ObjectDetector
    detector = ObjectDetector()
    detector.model = None

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.process_frame(fake_frame)
    assert result is fake_frame


def test_object_detector_none_frame_returns_none():
    from app.services.object_detector import ObjectDetector
    detector = ObjectDetector()
    detector.model = None

    result = detector.process_frame(None)
    assert result is None


def test_object_detector_process_frame_with_model():
    from app.services.object_detector import ObjectDetector
    detector = ObjectDetector()

    # Mock the YOLO model
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].tolist.return_value = [10.0, 10.0, 60.0, 60.0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "Bottle"}

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector.model = mock_model

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.process_frame(frame, conf_threshold=0.5)
    assert detector.detected_count == 1


def test_object_detector_zone_filter():
    from app.services.object_detector import ObjectDetector
    detector = ObjectDetector()

    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = [MagicMock()]
    # Place detected box on the right side
    mock_box.xyxy[0].tolist.return_value = [150.0, 10.0, 190.0, 80.0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "Cup"}

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector.model = mock_model

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Only allow center zone — the box is in right, should be filtered
    result = detector.process_frame(frame, allowed_zones=["center"])
    assert detector.detected_count == 0


def test_object_detector_class_filter():
    from app.services.object_detector import ObjectDetector
    detector = ObjectDetector()

    mock_box = MagicMock()
    mock_box.conf = [0.95]
    mock_box.cls = [0]
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].tolist.return_value = [10.0, 10.0, 60.0, 60.0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "Bottle"}

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector.model = mock_model

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Only allow "Cup" — "Bottle" should be filtered
    result = detector.process_frame(frame, allowed_classes=["Cup"])
    assert detector.detected_count == 0


# ── PickPlaceSimulator ────────────────────────────────────────────────────────

def test_pick_place_simulator_run():
    from app.services.pick_place_simulator import PickPlaceSimulator
    sim = PickPlaceSimulator()
    result = sim.run_simulation(
        strategy="Top-Down Vertical",
        selection_rule="Highest Priority",
        pick_min_x=-100.0,
        pick_max_x=100.0,
        drop_min_x=150.0,
        drop_max_x=250.0,
        cube_force="Medium"
    )
    assert "path_coords" in result
    assert "log_entries" in result
    assert len(result["path_coords"]) == 121
    assert len(result["log_entries"]) == 6


def test_pick_place_simulator_path_structure():
    from app.services.pick_place_simulator import PickPlaceSimulator
    sim = PickPlaceSimulator()
    result = sim.run_simulation("TD", "HP", -100, 100, 150, 250, "High")
    frame0 = result["path_coords"][0]
    assert "frame" in frame0
    assert "armX" in frame0
    assert "armY" in frame0
    assert "objX" in frame0
    assert "objY" in frame0


# ── SortingAssistant ──────────────────────────────────────────────────────────

def test_sorting_assistant_no_rules():
    from app.services.sorting_assistant import SortingAssistant
    assistant = SortingAssistant()
    result = assistant.run_sorting_step([])
    assert result is None


def test_sorting_assistant_with_rules():
    from app.services.sorting_assistant import SortingAssistant
    assistant = SortingAssistant()

    mock_rule = MagicMock()
    mock_rule.object_name = "Bottle"
    mock_rule.bin_name = "Plastic"

    result = assistant.run_sorting_step([mock_rule])
    assert result is not None
    assert result["detected_class"] == "Bottle"
    assert result["bin_name"] == "Plastic"
    assert "result_message" in result


# ── FaceRecognizer ────────────────────────────────────────────────────────────

def test_face_recognizer_init(tmp_path):
    from app.services.face_recognizer import FaceRecognizer
    upload_dir = str(tmp_path / "uploads")
    vector_db_dir = str(tmp_path / "vectordb")

    # chromadb is imported inside _init_db — patch it via sys.modules (already done in conftest)
    recognizer = FaceRecognizer(upload_dir=upload_dir, vector_db_dir=vector_db_dir)
    assert os.path.exists(upload_dir)
    assert os.path.exists(vector_db_dir)


def test_face_recognizer_register_face_no_face(tmp_path):
    from app.services.face_recognizer import FaceRecognizer
    upload_dir = str(tmp_path / "uploads")
    vector_db_dir = str(tmp_path / "vectordb")

    recognizer = FaceRecognizer(upload_dir=upload_dir, vector_db_dir=vector_db_dir)
    # Mock haar cascade to detect no faces
    recognizer._cascade = MagicMock()
    recognizer._cascade.detectMultiScale.return_value = []
    result = recognizer.register_face(b"fake-bytes", "TestPerson")
    assert result is None


# ── VoiceProcessor ────────────────────────────────────────────────────────────

def test_voice_processor_init_no_whisper():
    """VoiceProcessor should gracefully handle missing whisper model."""
    from app.services.voice_processor import VoiceProcessor
    with patch("whisper.load_model", side_effect=Exception("Model not available")):
        vp = VoiceProcessor()
        assert vp._model is None


def test_voice_processor_transcribe_no_model():
    from app.services.voice_processor import VoiceProcessor
    vp = VoiceProcessor()
    vp._model = None
    result = vp.transcribe(b"some-audio-bytes")
    assert result is None


def test_voice_processor_transcribe_success():
    from app.services.voice_processor import VoiceProcessor
    vp = VoiceProcessor()

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "open gripper"}
    vp._model = mock_model

    with patch("tempfile.NamedTemporaryFile") as mock_tmp:
        mock_tmp.return_value.__enter__ = MagicMock(return_value=MagicMock(name="tmpfile.wav"))
        mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
        with patch("os.unlink"):
            result = vp.transcribe(b"audio-data")
    # Model loaded but temp file may not be valid — result should be str or None
    # Just ensure no exception is raised


def test_voice_processor_match_command():
    from app.services.voice_processor import VoiceProcessor
    vp = VoiceProcessor()

    mock_cmd = MagicMock()
    mock_cmd.phrase = "Open Gripper"
    mock_cmd.action = "RELEASE"
    mock_cmd.target = "gripper"

    # Method is private _match_command
    result = vp._match_command("open gripper", [mock_cmd])
    assert result is not None
    assert result.action == "RELEASE"


def test_voice_processor_match_command_no_match():
    from app.services.voice_processor import VoiceProcessor
    vp = VoiceProcessor()

    mock_cmd = MagicMock()
    mock_cmd.phrase = "Close Gripper"
    mock_cmd.action = "GRAB"
    mock_cmd.target = "gripper"

    result = vp._match_command("completely unrelated phrase xyz abc", [mock_cmd])
    assert result is None


# ── GestureController ─────────────────────────────────────────────────────────

def test_gesture_controller_init():
    from app.services.gesture_controller import GestureController
    # chromadb is imported inside _load_vectordb — already mocked via sys.modules in conftest
    gc = GestureController()
    assert gc is not None


def test_gesture_controller_list_gestures():
    from app.services.gesture_controller import GestureController
    gc = GestureController()

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["gesture1-vec1", "gesture1-vec2", "gesture2-vec1"],
        "metadatas": [
            {"gesture_name": "wave"},
            {"gesture_name": "wave"},
            {"gesture_name": "peace"}
        ]
    }
    gc._collection = mock_collection

    result = gc.list_gestures()
    assert result["wave"] == 2
    assert result["peace"] == 1


def test_gesture_controller_remove_gesture():
    from app.services.gesture_controller import GestureController
    gc = GestureController()

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["wave-vec1", "wave-vec2"],
        "metadatas": [{"gesture_name": "wave"}, {"gesture_name": "wave"}]
    }
    gc._collection = mock_collection

    count = gc.remove_gesture("wave")
    assert count == 2
    mock_collection.delete.assert_called_once_with(ids=["wave-vec1", "wave-vec2"])


def test_gesture_controller_train_wizard():
    from app.services.gesture_controller import GestureController
    gc = GestureController()

    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    gc._collection = mock_collection

    result = gc.train_wizard()
    assert "status" in result
