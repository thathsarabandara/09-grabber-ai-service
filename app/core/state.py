from app.services.face_recognizer import FaceRecognizer
from app.services.object_detector import ObjectDetector
from app.services.gesture_controller import GestureController
from app.services.voice_processor import VoiceProcessor
from app.services.pick_place_simulator import PickPlaceSimulator
from app.services.sorting_assistant import SortingAssistant

# Instantiate the CV engines
face_recognizer = FaceRecognizer()
object_detector = ObjectDetector()
gesture_controller = GestureController()
voice_processor = VoiceProcessor()
pick_place_simulator = PickPlaceSimulator()
sorting_assistant = SortingAssistant()

# Task metadata — titles and simulated performance metrics shown in the Control Center
task_metadata = {
    "obj-detect": {"title": "Object Detection",       "accuracy": "98.5%", "latency": "12ms"},
    "face-rec":   {"title": "Face Recognition",       "accuracy": "95.2%", "latency": "15ms"},
    "gesture":    {"title": "Hand Gesture Controls",  "accuracy": "92.1%", "latency": "22ms"},
    "voice":      {"title": "Voice Commands",         "accuracy": "88.4%", "latency": "110ms"},
    "pick-place": {"title": "AI Pick & Place",        "accuracy": "96.8%", "latency": "45ms"},
    "sorting":    {"title": "Smart Sorting Assistant","accuracy": "94.0%", "latency": "30ms"},
}
