# 🧠 Grabber AI Service

> **Repository `09`** · The intelligence layer of the Grabber platform. Provides object detection, coordinate mapping, and autonomous pick-and-place logic using computer vision.

[![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python)]()
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi)]()
[![AI](https://img.shields.io/badge/AI-PyTorch%20%7C%20YOLOv8-EE4C2C?logo=pytorch)]()
[![Vision](https://img.shields.io/badge/Vision-OpenCV%20%7C%20MediaPipe-5C3EE8?logo=opencv)]()
[![Status](https://img.shields.io/badge/Status-Planned-yellow)]()

---

## 🧭 What Is This Repository?

The **AI Service** is responsible for all "intelligent" features of the robotic arm. It processes video frames from the ESP32-CAM and translates visual information into actionable robot coordinates.

**Why separate from the Backend?**
AI models have fundamentally different runtime requirements (Python, potentially GPU access, high RAM usage, and specific model versioning) compared to the Node.js backend services.

---

## 📦 Module Structure

```
09-grabber-ai-service/
├── src/
│   ├── detector/          ← YOLOv8 / MobileNet object detection on camera frames
│   ├── click_to_pick/     ← Pixel-to-joint coordinate mapping logic
│   ├── auto_pick/         ← Semi-autonomous pick-and-place sequencing
│   ├── sorting/           ← Rule-based object sorting (by color/shape)
│   ├── gesture/           ← MediaPipe hand tracking → gesture-to-command mapping
│   └── voice/             ← Speech recognition → command intent parsing
├── models/                ← Pre-trained model weights (.pt, .onnx)
├── requirements.txt
└── README.md
```

---

## 🎯 Intelligent Features

| Feature | Description |
|---|---|
| **Object Detection** | Real-time identification of items (blocks, tools, etc.) in the workspace. |
| **Click-to-Pick** | User clicks an object in the Web UI; AI calculates the joint angles required to grab it. |
| **Smart Sorting** | Automatically detects objects and sorts them into bins based on color or category. |
| **Gesture Control** | Control the robot movements using hand gestures captured by the operator's webcam. |
| **Voice Commands** | "Grab the red block" — parses natural language into robot tasks. |

---

## 🛠️ Tech Stack
- **FastAPI**: High-performance Python web framework for the API.
- **PyTorch / Ultralytics**: Running the YOLOv8 object detection models.
- **OpenCV**: Image preprocessing and frame manipulation.
- **MediaPipe**: Used for hand tracking and gesture recognition.
- **ONNX Runtime**: (Optional) For optimized model inference on CPU.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU + CUDA for faster inference

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the service
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 🔗 Related Repositories
| Repo | Role |
|---|---|
| [`01-grabber-architecture`](../01-grabber-architecture) | Overall intelligence strategy |
| [`04-grabber-web-dashboard`](../04-grabber-web-dashboard) | Displays AI overlays and handles "Click-to-Pick" |
| [`07-grabber-robot-service`](../07-grabber-robot-service) | Executes the pick-and-place sequences calculated here |
| [`08-grabber-telemetry-service`](../08-grabber-telemetry-service) | Provides the live frame stream for analysis |

---
<div align="center">
  <sub>Part of the <strong>Grabber</strong> AI-Powered Industrial Robotic Arm Platform</sub>
</div>
