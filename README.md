# 🧠 Grabber AI Service

> **Repository `09`** · The intelligence layer of the Grabber platform — built on Python 3.11, FastAPI, and SQLAlchemy. Integrates YOLOv11 object detectors, MediaPipe Hands trackers, OpenCV, and OpenAI Whisper to provide object detection, facial operator recognition, hand gesture controls, voice commands, and automated sorting scripts. Relies on ChromaDB vector engines for face and hand pose lookups.

[![Language](https://img.shields.io/badge/Language-Python%203.11-3776AB?logo=python&style=flat-square)]()
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi&style=flat-square)]()
[![AI/Vision](https://img.shields.io/badge/AI-YOLOv11%20%7C%20MediaPipe-red.svg?style=flat-square)]()
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-blue.svg?style=flat-square)]()
[![Audio](https://img.shields.io/badge/Speech-OpenAI%20Whisper-brightgreen.svg?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg?style=flat-square)]()

---

## 🎥 Video Demonstration

<div align="center">
  <a href="https://youtu.be/Mtr7tXCaWqA?si=0_PxXe9ioOuXJJnB">
    <img src="https://img.youtube.com/vi/Mtr7tXCaWqA/maxresdefault.jpg" alt="Grabber Demo Video" width="70%">
  </a>
  <br/>
  <sub>Click the image above to watch the demonstration video on YouTube.</sub>
</div>

---

## 🧭 What Is This Repository?

The **AI Service** is responsible for translating raw visual feed inputs and audio buffers into robot coordinates and logic commands. Deployed as a dedicated service, it supports GPU/CUDA acceleration to run object detection and speech recognition without impacting core robot control operations.

### Key Core Features
1. **YOLOv11 Object Detection**: Detects blocks or tools in the robot's workspace, supporting confidence filters and workspace zone categorization (Left/Center/Right).
2. **ChromaDB Face Verification**: Stores operator faces as L2-normalized 10,000-dimensional embeddings in ChromaDB to verify access permissions before executing commands.
3. **Dual-Mode Gesture Controller**: Uses MediaPipe Tasks to track wrist movements for translation commands (`BASE`, `SHOULDER`, `ELBOW`) and count extended fingers to control the gripper. Also supports registering custom static poses in ChromaDB.
4. **Whisper Voice Command Interpreter**: Transcribes audio inputs using OpenAI's Whisper model and matches commands against database templates using fuzzy substring word overlap scoring.
5. **Real-time Stream Annotator**: Proxies raw ESP32-CAM MJPEG feeds, applies computer vision overlays, and streams the annotated video back to client dashboards.

---

## 📦 Project Structure

The project implements a clean layer division, separating database structures, Pydantic parameters, API routers, and connection managers:

```
09-grabber-ai-service/
├── app/
│   ├── api/                 # Endpoint routers and dependency injection
│   │   └── routes/          # Core routers (face, gesture, voice, pick_place, sorting, tasks, stream, health)
│   ├── core/                # DB sessions, ChromaDB persistence, and configuration settings
│   │   ├── config.py        # Pydantic settings config base
│   │   ├── db.py            # Sync database connection engine and schemas initializer
│   │   └── state.py         # Global CV engines initialization wrapper
│   ├── models/              # SQLAlchemy database tables (Operator, TaskState, Voice, etc.)
│   ├── schemas/             # Pydantic schemas validating payloads
│   └── services/            # Custom logic services (YOLO, MediaPipe, Whisper, Face)
├── models/                  # Local folder mapping MediaPipe task weights
├── yolo11x.pt               # YOLOv11 full network weights (fallback to yolo11l/yolo11n)
├── Dockerfile               # Production multi-stage build configuration with GPU hooks
├── docker-compose.yml       # Dev stack execution setups with GPU configurations
├── requirements.txt         # Production library dependencies
└── README.md
```

### Module Code Index

* **App Entry & Configurations**:
  * [app/main.py](app/main.py): Sets up the FastAPI application. Instantiates static folder routing under `/uploads/operators`, loads database schemas via `init_db`, and registers the API endpoints.
  * [app/core/db.py](app/core/db.py): Initializes the SQLAlchemy database engine and handles metadata creation.
  * [app/core/config.py](app/core/config.py): Core settings module reading variables from `.env`.
  * [app/core/state.py](app/core/state.py): Globally initializes the computer vision engines (OpenCV Haar Cascades, YOLOv11, MediaPipe Landmarkers, Whisper model).

* **API Routers**:
  * [app/api/routes/stream.py](app/api/routes/stream.py): Streams MJPEG feeds from the ESP32-CAM, decodes JPEGs, applies CV annotations, and streams the annotated video back to client dashboards.
  * [app/api/routes/face.py](app/api/routes/face.py): Handles operator face registration, registers embeddings in ChromaDB, and logs metadata in MySQL.
  * [app/api/routes/gesture.py](app/api/routes/gesture.py): Manages custom hand gesture registrations in ChromaDB.
  * [app/api/routes/voice.py](app/api/routes/voice.py): Handles audio uploads, transcribes inputs via Whisper, and triggers matched commands.
  * [app/api/routes/tasks.py](app/api/routes/tasks.py): Lists, starts, and stops AI tasks.

* **Services**:
  * [app/services/object_detector.py](app/services/object_detector.py): Loads the YOLO model (searching locally for `yolo11x.pt`, `yolo11l.pt`, and falling back to auto-downloading `yolo11n.pt`). Processes bounding boxes, labels, and divides the frame into left, center, and right thirds.
  * [app/services/face_recognizer.py](app/services/face_recognizer.py): Uses Haar Cascades to detect faces, applies CLAHE lighting normalization, generates 10,000-dimensional embeddings, and matches vectors in ChromaDB using cosine distance (match threshold of `0.35`).
  * [app/services/gesture_controller.py](app/services/gesture_controller.py): Implements MediaPipe Tasks HandLandmarker. Tracks wrist coordinates for movement commands and counts fingers to toggle the gripper. Also supports registering custom landmark coordinates in ChromaDB.
  * [app/services/voice_processor.py](app/services/voice_processor.py): Loads Whisper model weights and matches inputs using word overlap scoring (match threshold of `>=0.5`).

---

## 📊 Database Schema Specifications

The service connects to a MySQL database and uses **SQLAlchemy** declarative models:

### 1. Operators Table (`operators`)
Logs registered operators and their access levels.
* **Columns**:
  * `id`: `INTEGER` (Primary Key Auto-Increment)
  * `name`: `VARCHAR(255)` (Unique name used as ChromaDB index reference)
  * `face_path`: `VARCHAR(500)` (Saved cropped face filename stored in `uploads/operators/`)
  * `access_level`: `VARCHAR(50)` (e.g., `admin`, `operator`)
  * `created_at`: `DATETIME`

### 2. Task States Table (`task_states`)
Tracks whether an AI task is active or idle.
* **Columns**:
  * `task_id`: `VARCHAR(100)` (Primary Key, e.g., `obj-detect`, `face-rec`, `gesture`)
  * `status`: `VARCHAR(50)` (Current state: `active` or `idle`)
  * `updated_at`: `DATETIME`

### 3. Voice Commands Table (`voice_commands`)
Stores voice command templates.
* **Columns**:
  * `id`: `INTEGER` (Primary Key Auto-Increment)
  * `phrase`: `VARCHAR(255)` (Target command phrase, e.g., "open gripper")
  * `action`: `VARCHAR(100)` (Corresponding system action)

### 4. Gesture Mappings Table (`gesture_mappings` & `gesture_settings`)
Maps hand gestures to robot actions.
* **Columns**:
  * `gesture_name`: `VARCHAR(100)` (Primary Key)
  * `action`: `VARCHAR(100)` (e.g., `BASE_LEFT`, `ELBOW_RIGHT`)
* **Settings Columns**:
  * `control_enabled`: `BOOLEAN` (Toggles physical robot gesture control)
  * `safety_enabled`: `BOOLEAN` (Requires an active face match to verify operator permissions)

---

## 🧠 AI Pipeline Details & Configurations

### 1. Face Recognition Vector Search
```
Input Image ---> Haar Cascade ---> 100x100 Grayscale ---> CLAHE Normalization
                                                               |
Unit Cosine Vector (10k dimensions) <--------------------------+
       |
ChromaDB persistent index ---> cosine distance query ---> Match Name (threshold < 0.35)
```

### 2. Hand Gesture Mapping Logic
* **Built-in Translation Control**: Tracks the coordinate difference of the wrist joint across a rolling window of 6 frames. Calculates displacement vectors to determine movement directions.
* **Grip Control**: Counts extended fingers using knuckle joint angles:
  * **Open Grip**: Detected if $\ge 4$ fingers are extended.
  * **Close Grip**: Detected if $\le 1$ finger is extended.
* **ChromaDB Pose Override**: When hand landmarks are detected, the system extracts a 63-dimensional coordinate array and queries ChromaDB for custom pose overrides. Custom poses take priority over translation controls.

### 3. Whisper Speech Command Processor
* The voice endpoint accepts WAV/MP3 uploads, saves them temporarily to disk, and runs transcription via Whisper.
* The transcribed string is cleaned and compared to phrases in the `voice_commands` database table. A command is matched if it shares $\ge 50\%$ word overlap with the target phrase.

---

## ⚙️ Core API Endpoints

### 1. Task Controls
* **List Tasks**: `GET /api/v1/ai/tasks`
* **Start Task**: `POST /api/v1/ai/tasks/start`
  * Body: `{"task_id": "obj-detect"}`
* **Stop Task**: `POST /api/v1/ai/tasks/stop`
  * Body: `{"task_id": "obj-detect"}`

### 2. Face Registration & Operators
* **Register Operator Face**: `POST /api/v1/ai/face/register` (Multipart Form)
  * Parameters: `files` (Multiple images), `name` (Operator name), `access_level` (Access group)
* **List Operators**: `GET /api/v1/ai/face/operators`
* **Delete Operator**: `DELETE /api/v1/ai/face/operators/{id}`
* **Query ChromaDB Stats**: `POST /api/v1/ai/face/retrain`

### 3. Speech Processing
* **Transcribe Command**: `POST /api/v1/ai/voice/transcribe` (Multipart Form)
  * Parameter: `file` (WAV/MP3 audio file)

### 4. MJPEG Live Stream
* **Get Stream with CV overlays**: `GET /api/v1/ai/stream?camera_url=http://...`
  * Returns: Multipart MJPEG stream (`multipart/x-mixed-replace`) with live visual overlays based on active tasks.

---

## 🚀 Getting Started

### 1. Environment Configurations
Create a `.env` configuration file in the project root:
```env
PROJECT_NAME="Grabber AI Service"

# MySQL Database URI
DATABASE_URL="mysql+pymysql://thathsara:BandaPutha@db/grabber_ai"

# Vector DB storage paths inside container volumes
CHROMA_DB_DIR=/data/vectordb/faces
GESTURE_DB_DIR=/data/vectordb/gestures
UPLOAD_DIR=/data/uploads/operators

# Whisper model config
WHISPER_MODEL=base

# MediaPipe configuration thresholds
MEDIAPIPE_DETECTION_CONFIDENCE=0.7
MEDIAPIPE_TRACKING_CONFIDENCE=0.6

# MQTT settings
MQTT_BROKER=grabber-mqtt-server
MQTT_PORT=1883
MQTT_USERNAME=thathsara
MQTT_PASSWORD=BandaPutha
```

### 2. Local Setup
Ensure you have Python 3.11+, a running MySQL database, and FFmpeg installed on your system:
```bash
# Initialize and activate virtual environment
python -m venv aienv
source aienv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the AI service
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

### 3. Run via Docker Compose (with GPU Support)
Build and run the container locally:
```bash
# Deploys with local model cache volume mapping and GPU reservations
docker compose up -d --build
```

---

## 🔗 Related Grabber Repositories

| Repository | Purpose |
|---|---|
| [`01-grabber-architecture`](https://github.com/thathsarabandara/01-grabber-architecture) | System blueprints, MQTT schemas, and database designs |
| [`03-grabber-mobile-app`](https://github.com/thathsarabandara/03-grabber-mobile-app) | Flutter app remote teleoperation HUD |
| [`05-grabber-api-gateway`](https://github.com/thathsarabandara/05-grabber-api-gateway) | Inbound router proxying app REST & WebSocket requests |
| [`06-grabber-auth-service`](https://github.com/thathsarabandara/06-grabber-auth-service) | Service managing user profiles, image updates, and JWT sessions |
| [`07-grabber-robot-service`](https://github.com/thathsarabandara/07-grabber-robot-service) | Service processing joint commands and homing schedules |
| [`08-grabber-telemetry-service`](https://github.com/thathsarabandara/08-grabber-telemetry-service) | Core service publishing live telemetry and webcam captures |

---

<div align="center">
  <sub>Part of the <strong>Grabber</strong> AI-Powered Industrial Robotic Arm Platform</sub>
</div>
