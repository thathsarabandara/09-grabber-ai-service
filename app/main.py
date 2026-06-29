from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, tasks, status, face, stream, gesture, voice, pick_place, sorting
from app.core.db import init_db
from app.core.config import settings
import os

# Initialize database schemas
try:
    init_db()
    print("Database tables initialized successfully.")
except Exception as e:
    print(f"Failed to initialize database: {e}")

app = FastAPI(title="Grabber Backend AI Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(tasks.router, prefix="/api/v1/ai", tags=["ai"])
app.include_router(status.router, prefix="/api/v1/ai", tags=["ai-status"])
app.include_router(face.router, prefix="/api/v1/ai", tags=["ai-face"])
app.include_router(stream.router, prefix="/api/v1/ai", tags=["ai-stream"])
app.include_router(gesture.router, prefix="/api/v1/ai", tags=["ai-gesture"])
app.include_router(voice.router, prefix="/api/v1/ai", tags=["ai-voice"])
app.include_router(pick_place.router, prefix="/api/v1/ai", tags=["ai-pick-place"])
app.include_router(sorting.router, prefix="/api/v1/ai", tags=["ai-sorting"])

# Serve uploaded operator images from the configured upload directory
UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads/operators", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def read_root():
    return {"message": "Welcome to Grabber Backend AI Service"}
