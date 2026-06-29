FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    pkg-config \
    build-essential \
    default-libmysqlclient-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=1000

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install opencv-python-headless

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
