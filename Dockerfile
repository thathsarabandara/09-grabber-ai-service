# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    build-essential \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=1000

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless && \
    pip install opencv-contrib-python-headless


# Stage 2: Runtime image
FROM python:3.11-slim AS runner
WORKDIR /app

# Install runtime system libraries for OpenCV, MediaPipe and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY . .

# Expose port
EXPOSE 8004

# Healthcheck using native Python standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/api/health', timeout=2)"

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
