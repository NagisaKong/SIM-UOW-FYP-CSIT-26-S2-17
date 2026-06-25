# ── Backend: FastAPI + face-recognition AI (CPU build) ───────────────
# Mirrors the CI setup: CPU-only torch, system libs for OpenCV/insightface.
# GPU is not available on Railway, so the heavy ensemble models are disabled
# via env (see railway.json / deploy docs); SCRFD+ArcFace still run on CPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    APP_ENV=production

WORKDIR /app

# System libraries required by OpenCV / insightface at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so requirements.txt does not pull the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1.0" "torchvision>=0.16.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Railway/most PaaS inject $PORT; main_api.py reads HOST/PORT from env.
EXPOSE 8000
CMD ["python", "main_api.py"]
