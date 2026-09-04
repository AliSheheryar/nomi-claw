# CUDA 12.4 runtime + cuDNN, matches the torch cu124 wheels used elsewhere.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models

# System deps: Python 3.11, ffmpeg (for the pipeline), build tools (autoawq wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
      python3-pip \
      ffmpeg \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default `python` / `pip`
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip

WORKDIR /app

# Install Python deps first for better layer caching.
# Torch cu124 wheels come from PyTorch's index; the rest from PyPI.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install \
      "transformers>=4.49,<4.52" \
      accelerate \
      autoawq \
      qwen-vl-utils \
      av \
      decord \
      ffmpeg-python \
      "scenedetect[opencv]" \
      pillow

# Copy pipeline code
COPY nomi_claw.py ./
COPY pipeline/ ./pipeline/
COPY ground_video.py ./

# Model cache and IO live on host-mounted volumes:
#   -v <host models dir>:/models
#   -v <host videos dir>:/videos
#   -v <host outputs dir>:/app/out
VOLUME ["/models", "/videos", "/app/out"]

# Default: show help. Override with:
#   docker run ... nomi-claw python nomi_claw.py /videos/wedding.mp4
ENTRYPOINT []
CMD ["python", "nomi_claw.py", "--help"]
