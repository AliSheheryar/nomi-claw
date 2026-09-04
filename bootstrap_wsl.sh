#!/usr/bin/env bash
# bootstrap_wsl.sh — one-shot setup for nomi-claw inside WSL2 Ubuntu.
#
# From WSL Ubuntu shell:
#     cd /mnt/d/nomi-claw
#     bash bootstrap_wsl.sh
#
# Skips Docker entirely. Uses WSL's built-in NVIDIA GPU passthrough.

set -euo pipefail

log() { echo "[bootstrap] $*"; }

log "== [1/4] system packages =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip ffmpeg git build-essential ca-certificates curl

log "== [2/4] venv =="
if [[ ! -d .venv ]]; then
    python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

log "== [3/4] pip deps =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install \
    "transformers>=4.49,<4.52" \
    accelerate \
    autoawq \
    qwen-vl-utils \
    av \
    decord \
    ffmpeg-python \
    "scenedetect[opencv]" \
    pillow

log "== CUDA check =="
python -c "import torch; assert torch.cuda.is_available(); \
print('CUDA OK:', torch.cuda.get_device_name(0), \
'| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"

log "== [4/4] pre-download model (~5-6 GB) =="
export HF_HOME="$(pwd)/models"
mkdir -p "$HF_HOME"
python -c "from transformers import AutoProcessor, AutoModelForImageTextToText; \
m='Qwen/Qwen2.5-VL-7B-Instruct-AWQ'; \
AutoProcessor.from_pretrained(m); AutoModelForImageTextToText.from_pretrained(m); \
print('model cached')"

log ""
log "Done."
log "Activate later:  source .venv/bin/activate"
log 'Run:             python nomi_claw.py "/mnt/d/path/to/video.mp4"'
