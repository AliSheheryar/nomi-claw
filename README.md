# nomi-claw — local VLM wedding-clip auto-editor

Detects meaningful moments in wedding/mehendi footage with **Qwen2.5-VL-7B (AWQ)**
running locally on an RTX 4060 (8 GB), then auto-edits a highlight reel with
jump-cuts on busy beats and slow-mo on tender/romantic/laughter beats.

## Setup (once)

### Cross-platform (Python — recommended)

Requires any Python 3.9+ on PATH. On Windows, run from an admin shell so
system-package installs (Python 3.11, ffmpeg, git) can proceed:

```bash
python bootstrap.py                 # everything: sys deps + venv + torch + model
python bootstrap.py --small         # use 3B model (~3 GB) instead of 7B
python bootstrap.py --no-system     # skip OS package installs, only venv + pip
python bootstrap.py --skip-model    # skip the 5-6 GB model download
```

Works on Windows (winget), Linux (apt/dnf), macOS (brew).

### Windows only — PowerShell bootstrap

Right-click PowerShell → **Run as Administrator**, then from the repo root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\bootstrap.ps1
```

This installs winget (if missing), Git, GitHub CLI, Python 3.11 + launcher,
ffmpeg, Visual C++ Build Tools, checks the NVIDIA driver, then runs `setup.ps1`.

### Machine already has Python 3.11 + ffmpeg

From `D:\nomi-claw`:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates `.venv`, installs deps, and downloads the model (~5–6 GB) into `.\models`.

## Run

```powershell
. .\.venv\Scripts\Activate.ps1
python ground_video.py "C:\path\to\wedding.mp4"
```

## Docker (alternative to venv install)

Requires: Docker Desktop + WSL2 backend on Windows, NVIDIA driver >= R550,
and the NVIDIA Container Toolkit enabled in WSL2. (On Linux hosts: install
`nvidia-container-toolkit` from your distro.)

### Pull the prebuilt image (fastest — no local build)

The GitHub Actions workflow at `.github/workflows/docker.yml` builds and
publishes to GitHub Container Registry on every push to `main`.

```bash
docker pull ghcr.io/alisheheryar/nomi-claw:latest
```

Then run (mount host video + output dirs, expose GPU):

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/models" \
  -v "$PWD/videos:/videos:ro" \
  -v "$PWD/out:/app/out" \
  ghcr.io/alisheheryar/nomi-claw:latest \
  python nomi_claw.py /videos/wedding.mp4 --target-seconds 45
```

Windows PowerShell equivalent:

```powershell
docker run --rm --gpus all `
  -v "${PWD}\models:/models" `
  -v "${PWD}\videos:/videos:ro" `
  -v "${PWD}\out:/app/out" `
  ghcr.io/alisheheryar/nomi-claw:latest `
  python nomi_claw.py /videos/wedding.mp4 --target-seconds 45
```

### Build locally instead

```bash
docker compose build
```

Put source videos in `./videos/`, then run:

```bash
docker compose run --rm nomi-claw python nomi_claw.py /videos/wedding.mp4 --target-seconds 45
```

Outputs land in `./out/<video-stem>/highlight.mp4`. The model cache lives in
`./models/` on the host and survives container recreation.

Flags work exactly as in the venv install (`--small`, `--fps`, `--target-seconds`,
`--out`). Paths inside the container: `/videos` (read-only), `/app/out`, `/models`.

Common flags:
- `--target-seconds 60`  — aim for ~60 s highlight (default 45)
- `--small`              — use 3B model if 7B OOMs
- `--fps 3`              — denser proxy sampling (more accurate, slower)
- `--out my_outputs`     — where per-video working dirs go

Output: `out/<video-stem>/highlight.mp4`, plus `moments.json` and `edl.json`
alongside it for inspection and iteration.

## Iterating without re-running the VLM

The expensive step is stage 3 (Qwen). To re-tune editing rules only:
1. Keep `moments.json` and `scenes.json`
2. Delete `edl.json` and `highlight.mp4`
3. Re-run `ground_video.py <video>` — it will skip stages 1–3 (cached outputs
   detected) and only redo planning + render.

## Tuning knobs

- **Prompt** — edit `pipeline/prompts.py`
- **Effects** — edit `SLOWMO_EMOTIONS`, `JUMPCUT_EMOTIONS`, `SLOWMO_FACTOR`,
  `JUMPCUT_HEAD_TAIL` in `pipeline/editor.py`
- **Scene sensitivity** — pass `threshold` to `scenes.detect` (lower = more cuts)
