# nomi-claw — local VLM wedding-clip auto-editor

Detects meaningful moments in wedding/mehendi footage with **Qwen2.5-VL-7B (AWQ)**
running locally on an RTX 4060 (8 GB), then auto-edits a highlight reel with
jump-cuts on busy beats and slow-mo on tender/romantic/laughter beats.

## Setup (once)

Requires Python 3.11, ffmpeg on PATH, and a CUDA GPU. From `D:\nomi-claw`:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates `.venv`, installs deps, and downloads the model (~5–6 GB) into `.\models`.

## Run

```powershell
. .\.venv\Scripts\Activate.ps1
python ground_video.py "C:\path\to\wedding.mp4"
```

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
