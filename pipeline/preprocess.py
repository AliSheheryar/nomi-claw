"""Stage 1 — build a low-fps, low-res proxy for the VLM."""
import subprocess
from pathlib import Path


def build_proxy(src: Path, workdir: Path, fps: int = 2, short_side: int = 448) -> Path:
    src = Path(src)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "proxy.mp4"
    if out.exists():
        return out
    vf = f"fps={fps},scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)
