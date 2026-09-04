"""Stage 5 — render highlight video from EDL using ffmpeg."""
import subprocess
from pathlib import Path

from .editor import SLOWMO_FACTOR, JUMPCUT_HEAD_TAIL


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _atempo_chain(factor: float) -> str:
    """atempo only accepts 0.5..2.0 per stage — chain if outside."""
    parts = []
    f = factor
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    parts.append(f"atempo={f:.4f}")
    return ",".join(parts)


def _render_segment(src: Path, seg: dict, out: Path) -> None:
    s, e = seg["start"], seg["end"]
    effect = seg.get("effect", "cut")

    if effect == "jumpcut":
        # Keep only head + tail of the segment, concatted
        head_end = min(s + JUMPCUT_HEAD_TAIL, e)
        tail_start = max(e - JUMPCUT_HEAD_TAIL, head_end)
        parts = [(s, head_end)]
        if tail_start < e - 0.05:
            parts.append((tail_start, e))
        tmp_files = []
        for i, (ps, pe) in enumerate(parts):
            tmp = out.with_name(out.stem + f"_jc{i}.mp4")
            _run(["ffmpeg", "-y", "-loglevel", "error",
                  "-ss", f"{ps:.3f}", "-to", f"{pe:.3f}", "-i", str(src),
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                  "-c:a", "aac", "-b:a", "160k", str(tmp)])
            tmp_files.append(tmp)
        if len(tmp_files) == 1:
            tmp_files[0].rename(out)
        else:
            _concat(tmp_files, out)
            for f in tmp_files:
                f.unlink(missing_ok=True)
        return

    vf, af = None, None
    if effect == "slowmo":
        vf = f"setpts={1/SLOWMO_FACTOR:.4f}*PTS"
        af = _atempo_chain(SLOWMO_FACTOR)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf]
    if af:
        cmd += ["-af", af]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", str(out)]
    _run(cmd)


def _concat(parts: list[Path], out: Path) -> None:
    """Concat via re-encode (safer than -c copy across differing codecs)."""
    list_file = out.with_suffix(".concat.txt")
    list_file.write_text("".join(
        f"file '{p.resolve().as_posix()}'\n" for p in parts
    ))
    _run(["ffmpeg", "-y", "-loglevel", "error",
          "-f", "concat", "-safe", "0", "-i", str(list_file),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
          "-c:a", "aac", "-b:a", "160k",
          "-movflags", "+faststart", str(out)])
    list_file.unlink(missing_ok=True)


def render(src: Path, edl: dict, out: Path) -> Path:
    src = Path(src)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = edl["segments"]
    if not segments:
        raise RuntimeError("EDL has no segments — nothing to render.")

    seg_dir = out.parent / "_segments"
    seg_dir.mkdir(exist_ok=True)
    parts = []
    for i, seg in enumerate(segments):
        seg_out = seg_dir / f"seg_{i:03d}.mp4"
        _render_segment(src, seg, seg_out)
        parts.append(seg_out)

    _concat(parts, out)

    for p in parts:
        p.unlink(missing_ok=True)
    try:
        seg_dir.rmdir()
    except OSError:
        pass
    return out
