"""
nomi_claw.py — single-file local VLM wedding-clip auto-editor.

One dependency install, one file to run. Uses Qwen2.5-VL (AWQ) locally on
an RTX 4060 (8 GB) to detect meaningful moments in wedding/mehendi footage,
then auto-cuts a highlight with jump-cuts on busy beats and slow-mo on
tender / romantic / laughter beats.

Setup (once, from D:\\nomi-claw):
    py -3.11 -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install "transformers>=4.49,<4.52" accelerate autoawq qwen-vl-utils av decord ffmpeg-python "scenedetect[opencv]" pillow

Run:
    python nomi_claw.py "path\\to\\wedding.mp4"
    python nomi_claw.py "path\\to\\wedding.mp4" --small --target-seconds 60

Prereqs on PATH: ffmpeg, ffprobe. Prereq GPU: any CUDA card with >=4 GB VRAM
(3B model) or >=7 GB (7B model). Model weights auto-download to ./models
on first run (~5-6 GB for 7B AWQ, ~3 GB for 3B AWQ).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Cache HF downloads inside the project dir
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))


# ============================================================================
# CONFIG — edit these to tune behavior without touching pipeline code
# ============================================================================

MODEL_IDS = {
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
}

PROXY_FPS_DEFAULT = 2          # VLM sampling density (frames/sec)
PROXY_SHORT_SIDE = 448          # downscale short edge to this many px
MAX_WINDOW_SEC = 20.0           # chunk scenes longer than this
WINDOW_OVERLAP_SEC = 2.0        # overlap between consecutive windows

SCENE_THRESHOLD = 27.0          # PySceneDetect ContentDetector threshold

SLOWMO_EMOTIONS = {"tender", "romantic", "laughter"}
JUMPCUT_EMOTIONS = {"argument", "busy"}
SLOWMO_FACTOR = 0.5             # 0.5 = half speed
JUMPCUT_HEAD_TAIL = 0.8         # keep first + last N seconds of jump-cut clips
MIN_IMPORTANCE = 2              # drop moments below this importance
MIN_SEGMENT_SEC = 0.3           # discard any segment shorter than this

ENUMERATION_PROMPT = """You are analyzing a segment of a wedding / mehendi video.
The segment starts at absolute time {t0:.2f}s and ends at {t1:.2f}s in the original video.

List every distinct meaningful moment you see. For EACH moment return ONE JSON object with fields:
  "start":      absolute seconds (float, within [{t0:.2f}, {t1:.2f}])
  "end":        absolute seconds (float, within [{t0:.2f}, {t1:.2f}])
  "label":      3-6 words describing the action
  "emotion":    one of tender | romantic | laughter | argument | neutral | busy
  "importance": integer 1..5  (5 = must keep, 1 = filler)

Focus on:
- family members sitting beside the groom on the sofa
- family members applying mehendi on the groom (one by one -- treat each as its own moment)
- playful arguments and reactions
- laughter and smiles
- quiet tender exchanges and close-ups

Return ONLY a JSON array of these objects. No prose, no code fences.
If nothing meaningful happens, return [].
"""


# ============================================================================
# STAGE 1 — preprocess: build a small proxy for the VLM
# ============================================================================

def build_proxy(src: Path, workdir: Path, fps: int) -> Path:
    out = workdir / "proxy.mp4"
    if out.exists():
        return out
    vf = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),-2,{PROXY_SHORT_SIDE})':"
        f"'if(gt(iw,ih),{PROXY_SHORT_SIDE},-2)'"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src), "-vf", vf, "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         str(out)],
        check=True,
    )
    return out


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


# ============================================================================
# STAGE 2 — scene detection (PySceneDetect)
# ============================================================================

def detect_scenes(proxy: Path, workdir: Path) -> list[dict]:
    out = workdir / "scenes.json"
    if out.exists():
        return json.loads(out.read_text())

    from scenedetect import detect, ContentDetector
    raw = detect(str(proxy), ContentDetector(threshold=SCENE_THRESHOLD))
    scenes = [
        {"idx": i, "start": s.get_seconds(), "end": e.get_seconds()}
        for i, (s, e) in enumerate(raw)
    ]
    if not scenes:
        scenes = [{"idx": 0, "start": 0.0, "end": probe_duration(proxy)}]
    out.write_text(json.dumps(scenes, indent=2))
    return scenes


# ============================================================================
# STAGE 3 — VLM moment enumeration (Qwen2.5-VL)
# ============================================================================

_model = None
_processor = None


def _load_model(model_key: str):
    global _model, _processor
    if _model is not None:
        return _model, _processor
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    mid = MODEL_IDS[model_key]
    print(f"      loading {mid} ...")
    _processor = AutoProcessor.from_pretrained(mid)
    _model = AutoModelForImageTextToText.from_pretrained(
        mid,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    _model.eval()
    return _model, _processor


def _windows(scenes: list[dict]) -> Iterable[tuple[float, float]]:
    for s in scenes:
        t0, t1 = float(s["start"]), float(s["end"])
        if t1 - t0 <= MAX_WINDOW_SEC:
            yield t0, t1
            continue
        cur = t0
        while cur < t1:
            end = min(cur + MAX_WINDOW_SEC, t1)
            yield cur, end
            if end >= t1:
                break
            cur = end - WINDOW_OVERLAP_SEC


def _extract_window(proxy: Path, t0: float, t1: float, workdir: Path) -> Path:
    out = workdir / f"_win_{int(t0*1000):08d}_{int(t1*1000):08d}.mp4"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(proxy),
             "-c:v", "libx264", "-preset", "veryfast", "-an", str(out)],
            check=True,
        )
    return out


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def _ask_vlm(window_path: Path, t0: float, t1: float, model_key: str) -> list[dict]:
    import torch
    from qwen_vl_utils import process_vision_info
    model, processor = _load_model(model_key)

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(window_path), "fps": 2.0,
             "min_pixels": 128 * 28 * 28, "max_pixels": 256 * 28 * 28},
            {"type": "text", "text": ENUMERATION_PROMPT.format(t0=t0, t1=t1)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _parse_json_array(reply)


def enumerate_moments(proxy: Path, scenes: list[dict], workdir: Path,
                      model_key: str) -> list[dict]:
    out = workdir / "moments.json"
    if out.exists():
        return json.loads(out.read_text())

    moments: list[dict] = []
    windows = list(_windows(scenes))
    for i, (t0, t1) in enumerate(windows, 1):
        print(f"      window {i}/{len(windows)}  {t0:.1f}s -> {t1:.1f}s")
        win = _extract_window(proxy, t0, t1, workdir)
        for m in _ask_vlm(win, t0, t1, model_key):
            try:
                s = max(t0, min(float(m["start"]), t1))
                e = max(s + 0.2, min(float(m["end"]), t1))
            except (KeyError, TypeError, ValueError):
                continue
            moments.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "label": str(m.get("label", "moment"))[:80],
                "emotion": str(m.get("emotion", "neutral")).lower(),
                "importance": int(m.get("importance", 3) or 3),
            })

    # De-dupe overlap seams
    moments.sort(key=lambda x: (x["start"], x["end"]))
    dedup: list[dict] = []
    for m in moments:
        if dedup and abs(dedup[-1]["start"] - m["start"]) < 0.5 \
                 and abs(dedup[-1]["end"] - m["end"]) < 0.5 \
                 and dedup[-1]["label"] == m["label"]:
            continue
        dedup.append(m)

    # Clean up window files
    for p in workdir.glob("_win_*.mp4"):
        p.unlink(missing_ok=True)

    out.write_text(json.dumps(dedup, indent=2))
    return dedup


# ============================================================================
# STAGE 4 — edit-decision-list planner (deterministic)
# ============================================================================

def _snap(t: float, boundaries: list[float]) -> float:
    return min(boundaries, key=lambda b: abs(b - t))


def _effective_duration(m: dict) -> float:
    raw = max(0.1, m["end"] - m["start"])
    if m.get("effect") == "slowmo":
        return raw / SLOWMO_FACTOR
    if m.get("effect") == "jumpcut":
        return min(raw, JUMPCUT_HEAD_TAIL * 2)
    return raw


def plan_edl(moments: list[dict], scenes: list[dict], workdir: Path,
             target_seconds: float) -> dict:
    out = workdir / "edl.json"
    if out.exists():
        return json.loads(out.read_text())

    boundaries = sorted({s["start"] for s in scenes} | {s["end"] for s in scenes})

    picks: list[dict] = []
    for m in moments:
        if m.get("importance", 3) < MIN_IMPORTANCE:
            continue
        s = _snap(m["start"], boundaries)
        e = _snap(m["end"], boundaries)
        if e - s < MIN_SEGMENT_SEC:
            continue
        emo = m.get("emotion", "neutral")
        effect = ("slowmo" if emo in SLOWMO_EMOTIONS
                  else "jumpcut" if emo in JUMPCUT_EMOTIONS
                  else "cut")
        picks.append({
            "start": round(s, 3), "end": round(e, 3),
            "label": m["label"], "emotion": emo,
            "importance": m["importance"], "effect": effect,
        })

    # Merge overlapping same-label
    picks.sort(key=lambda x: x["start"])
    merged: list[dict] = []
    for p in picks:
        if merged and p["start"] <= merged[-1]["end"] + 0.05 \
                 and p["label"] == merged[-1]["label"]:
            merged[-1]["end"] = max(merged[-1]["end"], p["end"])
        else:
            merged.append(p)

    # Greedy pick by importance up to target length
    merged.sort(key=lambda x: (-x["importance"], x["start"]))
    chosen: list[dict] = []
    total = 0.0
    for p in merged:
        chosen.append(p)
        total += _effective_duration(p)
        if total >= target_seconds:
            break

    chosen.sort(key=lambda x: x["start"])
    edl = {
        "target_seconds": target_seconds,
        "estimated_seconds": round(total, 2),
        "segments": chosen,
    }
    out.write_text(json.dumps(edl, indent=2))
    return edl


# ============================================================================
# STAGE 5 — render with ffmpeg
# ============================================================================

def _atempo_chain(factor: float) -> str:
    parts = []
    f = factor
    while f < 0.5:
        parts.append("atempo=0.5"); f /= 0.5
    while f > 2.0:
        parts.append("atempo=2.0"); f /= 2.0
    parts.append(f"atempo={f:.4f}")
    return ",".join(parts)


def _render_segment(src: Path, seg: dict, out: Path) -> None:
    s, e = seg["start"], seg["end"]
    effect = seg.get("effect", "cut")

    if effect == "jumpcut":
        head_end = min(s + JUMPCUT_HEAD_TAIL, e)
        tail_start = max(e - JUMPCUT_HEAD_TAIL, head_end)
        parts = [(s, head_end)]
        if tail_start < e - 0.05:
            parts.append((tail_start, e))
        tmp = []
        for i, (ps, pe) in enumerate(parts):
            t = out.with_name(out.stem + f"_jc{i}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{ps:.3f}", "-to", f"{pe:.3f}", "-i", str(src),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-c:a", "aac", "-b:a", "160k", str(t)],
                check=True,
            )
            tmp.append(t)
        if len(tmp) == 1:
            tmp[0].rename(out)
        else:
            _concat(tmp, out)
            for f in tmp:
                f.unlink(missing_ok=True)
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(src)]
    if effect == "slowmo":
        cmd += ["-vf", f"setpts={1/SLOWMO_FACTOR:.4f}*PTS",
                "-af", _atempo_chain(SLOWMO_FACTOR)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", str(out)]
    subprocess.run(cmd, check=True)


def _concat(parts: list[Path], out: Path) -> None:
    list_file = out.with_suffix(".concat.txt")
    list_file.write_text("".join(
        f"file '{p.resolve().as_posix()}'\n" for p in parts
    ))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", str(out)],
        check=True,
    )
    list_file.unlink(missing_ok=True)


def render(src: Path, edl: dict, out: Path) -> Path:
    if not edl["segments"]:
        raise RuntimeError("EDL has no segments -- nothing to render.")
    seg_dir = out.parent / "_segments"
    seg_dir.mkdir(exist_ok=True)
    parts = []
    for i, seg in enumerate(edl["segments"]):
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


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("video", help="Path to source video")
    ap.add_argument("--target-seconds", type=int, default=45)
    ap.add_argument("--small", action="store_true", help="Use 3B model instead of 7B")
    ap.add_argument("--fps", type=int, default=PROXY_FPS_DEFAULT)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    src = Path(args.video).resolve()
    if not src.exists():
        print(f"[!] video not found: {src}", file=sys.stderr)
        return 2

    workdir = Path(args.out).resolve() / src.stem
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[+] workdir: {workdir}")

    print("[1/5] proxy ...")
    proxy = build_proxy(src, workdir, fps=args.fps)

    print("[2/5] scenes ...")
    scenes = detect_scenes(proxy, workdir)
    print(f"      {len(scenes)} scene(s)")

    print("[3/5] VLM enumeration ...")
    model_key = "3B" if args.small else "7B"
    moments = enumerate_moments(proxy, scenes, workdir, model_key)
    print(f"      {len(moments)} moment(s)")

    print("[4/5] planning edits ...")
    edl = plan_edl(moments, scenes, workdir, target_seconds=args.target_seconds)
    print(f"      {len(edl['segments'])} segment(s), ~{edl['estimated_seconds']:.1f}s total")

    print("[5/5] rendering ...")
    out_mp4 = render(src, edl, workdir / "highlight.mp4")

    print(f"\nDone. Output: {out_mp4}")
    print(f"Inspect:      {workdir/'moments.json'}")
    print(f"              {workdir/'edl.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
