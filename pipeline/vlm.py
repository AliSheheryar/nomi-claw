"""Stage 3 — Qwen2.5-VL moment enumeration per scene."""
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

from .prompts import ENUMERATION_PROMPT

MODEL_IDS = {
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
}

# Chunk any scene longer than this into overlapping windows so the VLM
# sees at most ~20s of content at a time (keeps VRAM in check).
MAX_WINDOW_SEC = 20.0
WINDOW_OVERLAP_SEC = 2.0

_model = None
_processor = None


def _load(model_key: str):
    global _model, _processor
    if _model is not None:
        return _model, _processor
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    mid = MODEL_IDS[model_key]
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
             "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
             "-i", str(proxy),
             "-c:v", "libx264", "-preset", "veryfast", "-an", str(out)],
            check=True,
        )
    return out


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    # Strip ``` fences if the model adds them
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    # Locate the first [ ... ] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def _ask(window_path: Path, t0: float, t1: float, model_key: str) -> list[dict]:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load(model_key)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(window_path), "fps": 2.0,
             "min_pixels": 128*28*28, "max_pixels": 256*28*28},
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
        gen = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _parse_json_array(reply)


def enumerate_moments(proxy: Path, scenes: list[dict], workdir: Path,
                      model: str = "7B") -> list[dict]:
    workdir = Path(workdir)
    out = workdir / "moments.json"
    if out.exists():
        return json.loads(out.read_text())

    moments: list[dict] = []
    for t0, t1 in _windows(scenes):
        win = _extract_window(proxy, t0, t1, workdir)
        got = _ask(win, t0, t1, model)
        for m in got:
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

    # Sort + drop windowing artifacts (dupes at overlap seams)
    moments.sort(key=lambda x: (x["start"], x["end"]))
    dedup: list[dict] = []
    for m in moments:
        if dedup and abs(dedup[-1]["start"] - m["start"]) < 0.5 \
                 and abs(dedup[-1]["end"] - m["end"]) < 0.5 \
                 and dedup[-1]["label"] == m["label"]:
            continue
        dedup.append(m)

    out.write_text(json.dumps(dedup, indent=2))
    return dedup
