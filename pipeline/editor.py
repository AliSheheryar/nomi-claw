"""Stage 4 — deterministic edit-decision list from moments + scenes."""
import json
from pathlib import Path

SLOWMO_EMOTIONS = {"tender", "romantic", "laughter"}
JUMPCUT_EMOTIONS = {"argument", "busy"}
SLOWMO_FACTOR = 0.5           # 0.5x speed
JUMPCUT_HEAD_TAIL = 0.8       # keep first 0.8s + last 0.8s of jump-cut clips


def _snap(t: float, boundaries: list[float]) -> float:
    """Snap t to the nearest scene boundary."""
    return min(boundaries, key=lambda b: abs(b - t))


def _effective_duration(m: dict) -> float:
    raw = max(0.1, m["end"] - m["start"])
    if m.get("effect") == "slowmo":
        return raw / SLOWMO_FACTOR       # slow-mo makes it longer
    if m.get("effect") == "jumpcut":
        return min(raw, JUMPCUT_HEAD_TAIL * 2)
    return raw


def plan(moments: list[dict], scenes: list[dict], workdir: Path,
         target_seconds: float = 45.0) -> list[dict]:
    workdir = Path(workdir)
    out = workdir / "edl.json"
    if out.exists():
        return json.loads(out.read_text())

    # Scene boundary times
    boundaries = sorted({s["start"] for s in scenes} | {s["end"] for s in scenes})

    picks: list[dict] = []
    for m in moments:
        if m.get("importance", 3) < 2:
            continue
        s = _snap(m["start"], boundaries)
        e = _snap(m["end"], boundaries)
        if e - s < 0.3:
            continue
        emo = m.get("emotion", "neutral")
        if emo in SLOWMO_EMOTIONS:
            effect = "slowmo"
        elif emo in JUMPCUT_EMOTIONS:
            effect = "jumpcut"
        else:
            effect = "cut"
        picks.append({
            "start": round(s, 3),
            "end": round(e, 3),
            "label": m["label"],
            "emotion": emo,
            "importance": m["importance"],
            "effect": effect,
        })

    # Merge overlapping same-label picks
    picks.sort(key=lambda x: x["start"])
    merged: list[dict] = []
    for p in picks:
        if merged and p["start"] <= merged[-1]["end"] + 0.05 and p["label"] == merged[-1]["label"]:
            merged[-1]["end"] = max(merged[-1]["end"], p["end"])
        else:
            merged.append(p)

    # Greedy pick by importance until target reached
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
