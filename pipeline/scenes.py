"""Stage 2 — shot-boundary detection with PySceneDetect."""
import json
from pathlib import Path

from scenedetect import detect as _detect, ContentDetector


def detect(proxy: Path, workdir: Path, threshold: float = 27.0) -> list[dict]:
    workdir = Path(workdir)
    out = workdir / "scenes.json"
    if out.exists():
        return json.loads(out.read_text())

    raw = _detect(str(proxy), ContentDetector(threshold=threshold))
    scenes = [
        {"idx": i, "start": s.get_seconds(), "end": e.get_seconds()}
        for i, (s, e) in enumerate(raw)
    ]
    # If PySceneDetect finds nothing (single-take clip), treat whole video as one scene.
    if not scenes:
        from .preprocess import probe_duration
        dur = probe_duration(proxy)
        scenes = [{"idx": 0, "start": 0.0, "end": dur}]
    out.write_text(json.dumps(scenes, indent=2))
    return scenes
