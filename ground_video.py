"""CLI entrypoint — runs the 5-stage wedding-clip pipeline end-to-end."""
import argparse
import json
import os
import sys
from pathlib import Path

# Point HF cache at the local ./models dir so downloads land in-project.
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))

from pipeline import preprocess, scenes, vlm, editor, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Local VLM wedding-clip highlight generator (Qwen2.5-VL).",
    )
    ap.add_argument("video", help="Path to the source video")
    ap.add_argument("--target-seconds", type=int, default=45,
                    help="Approx target length of the highlight (default 45s)")
    ap.add_argument("--small", action="store_true",
                    help="Use Qwen2.5-VL-3B-AWQ instead of 7B (lower VRAM)")
    ap.add_argument("--fps", type=int, default=2,
                    help="Proxy fps for VLM (default 2)")
    ap.add_argument("--out", default="out",
                    help="Output root dir (per-video subdir created)")
    args = ap.parse_args()

    src = Path(args.video).resolve()
    if not src.exists():
        print(f"[!] video not found: {src}", file=sys.stderr)
        return 2

    workdir = Path(args.out).resolve() / src.stem
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[+] workdir: {workdir}")

    print("[1/5] Building proxy (ffmpeg)...")
    proxy = preprocess.build_proxy(src, workdir, fps=args.fps)

    print("[2/5] Detecting scenes (PySceneDetect)...")
    shots = scenes.detect(proxy, workdir)
    print(f"      {len(shots)} scene(s)")

    print("[3/5] Enumerating moments (Qwen2.5-VL)... [heavy]")
    moments = vlm.enumerate_moments(
        proxy, shots, workdir,
        model="3B" if args.small else "7B",
    )
    print(f"      {len(moments)} moment(s)")

    print("[4/5] Planning edits...")
    edl = editor.plan(moments, shots, workdir,
                      target_seconds=args.target_seconds)
    print(f"      {len(edl['segments'])} segment(s), "
          f"~{edl['estimated_seconds']:.1f}s total")

    print("[5/5] Rendering highlight.mp4 (ffmpeg)...")
    out_mp4 = render.render(src, edl, workdir / "highlight.mp4")

    print()
    print(f"Done. Output: {out_mp4}")
    print(f"Inspect:      {workdir/'moments.json'}")
    print(f"              {workdir/'edl.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
