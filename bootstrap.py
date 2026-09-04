"""
bootstrap.py — one-shot Python setup for nomi-claw on the host.

Cross-platform (Windows / Linux / macOS). Detects what's missing and installs it,
then creates a .venv, installs Python deps, and pre-downloads the Qwen model.

Prereq: any Python 3.9+ available to launch this script (Windows Store Python,
system Python, whatever). It will install Python 3.11 alongside if needed.

Run:
    python bootstrap.py                # everything
    python bootstrap.py --skip-model   # skip the 5-6 GB model download
    python bootstrap.py --small        # download 3B AWQ (~3 GB) instead of 7B
    python bootstrap.py --no-system    # skip system-package installs (venv+pip only)

On Windows, running as Administrator lets it install system packages via winget.
Without admin it falls back to venv-only mode and prints what's missing.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV_DIR = ROOT / ".venv"
MODELS_DIR = ROOT / "models"

MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

PIP_DEPS = [
    "transformers>=4.49,<4.52",
    "accelerate",
    "autoawq",
    "qwen-vl-utils",
    "av",
    "decord",
    "ffmpeg-python",
    "scenedetect[opencv]",
    "pillow",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log(">>> " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def is_admin() -> bool:
    if os.name == "nt":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return os.geteuid() == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# system packages
# ---------------------------------------------------------------------------

def install_windows(no_system: bool) -> None:
    if no_system:
        log("skipping system installs (--no-system)")
        return
    if not have("winget"):
        log("winget not found. Install App Installer from the Microsoft Store, then rerun.")
        return
    if not is_admin():
        log("WARNING: not running as Administrator — winget package installs may prompt or fail.")

    packages = [
        ("git", "Git.Git"),
        ("ffmpeg", "Gyan.FFmpeg"),
        ("gh", "GitHub.cli"),
    ]
    # Only install Python 3.11 if no 3.11 present
    if not _python311_available():
        packages.insert(0, ("python", "Python.Python.3.11"))
        packages.insert(1, ("py", "Python.Launcher"))

    for cmd, pkg in packages:
        if have(cmd):
            log(f"ok  : {cmd} present")
            continue
        log(f"inst: {pkg}")
        try:
            run(["winget", "install", "-e", "--id", pkg,
                 "--accept-source-agreements", "--accept-package-agreements",
                 "--silent"])
        except subprocess.CalledProcessError as e:
            log(f"WARN: winget failed for {pkg} ({e.returncode}) — install manually.")


def install_linux(no_system: bool) -> None:
    if no_system:
        log("skipping system installs (--no-system)")
        return
    if not is_admin():
        log("WARNING: not root — skipping apt installs. Rerun with sudo or --no-system.")
        return
    if have("apt-get"):
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y",
             "python3.11", "python3.11-venv", "python3.11-dev",
             "python3-pip", "ffmpeg", "git", "build-essential", "curl", "ca-certificates"])
    elif have("dnf"):
        run(["dnf", "install", "-y",
             "python3.11", "python3.11-devel", "ffmpeg", "git", "gcc", "gcc-c++", "make"])
    else:
        log("WARN: unsupported Linux package manager. Install python3.11, ffmpeg, git manually.")


def install_macos(no_system: bool) -> None:
    if no_system:
        log("skipping system installs (--no-system)")
        return
    if not have("brew"):
        log("Homebrew not found. Install from https://brew.sh then rerun.")
        return
    run(["brew", "install", "python@3.11", "ffmpeg", "git", "gh"])


def install_system_packages(no_system: bool) -> None:
    sysname = platform.system()
    log(f"OS: {sysname}")
    if sysname == "Windows":
        install_windows(no_system)
    elif sysname == "Linux":
        install_linux(no_system)
    elif sysname == "Darwin":
        install_macos(no_system)
    else:
        log(f"WARN: unknown OS {sysname} — skipping system installs.")


# ---------------------------------------------------------------------------
# python 3.11 discovery / venv
# ---------------------------------------------------------------------------

SUPPORTED_PY = ("Python 3.11", "Python 3.12")


def _python311_available() -> str | None:
    """Return command for a supported Python (3.11 or 3.12) if available."""
    candidates: list[list[str]] = []
    if os.name == "nt":
        candidates += [["py", "-3.11"], ["py", "-3.12"],
                       ["python"], ["python3.12"], ["python3.11"]]
    else:
        candidates += [["python3.11"], ["python3.12"], ["python3"], ["python"]]
    for c in candidates:
        try:
            out = subprocess.run(c + ["--version"], capture_output=True, text=True)
            banner = (out.stdout + out.stderr).strip()
            if out.returncode == 0 and any(v in banner for v in SUPPORTED_PY):
                return " ".join(c)
        except FileNotFoundError:
            continue
    return None


def create_venv() -> Path:
    py311 = _python311_available()
    if VENV_DIR.exists():
        log(f"venv exists: {VENV_DIR}")
    elif py311:
        log(f"creating venv with: {py311}")
        run(py311.split() + ["-m", "venv", str(VENV_DIR)])
    else:
        log("WARN: no Python 3.11 available; falling back to the interpreter running this script.")
        log(f"      ({sys.executable}, Python {sys.version.split()[0]})")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


# ---------------------------------------------------------------------------
# pip + model
# ---------------------------------------------------------------------------

def install_pip_deps(vpy: Path) -> None:
    run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
    log("installing torch (CUDA 12.4 wheels)")
    run([str(vpy), "-m", "pip", "install", "torch", "torchvision",
         "--index-url", TORCH_INDEX])
    log("installing pipeline deps")
    run([str(vpy), "-m", "pip", "install", *PIP_DEPS])


def check_cuda(vpy: Path) -> None:
    log("CUDA visibility check")
    try:
        subprocess.run(
            [str(vpy), "-c",
             "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; "
             "print('CUDA OK:', torch.cuda.get_device_name(0), "
             "'| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"],
            check=True,
        )
    except subprocess.CalledProcessError:
        log("WARN: torch cannot see a CUDA GPU. Install NVIDIA driver R550+ and rerun.")


def download_model(vpy: Path, small: bool) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    mid = MODEL_3B if small else MODEL_7B
    log(f"pre-downloading model → {MODELS_DIR}  ({mid})")
    env = os.environ.copy()
    env["HF_HOME"] = str(MODELS_DIR)
    subprocess.run(
        [str(vpy), "-c",
         "from transformers import AutoProcessor, AutoModelForImageTextToText; "
         f"m='{mid}'; AutoProcessor.from_pretrained(m); "
         "AutoModelForImageTextToText.from_pretrained(m); print('model cached')"],
        check=True, env=env,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Set up nomi-claw on this host.")
    ap.add_argument("--no-system", action="store_true",
                    help="Skip OS-level package installs (Python, ffmpeg, git). Do only venv + pip.")
    ap.add_argument("--skip-model", action="store_true",
                    help="Do not pre-download the Qwen model.")
    ap.add_argument("--small", action="store_true",
                    help="Download the 3B AWQ model (~3 GB) instead of 7B (~5-6 GB).")
    args = ap.parse_args()

    log("== [1/5] system packages ==")
    install_system_packages(args.no_system)

    log("== [2/5] verify tools ==")
    for c in ["ffmpeg", "git"]:
        log(f"  {c}: {'OK' if have(c) else 'MISSING'}")

    log("== [3/5] venv ==")
    vpy = create_venv()
    log(f"  venv python: {vpy}")

    log("== [4/5] pip deps ==")
    install_pip_deps(vpy)
    check_cuda(vpy)

    log("== [5/5] model ==")
    if args.skip_model:
        log("skipped (--skip-model)")
    else:
        try:
            download_model(vpy, small=args.small)
        except subprocess.CalledProcessError as e:
            log(f"WARN: model download failed ({e.returncode}). Rerun later with:")
            log("      python bootstrap.py --no-system")

    log("")
    log("Done.")
    if os.name == "nt":
        log(r"Activate:  .\.venv\Scripts\Activate.ps1")
    else:
        log("Activate:  source .venv/bin/activate")
    log('Run:       python nomi_claw.py "path/to/video.mp4"')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        log(f"FATAL: command failed with exit {e.returncode}")
        sys.exit(e.returncode)
