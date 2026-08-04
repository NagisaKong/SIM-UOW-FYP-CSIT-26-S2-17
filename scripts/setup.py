"""One-command local setup: dependencies, .env, and GPU runtimes.

Safe to re-run. It never overwrites an existing .env, never installs CUDA
wheels on a machine without an NVIDIA card, and stops at the first step that
genuinely fails rather than leaving a half-built environment behind.

Run:
    python scripts/setup.py
    python scripts/setup.py --no-gpu      # skip the GPU switch entirely
    python scripts/setup.py --check       # report only, change nothing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_ENV = _REPO / ".env"
_ENV_EXAMPLE = _REPO / ".env.example"


def _say(step: str, msg: str) -> None:
    print(f"\n[{step}] {msg}")


def _run(args: list[str], why: str) -> int:
    print(f"    $ {' '.join(args)}")
    # Our own prints are buffered while a subprocess writes straight to the
    # console, so without this the child's output appears above the heading
    # that introduces it.
    sys.stdout.flush()
    return subprocess.run(args, cwd=_REPO).returncode


def check_python() -> bool:
    v = sys.version_info
    print(f"    python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if v < (3, 10):
        print("    ERROR: Python 3.10+ required.")
        return False
    # Installing into the system interpreter is a common way to end up with a
    # broken machine-wide environment, so say something — but do not block,
    # since CI and Docker both install globally on purpose.
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv:
        print("    NOTE: not running inside a virtual environment. Packages "
              "will be installed system-wide. Consider:")
        print("        python -m venv .venv")
        print("        .venv\\Scripts\\activate     (Windows)")
        print("        source .venv/bin/activate   (macOS / Linux)")
    return True


def install_requirements(check_only: bool) -> bool:
    req = _REPO / "requirements.txt"
    if not req.is_file():
        print(f"    ERROR: {req} not found.")
        return False
    if check_only:
        print("    would run: pip install -r requirements.txt")
        return True
    rc = _run([sys.executable, "-m", "pip", "install", "-r", str(req)],
              "project dependencies")
    if rc != 0:
        print(f"    ERROR: dependency install failed (exit {rc}).")
        return False
    return True


def ensure_env(check_only: bool) -> bool:
    if _ENV.is_file():
        print("    .env already exists — left untouched.")
        # A copied template still has the placeholder DSN, which fails at
        # connect time with a confusing error rather than a clear one.
        text = _ENV.read_text(encoding="utf-8", errors="ignore")
        if "user:password@host" in text:
            print("    WARNING: DATABASE_URL is still the placeholder. Ask the "
                  "project leader for the real value before running the app.")
        return True
    if not _ENV_EXAMPLE.is_file():
        print(f"    ERROR: {_ENV_EXAMPLE} missing; cannot create .env.")
        return False
    if check_only:
        print("    would create .env from .env.example")
        return True
    shutil.copyfile(_ENV_EXAMPLE, _ENV)
    print("    created .env from .env.example")
    print("    ACTION REQUIRED: set DATABASE_URL in .env before running.")
    return True


def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        # capture_output so the probe itself stays silent — check_gpu.py is
        # what reports the details, and only when it is actually run.
        return subprocess.run(["nvidia-smi"], capture_output=True,
                              timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def setup_gpu(check_only: bool, skip: bool) -> bool:
    checker = _REPO / "scripts" / "check_gpu.py"
    if skip:
        print("    skipped (--no-gpu)")
        return True
    if not has_nvidia_gpu():
        print("    no NVIDIA GPU detected — CPU runtimes are correct here, "
              "nothing to change.")
        return True
    if check_only:
        _run([sys.executable, str(checker)], "report GPU state")
        return True
    print("    NVIDIA GPU detected. requirements.txt installs CPU runtimes, "
          "so the wheels are switched now.")
    # check_gpu --fix is a no-op when the environment is already correct, so
    # this stays idempotent across re-runs.
    rc = _run([sys.executable, str(checker), "--fix"], "switch to GPU wheels")
    if rc != 0:
        print("    WARNING: GPU setup did not complete. The app still works on "
              "CPU; rerun `python scripts/check_gpu.py --fix` to retry.")
    return True


def prefetch_models(check_only: bool, skip: bool) -> bool:
    """Download model weights now rather than during the first request.

    InsightFace (~280 MB) is fetched inside the FastAPI lifespan handler
    otherwise, so a slow link turns into a failed startup; the YOLO and
    FaceLandmarker assets are fetched on the first analysed frame, which
    stalls that frame. Both are cached on disk, so this is a no-op afterwards.
    """
    script = _REPO / "scripts" / "prefetch_models.py"
    if skip:
        print("    skipped (--no-models)")
        return True
    if not script.is_file():
        print("    prefetch script missing — models will download on first use.")
        return True
    if check_only:
        print("    would run: prefetch_models.py --behaviour")
        return True
    rc = _run([sys.executable, str(script), "--behaviour"], "model weights")
    if rc != 0:
        print("    WARNING: prefetch incomplete. The app still works — the "
              "missing weights download on first use instead.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gpu", action="store_true",
                    help="do not switch to CUDA wheels even if a GPU is present")
    ap.add_argument("--no-models", action="store_true",
                    help="do not pre-download model weights")
    ap.add_argument("--check", action="store_true",
                    help="report what would happen; change nothing")
    args = ap.parse_args()

    print("=" * 66)
    print("  FYP-26-S2-17 — local environment setup")
    if args.check:
        print("  (--check: nothing will be modified)")
    print("=" * 66)

    _say("1/5", "Python interpreter")
    if not check_python():
        return 1

    _say("2/5", "Project dependencies")
    if not install_requirements(args.check):
        return 1

    _say("3/5", "Configuration (.env)")
    if not ensure_env(args.check):
        return 1

    # Before the models, so the weights are fetched with the runtime that will
    # actually load them.
    _say("4/5", "GPU runtimes")
    setup_gpu(args.check, args.no_gpu)

    _say("5/5", "Model weights")
    prefetch_models(args.check, args.no_models)

    print("\n" + "=" * 66)
    print("  Setup complete.")
    print("=" * 66)
    print("\nNext:")
    print("  1. Put the real DATABASE_URL in .env (ask the project leader).")
    print("  2. Verify:   python scripts/check_gpu.py")
    print("  3. Run:      python main_api.py")
    print("\nTests write to whatever DATABASE_URL points at, so point it at a")
    print("local database before running:  python -m pytest tests/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
