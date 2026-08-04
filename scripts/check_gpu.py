"""Report whether this machine will actually run inference on the GPU.

`pip list` is misleading here. `onnxruntime` (CPU) and `onnxruntime-gpu`
install into the same `onnxruntime/` package directory, so with both present
the CPU build wins and CUDA disappears from the provider list — while pip
still cheerfully reports that onnxruntime-gpu is installed. The same trap
exists for torch, where the CPU and CUDA wheels share the name `torch`.

Nothing errors in that state. Inference just runs about ten times slower,
which at 1 fps looks like dropped frames rather than a misconfiguration.

Run:
    python scripts/check_gpu.py          # report only
    python scripts/check_gpu.py --fix    # install the GPU runtimes, in order
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import shutil
import subprocess
import sys

_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


def _pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def apply_fix() -> int:
    """Install the GPU runtimes in the one order that works.

    Uninstalling the CPU onnxruntime deletes files the GPU distribution
    shares, so the reinstall is not optional and must follow immediately.
    """
    steps = [
        (["-m", "pip", "uninstall", "onnxruntime", "-y"],
         "remove the CPU onnxruntime that shadows the GPU build"),
        (["-m", "pip", "install", "--force-reinstall", "--no-deps",
          "onnxruntime-gpu"],
         "install onnxruntime-gpu (repairs the shared directory)"),
        (["-m", "pip", "install", "torch", "torchvision",
          "--index-url", _CUDA_INDEX],
         "replace the CPU torch with a CUDA build"),
    ]
    print("This will modify the current environment:\n")
    for i, (cmd, why) in enumerate(steps, 1):
        print(f"  {i}. python {' '.join(cmd)}\n     -> {why}")
    print()
    for i, (cmd, _) in enumerate(steps, 1):
        print(f"--- step {i}/{len(steps)} ---")
        rc = subprocess.run([sys.executable, *cmd]).returncode
        # The uninstall legitimately fails when the CPU build is absent.
        if rc != 0 and i != 1:
            print(f"\nStep {i} failed (exit {rc}). Environment may be "
                  "half-changed — rerun this command or follow "
                  "requirements-gpu.txt manually.")
            return rc
    print("\nDone. Re-checking in a fresh interpreter…\n")
    return subprocess.run([sys.executable, __file__]).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true",
                    help="install the GPU runtimes in the correct order")
    args = ap.parse_args()

    problems: list[str] = []
    notes: list[str] = []

    # ── driver ───────────────────────────────────────────────────────
    gpu_present = False
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            if out:
                gpu_present = True
                print(f"GPU        : {out}")
        except Exception:  # noqa: BLE001
            pass
    if not gpu_present:
        print("GPU        : none detected (nvidia-smi unavailable)")
        print("\nNo NVIDIA GPU here — the CPU defaults in .env.example are "
              "correct for this machine. Nothing to do.")
        if args.fix:
            print("Refusing --fix: installing CUDA wheels on a machine with "
                  "no NVIDIA GPU would only make things slower and larger.")
            return 1
        return 0

    # ── onnxruntime (InsightFace: SCRFD / ArcFace / 3D landmarks) ────
    cpu_v, gpu_v = _pkg("onnxruntime"), _pkg("onnxruntime-gpu")
    print(f"onnxruntime: cpu-build={cpu_v or '-'}  gpu-build={gpu_v or '-'}")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        has_cuda = any("CUDA" in p for p in providers)
        print(f"  providers: {providers}")
        if not has_cuda:
            if cpu_v and gpu_v:
                problems.append(
                    "Both onnxruntime and onnxruntime-gpu are installed. They "
                    "share one package directory and the CPU build is winning, "
                    "so InsightFace runs on CPU."
                )
            elif cpu_v and not gpu_v:
                problems.append(
                    "Only the CPU build of onnxruntime is installed, so "
                    "InsightFace cannot use the GPU."
                )
            else:
                problems.append(
                    "onnxruntime-gpu is installed but exposes no CUDA "
                    "provider — usually a CUDA/cuDNN runtime mismatch."
                )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"onnxruntime failed to import ({exc}). If you just "
                        "uninstalled the CPU build, it deleted files the GPU "
                        "build shares — reinstall onnxruntime-gpu.")

    # ── torch (Ultralytics YOLO phone detection) ─────────────────────
    try:
        import torch

        print(f"torch      : {torch.__version__} "
              f"(cuda build: {torch.version.cuda or 'none'}) "
              f"available={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            problems.append(
                "torch is a CPU-only build, so YOLO phone detection runs on "
                "CPU (~377 ms/frame instead of ~32 ms)."
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch failed to import ({exc}).")

    # ── what the app itself thinks ───────────────────────────────────
    try:
        sys.path.insert(0, ".")
        from core.attendancePipeline import AIConfig

        cfg = AIConfig()
        print(f"config     : {cfg.log_summary()}")
        if cfg.ctx_id < 0:
            notes.append(
                "AI_CTX_ID is -1 and AI_DEVICE is not cuda, so the app is "
                "configured for CPU regardless of the packages above. Set "
                "AI_DEVICE=cuda and AI_CTX_ID=0 in .env to use the GPU."
            )
    except Exception:  # noqa: BLE001
        pass

    print()
    if problems:
        print("PROBLEMS")
        for p in problems:
            print(f"  - {p}")
        print()
        if args.fix:
            return apply_fix()
        print("Fix automatically:")
        print("  python scripts/check_gpu.py --fix")
        print("\nOr manually, in this order — the uninstall must come first, "
              "and it damages the GPU build, hence --force-reinstall:")
        print("  pip uninstall onnxruntime -y")
        print("  pip install --force-reinstall --no-deps onnxruntime-gpu")
        print(f"  pip install torch torchvision --index-url {_CUDA_INDEX}")
        return 1

    if args.fix:
        print("Nothing to fix — GPU inference is already active.")
        return 0
    for n in notes:
        print(f"NOTE: {n}")
    if not notes:
        print("OK — GPU inference is available and configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
