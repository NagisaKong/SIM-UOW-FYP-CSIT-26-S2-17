"""Manually invoke the StyleGAN synthetic-pair threshold calibration (CR-04).

Why this exists
----------------
`core/training/synthetic_gen.py` (StyleGAN2 synthetic identity generation) and
`core/training/calibrate.py` (ArcFace scoring + threshold search) implement
the offline calibration path described in the technical documentation, but
neither module has ever had an entry point: nothing in the codebase imports
or runs them. This script is that entry point — it is the "manually invoked"
step CR-04 refers to.

It never runs during a live check-in and never runs automatically; it is
started by hand, on demand, by whoever wants to (re-)derive a similarity
threshold from GAN-synthesised identity pairs instead of from real enrolled
faces. The result is a suggested threshold only — this script does not write
to MODEL_CONFIGS. Deploying the value is a separate, deliberate step through
the admin Model Management screen (or scripts/seed_demo.py-style tooling),
consistent with the "administrator can deploy through Model Management"
description in the technical documentation.

Requires a StyleGAN2-FFHQ checkpoint. If it is not already at
`models/stylegan/stylegan2-ffhq-config-f.pkl`, this script downloads it from
NVIDIA's public CDN (~300 MB) the first time it runs, into the project's own
`models/` directory — not a user-profile cache.

Run:
    python scripts/calibrate_synthetic.py
    python scripts/calibrate_synthetic.py --pairs 20     # more synthetic pairs
    python scripts/calibrate_synthetic.py --model-path path/to/other.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# Allow `python scripts/calibrate_synthetic.py` to find the `core` package —
# sys.path[0] is this file's own directory (scripts/), not the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The PyTorch-native checkpoint from the stylegan2-ada-pytorch repo (the same
# repo core/training/legacy.py and dnnlib/ are vendored from), NOT the older
# TensorFlow-era stylegan2 checkpoint at nvlabs-fi-cdn.nvidia.com/stylegan2/ —
# that one unpickles into a `_TFNetworkStub` graph description that legacy.py
# can only convert by importing the original TF repo's `training.networks`
# module, which this project does not vendor. The ada-pytorch checkpoint below
# loads directly as a torch.nn.Module, no conversion step required.
_MODEL_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
)
_DEFAULT_MODEL_PATH = Path("models/stylegan/ffhq.pkl")
_ATTEMPTS = 3
_BACKOFF_SECONDS = 5


def _ensure_model(model_path: Path) -> Path:
    if model_path.is_file():
        print(f"[calibrate_synthetic] StyleGAN2 checkpoint already at {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[calibrate_synthetic] downloading StyleGAN2-FFHQ (~300 MB) -> {model_path}")
    print(f"[calibrate_synthetic] source: {_MODEL_URL}")

    last_error: Exception | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            urllib.request.urlretrieve(_MODEL_URL, model_path)
            print(f"[calibrate_synthetic] checkpoint ready at {model_path}")
            return model_path
        except Exception as exc:  # noqa: BLE001 — retry any download failure
            last_error = exc
            print(f"[calibrate_synthetic] attempt {attempt}/{_ATTEMPTS} failed: {exc}")
            if model_path.exists():
                model_path.unlink()  # don't leave a partial file behind
            if attempt < _ATTEMPTS:
                time.sleep(_BACKOFF_SECONDS)

    raise RuntimeError(
        f"Could not download the StyleGAN2-FFHQ checkpoint after {_ATTEMPTS} "
        "attempts. Calibration is aborted rather than run against a missing "
        "or partial model file."
    ) from last_error


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=Path, default=_DEFAULT_MODEL_PATH,
                     help="path to the StyleGAN2 .pkl checkpoint "
                          "(downloaded here if missing)")
    ap.add_argument("--pairs", type=int, default=1,
                     help="how many times to call prepare_calibration_set() "
                          "(each call yields 1 positive + 1 negative pair)")
    args = ap.parse_args()

    model_path = _ensure_model(args.model_path)

    # Imported after the download check so `--help` doesn't require torch/cv2.
    from core.training.calibrate import calibrate_threshold
    from core.training.synthetic_gen import SyntheticDataGenerator

    print("[calibrate_synthetic] loading StyleGAN2 generator...")
    generator = SyntheticDataGenerator(model_path=str(model_path))

    all_pairs: list[tuple] = []
    all_labels: list[int] = []
    for i in range(args.pairs):
        print(f"[calibrate_synthetic] generating synthetic pair set {i + 1}/{args.pairs}...")
        pairs, labels = generator.prepare_calibration_set()
        all_pairs.extend(pairs)
        all_labels.extend(labels)

    print(f"[calibrate_synthetic] scoring {len(all_pairs)} pairs through ArcFace...")
    threshold = calibrate_threshold(all_pairs, all_labels)

    print(f"\n[calibrate_synthetic] suggested similarity threshold: {threshold:.4f}")
    print("[calibrate_synthetic] this value is NOT written to the database. "
          "Review it and, if appropriate, deploy it through the admin "
          "Model Management screen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
