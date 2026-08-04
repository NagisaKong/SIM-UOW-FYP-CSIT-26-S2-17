"""Download the AI model weights into the container image at BUILD time.

Why this exists
---------------
InsightFace lazily downloads its model pack (``buffalo_l``, ~280 MB) from
GitHub the first time ``FaceAnalysis`` is constructed — which, for this app,
is inside the FastAPI lifespan handler. On a PaaS that means:

  * every cold start re-downloads the pack (the container filesystem is
    ephemeral, so nothing is cached between deploys);
  * a slow or failing GitHub download raises inside lifespan, so the whole
    application fails to start — including login, dashboards and reports that
    have nothing to do with face recognition. Railway then crash-loops the
    deployment, which is exactly the failure this script prevents.

Running the fetch during ``docker build`` instead makes the build fail loudly
if the weights are unavailable, and leaves container startup instant and
independent of GitHub.

The download target must match what the app asks for at runtime:
``FaceAnalysis(name="buffalo_l")`` defaults to ``root="~/.insightface"``, so
this writes to the same expanded path under the build user's home.

By default only the models the deployed configuration actually loads are
fetched. YOLO (phone detection) and the MediaPipe FaceLandmarker task file are
NOT part of the image build: behaviour analysis is off on the CPU deployment
(``AI_BEHAVIOUR=false``) and they would only bloat it.

Local development is the opposite case — there the behaviour models ARE used,
and downloading them lazily means the first analysed frame stalls on a 50 MB
transfer, or fails outright on a slow link. Pass ``--behaviour`` to fetch them
up front; ``scripts/setup.py`` does this for you.

Run:
    python scripts/prefetch_models.py               # recognition only (Docker)
    python scripts/prefetch_models.py --behaviour   # + YOLO and FaceLandmarker
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

_ATTEMPTS = 3
_BACKOFF_SECONDS = 5

# Same asset the DrowsinessDetector loads at runtime; keep the two in step.
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _load_env() -> None:
    """Read .env so we prefetch the models the app will actually load.

    Without this the fetch falls back to the built-in defaults and happily
    downloads, say, yolov8n while the configured model is yolov8m — leaving
    the real one to download lazily anyway. Absent .env (the Docker build) and
    absent python-dotenv are both fine; env vars already set always win.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(Path(__file__).resolve().parent.parent / ".env",
                    override=False)
    except Exception:  # noqa: BLE001 — configuration is optional here
        pass


def prefetch_insightface(name: str = "buffalo_l") -> str:
    """Fetch an InsightFace model pack, retrying transient network errors."""
    from insightface.utils.storage import ensure_available

    last_error: Exception | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            # ensure_available() returns immediately when the pack is already
            # on disk, so re-running this script is cheap and idempotent.
            path = ensure_available("models", name, root="~/.insightface")
            print(f"[prefetch] {name} ready at {path}")
            return path
        except Exception as exc:  # noqa: BLE001 — retry any download failure
            last_error = exc
            print(f"[prefetch] attempt {attempt}/{_ATTEMPTS} failed: {exc}")
            if attempt < _ATTEMPTS:
                time.sleep(_BACKOFF_SECONDS)

    raise RuntimeError(
        f"Could not download the '{name}' model pack after {_ATTEMPTS} attempts. "
        "The image build is aborted on purpose: shipping without weights would "
        "only move this failure to container startup."
    ) from last_error


def prefetch_face_landmarker() -> str:
    """Fetch the MediaPipe FaceLandmarker asset used for EAR / MAR."""
    path = Path(os.getenv("AI_FACEMESH_MODEL", "models/face_landmarker.task"))
    if path.is_file():
        print(f"[prefetch] face_landmarker already at {path}")
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prefetch] downloading face_landmarker -> {path}")
    urllib.request.urlretrieve(_FACE_LANDMARKER_URL, path)
    print(f"[prefetch] face_landmarker ready at {path}")
    return str(path)


def prefetch_yolo(model_name: str | None = None) -> str:
    """Fetch the Ultralytics weights used for phone detection.

    Constructing YOLO() triggers the download; ultralytics caches it, so this
    is a no-op once the file exists.
    """
    model_name = model_name or os.getenv("AI_PHONE_MODEL", "yolov8n.pt")
    from ultralytics import YOLO

    YOLO(model_name)
    print(f"[prefetch] YOLO weights ready: {model_name}")
    return model_name


def prefetch_behaviour() -> None:
    """Best-effort fetch of the behaviour models.

    Unlike the recognition pack these are not fatal: BehaviourAnalysisService
    disables the corresponding detector and keeps running when a model is
    missing, so a failure here should not stop setup.
    """
    for label, fn in (("face_landmarker", prefetch_face_landmarker),
                      ("YOLO", prefetch_yolo)):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[prefetch] WARNING: could not fetch {label}: {exc}")
            print(f"[prefetch] {label} will be downloaded on first use instead.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviour", action="store_true",
                    help="also fetch the YOLO and FaceLandmarker models "
                         "(local development; not wanted in the image)")
    args = ap.parse_args()

    _load_env()
    prefetch_insightface()
    if args.behaviour:
        prefetch_behaviour()
    return 0


if __name__ == "__main__":
    sys.exit(main())
