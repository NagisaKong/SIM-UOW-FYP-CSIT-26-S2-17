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

Only the models the deployed configuration actually loads are fetched. YOLO
(phone detection) and the MediaPipe FaceLandmarker task file are deliberately
NOT fetched here: behaviour analysis is off on the CPU deployment
(``AI_BEHAVIOUR=false``) and would only bloat the image. The GPU workstation
downloads those on first use.
"""

from __future__ import annotations

import sys
import time

_ATTEMPTS = 3
_BACKOFF_SECONDS = 5


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


def main() -> int:
    prefetch_insightface()
    return 0


if __name__ == "__main__":
    sys.exit(main())
