"""Face-embedding extractors — ArcFace (primary) and FaceNet (ensemble).

Both produce L2-normalised vectors, so downstream similarity is plain
cosine (a dot product after L2 normalisation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .config import AIConfig
from .detectors import Detection


# Surfaced model identifiers so the DB knows which row belongs to which model.
ARCFACE_MODEL_NAME = "arcface"
ARCFACE_MODEL_VERSION = "buffalo_l/r100"
ARCFACE_DIM = 512

FACENET_MODEL_NAME = "facenet"
FACENET_MODEL_VERSION = "vggface2"
FACENET_DIM = 512


@dataclass
class Embedding:
    detection: Detection
    vector: np.ndarray
    model_name: str
    model_version: str


# ────────────────────────────────────────────────────────────────────────────
# ArcFace via InsightFace (buffalo_l → ResNet100)
# ────────────────────────────────────────────────────────────────────────────

class ArcFaceRecognizer:
    model_name = ARCFACE_MODEL_NAME
    model_version = ARCFACE_MODEL_VERSION
    dim = ARCFACE_DIM

    def __init__(self, cfg: AIConfig, shared_app=None):
        if shared_app is None:
            from insightface.app import FaceAnalysis  # deferred

            self.app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(
                ctx_id=cfg.ctx_id,
                det_size=cfg.det_size,
                det_thresh=cfg.det_thresh,
            )
        else:
            self.app = shared_app

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Return a 512-d embedding for the given detection.

        When the detection already came from SCRFD we re-use the InsightFace
        `Face` object (aligned + recognized in one shot). For MTCNN detections
        we crop the ROI and run InsightFace on that crop.
        """
        if detection.source == "scrfd" and detection.raw is not None:
            return _normalise(detection.raw.normed_embedding)

        x1, y1, x2, y2 = detection.bbox
        h, w = frame.shape[:2]
        pad = 0.2
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - pad * bw))
        cy1 = max(0, int(y1 - pad * bh))
        cx2 = min(w, int(x2 + pad * bw))
        cy2 = min(h, int(y2 + pad * bh))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        faces = self.app.get(crop)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        return _normalise(best.normed_embedding)


# ────────────────────────────────────────────────────────────────────────────
# FaceNet via facenet-pytorch (InceptionResnetV1, VGGFace2)
# ────────────────────────────────────────────────────────────────────────────

class FaceNetRecognizer:
    model_name = FACENET_MODEL_NAME
    model_version = FACENET_MODEL_VERSION
    dim = FACENET_DIM

    def __init__(self, cfg: AIConfig):
        try:
            import torch  # type: ignore
            from facenet_pytorch import InceptionResnetV1  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch + torch are required for FaceNet. "
                "Install them or set AI_USE_FACENET=false."
            ) from exc

        self._torch = torch
        device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = device
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        x1, y1, x2, y2 = detection.bbox
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            return None
        pad = 0.1
        cx1 = max(0, int(x1 - pad * bw))
        cy1 = max(0, int(y1 - pad * bh))
        cx2 = min(w, int(x2 + pad * bw))
        cy2 = min(h, int(y2 + pad * bh))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        face = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Match facenet-pytorch pre-processing (no fixed_image_standardization
        # here because InceptionResnetV1 expects [-1, 1]-ish).
        face = (face - 127.5) / 128.0

        t = self._torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self._model(t).cpu().numpy()[0]
        return _normalise(emb)


# ────────────────────────────────────────────────────────────────────────────
# Batch helper — one detection list through one recognizer.
# ────────────────────────────────────────────────────────────────────────────

def embed_all(
    recognizer,
    frame: np.ndarray,
    detections: Iterable[Detection],
) -> list[Embedding]:
    out: list[Embedding] = []
    for d in detections:
        vec = recognizer.embed(frame, d)
        if vec is None:
            continue
        out.append(
            Embedding(
                detection=d,
                vector=vec,
                model_name=recognizer.model_name,
                model_version=recognizer.model_version,
            )
        )
    return out


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v
