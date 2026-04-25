"""Face detection — SCRFD (primary) and MTCNN (ensemble)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import AIConfig


@dataclass
class Detection:
    """Normalised detection record produced by either detector."""

    bbox: np.ndarray          # int[4]  (x1, y1, x2, y2)
    det_score: float
    kps: np.ndarray | None    # float[5, 2] or None
    source: str               # "scrfd" | "mtcnn"
    # The InsightFace Face object, if this detection came from SCRFD.
    # Kept so ArcFace can reuse its internal alignment without re-running
    # the model. `None` for MTCNN detections.
    raw: Any | None = None


# ────────────────────────────────────────────────────────────────────────────
# SCRFD (InsightFace buffalo_l: detection + recognition)
# ────────────────────────────────────────────────────────────────────────────

class ScrfdDetector:
    name = "scrfd"

    def __init__(self, cfg: AIConfig, shared_app=None):
        """`shared_app` lets the ArcFace recognizer re-use the same
        FaceAnalysis instance so we don't load SCRFD twice."""
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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        faces = self.app.get(frame)
        out: list[Detection] = []
        for f in faces:
            out.append(
                Detection(
                    bbox=f.bbox.astype(int),
                    det_score=float(f.det_score),
                    kps=np.asarray(f.kps) if getattr(f, "kps", None) is not None else None,
                    source="scrfd",
                    raw=f,
                )
            )
        return out


# ────────────────────────────────────────────────────────────────────────────
# MTCNN (facenet-pytorch)
# ────────────────────────────────────────────────────────────────────────────

class MtcnnDetector:
    name = "mtcnn"

    def __init__(self, cfg: AIConfig):
        try:
            from facenet_pytorch import MTCNN  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is required for MTCNN. "
                "Install it or set AI_USE_MTCNN=false."
            ) from exc

        import torch  # type: ignore

        self._torch = torch
        device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = device
        self._mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            min_face_size=cfg.anti_spoof_min_face_px,
            thresholds=[0.6, 0.7, 0.7],
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # MTCNN wants RGB HWC float.
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)
        if boxes is None:
            return []

        detections: list[Detection] = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            if box is None or prob is None:
                continue
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    bbox=np.array([int(x1), int(y1), int(x2), int(y2)], dtype=int),
                    det_score=float(prob),
                    kps=np.asarray(lm, dtype=float) if lm is not None else None,
                    source="mtcnn",
                )
            )
        return detections


# ────────────────────────────────────────────────────────────────────────────
# IoU utility used by the ensemble to merge overlapping detections
# ────────────────────────────────────────────────────────────────────────────

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
