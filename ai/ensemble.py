"""Ensemble voting over multi-model detection + recognition results.

Two stages:

1. **Detection fusion** — SCRFD and MTCNN can each produce a detection
   for the same face. We group them by IoU and keep the highest-confidence
   bbox per group.
2. **Recognition voting** — each grouped face has up to two embeddings
   (ArcFace, FaceNet). Each votes for the closest enrolled student; we
   fuse votes using a weighted score that combines confidence and
   cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AIConfig
from .detectors import Detection, bbox_iou
from .recognizers import Embedding
from .store import EmbeddingStore, StudentInfo


# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceGroup:
    """One physical face, with the evidence gathered from every model."""

    bbox: np.ndarray
    det_scores: dict[str, float] = field(default_factory=dict)
    embeddings: list[Embedding] = field(default_factory=list)

    def merged_bbox(self) -> np.ndarray:
        return self.bbox

    def average_det_score(self) -> float:
        if not self.det_scores:
            return 0.0
        return sum(self.det_scores.values()) / len(self.det_scores)


@dataclass
class Prediction:
    """Final fused prediction that leaves the pipeline."""

    bbox: list[int]
    recognised: bool
    account_id: int | None
    student_id: str | None
    full_name: str | None
    score: float                     # fused score
    per_model: dict[str, dict]       # diagnostic trace
    det_score: float


# ────────────────────────────────────────────────────────────────────────────
# Detection fusion
# ────────────────────────────────────────────────────────────────────────────

def fuse_detections(
    detections: list[Detection], iou_threshold: float
) -> list[FaceGroup]:
    """Group overlapping detections (any source) into single FaceGroups.

    Greedy: each detection either joins an existing group whose bbox has
    IoU ≥ threshold, or starts a new group.
    """
    groups: list[FaceGroup] = []

    # Order by detection score descending so the strongest bbox anchors the group.
    for det in sorted(detections, key=lambda d: d.det_score, reverse=True):
        placed = False
        for g in groups:
            if bbox_iou(det.bbox, g.bbox) >= iou_threshold:
                g.det_scores[det.source] = max(
                    g.det_scores.get(det.source, 0.0), det.det_score
                )
                placed = True
                break
        if not placed:
            groups.append(
                FaceGroup(
                    bbox=det.bbox.copy(),
                    det_scores={det.source: det.det_score},
                )
            )
    return groups


def assign_embeddings(
    groups: list[FaceGroup],
    embeddings: list[Embedding],
    iou_threshold: float,
) -> None:
    """Attach each embedding to the FaceGroup whose bbox it overlaps with."""
    for emb in embeddings:
        best: tuple[float, FaceGroup | None] = (0.0, None)
        for g in groups:
            iou = bbox_iou(emb.detection.bbox, g.bbox)
            if iou > best[0]:
                best = (iou, g)
        if best[1] is not None and best[0] >= iou_threshold * 0.75:
            best[1].embeddings.append(emb)


# ────────────────────────────────────────────────────────────────────────────
# Recognition voting
# ────────────────────────────────────────────────────────────────────────────

def vote(
    groups: list[FaceGroup],
    stores: dict[str, EmbeddingStore],
    weights: dict[str, float],
    cfg: AIConfig,
) -> list[Prediction]:
    """For each face group, query every store with its own embedding and
    combine the top candidates via a weighted cosine score."""

    predictions: list[Prediction] = []

    for group in groups:
        per_model: dict[str, dict] = {}
        # account_id -> accumulated weighted score
        vote_bag: dict[int, float] = {}
        # account_id -> StudentInfo (first one we see)
        info_bag: dict[int, StudentInfo] = {}
        # account_id -> weight sum of models that voted for it
        weight_bag: dict[int, float] = {}

        for emb in group.embeddings:
            store = stores.get(emb.model_name)
            if store is None:
                continue
            account_id, score = store.best_match(emb.vector)
            trace = {
                "matched": account_id is not None,
                "account_id": account_id,
                "score": score,
                "threshold": store.threshold,
            }
            per_model[emb.model_name] = trace

            if account_id is None:
                continue

            w = weights.get(emb.model_name, 1.0)
            vote_bag[account_id] = vote_bag.get(account_id, 0.0) + w * score
            weight_bag[account_id] = weight_bag.get(account_id, 0.0) + w
            info = store.info_for(account_id)
            if info is not None and account_id not in info_bag:
                info_bag[account_id] = info

        if not vote_bag:
            predictions.append(
                Prediction(
                    bbox=group.bbox.tolist(),
                    recognised=False,
                    account_id=None,
                    student_id=None,
                    full_name=None,
                    score=0.0,
                    per_model=per_model,
                    det_score=group.average_det_score(),
                )
            )
            continue

        winner = max(vote_bag.items(), key=lambda kv: kv[1])
        account_id = winner[0]
        # Normalise the fused score by total weight so it stays comparable to
        # single-model cosine similarities (0..1 ish).
        fused = vote_bag[account_id] / max(weight_bag[account_id], 1e-6)
        info = info_bag.get(account_id)

        predictions.append(
            Prediction(
                bbox=group.bbox.tolist(),
                recognised=True,
                account_id=account_id,
                student_id=info.student_id if info else None,
                full_name=info.full_name if info else None,
                score=fused,
                per_model=per_model,
                det_score=group.average_det_score(),
            )
        )

    return predictions
