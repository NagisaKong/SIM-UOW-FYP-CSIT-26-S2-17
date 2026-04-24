"""End-to-end inference pipeline.

Flow, per frame:

    GAN / CLAHE enhance (low-light frames only)
        └► SCRFD detect    ──►┐
        └► MTCNN detect    ──►┤  fuse by IoU
                              ├►  ArcFace embed  ──►┐
                              └►  FaceNet embed ──►┤  weighted vote
                                                   ▼
                                              Prediction list
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import AIConfig
from .db import EmbeddingRepo
from .detectors import Detection, MtcnnDetector, ScrfdDetector
from .enhancers import build_enhancer, maybe_enhance
from .ensemble import Prediction, assign_embeddings, fuse_detections, vote
from .recognizers import (
    ARCFACE_DIM,
    ARCFACE_MODEL_NAME,
    ARCFACE_MODEL_VERSION,
    FACENET_DIM,
    FACENET_MODEL_NAME,
    FACENET_MODEL_VERSION,
    ArcFaceRecognizer,
    Embedding,
    FaceNetRecognizer,
    embed_all,
)
from .store import EmbeddingStore, SupabaseEmbeddingStore


# ────────────────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    predictions: list[Prediction]
    enhanced: bool
    detections: list[Detection]


class AttendancePipeline:
    """Prototype attendance pipeline.

    Instantiate via `AttendancePipeline.from_env()` for a zero-arg start-up
    that loads config + Supabase state. The in-memory stores are kept live
    so new enrolments are recognised immediately.
    """

    # ── Construction ────────────────────────────────────────────────────
    def __init__(
        self,
        cfg: AIConfig,
        store_manager: SupabaseEmbeddingStore,
    ):
        self.cfg = cfg
        self.store_manager = store_manager

        # Enhancer — may be None if disabled.
        self.enhancer = build_enhancer(cfg)

        # SCRFD + ArcFace share the same FaceAnalysis instance to avoid
        # loading the buffalo_l bundle twice.
        self._arcface = ArcFaceRecognizer(cfg)
        self._scrfd = ScrfdDetector(cfg, shared_app=self._arcface.app)

        self._mtcnn = MtcnnDetector(cfg) if cfg.use_mtcnn else None

        if cfg.use_facenet:
            try:
                self._facenet = FaceNetRecognizer(cfg)
            except ImportError as exc:
                print(f"[pipeline] {exc}")
                self._facenet = None
        else:
            self._facenet = None

        self._weights = {
            ARCFACE_MODEL_NAME: cfg.arcface_weight,
            FACENET_MODEL_NAME: cfg.facenet_weight if self._facenet else 0.0,
        }

    @classmethod
    def from_env(cls, cfg: AIConfig | None = None) -> "AttendancePipeline":
        cfg = cfg or AIConfig()
        repo = EmbeddingRepo(cfg.database_url)
        manager = SupabaseEmbeddingStore(repo=repo)
        manager.register_store(
            EmbeddingStore(ARCFACE_MODEL_NAME, ARCFACE_DIM, cfg.arcface_threshold)
        )
        if cfg.use_facenet:
            manager.register_store(
                EmbeddingStore(FACENET_MODEL_NAME, FACENET_DIM, cfg.facenet_threshold)
            )
        loaded = manager.hydrate_all()
        print(f"[pipeline] DB hydrate: {loaded}")
        return cls(cfg, manager)

    # ── Inference ───────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        # 1. Enhance (optional, low-light-gated).
        frame_in, was_enhanced = maybe_enhance(frame, self.enhancer, self.cfg)

        # 2. Detect — SCRFD primary, MTCNN optional ensemble.
        scrfd_dets = self._scrfd.detect(frame_in)
        mtcnn_dets = self._mtcnn.detect(frame_in) if self._mtcnn is not None else []

        # Spoof-ish floor: anything smaller than the cut-off is probably a
        # photo held up far from the camera.
        min_px = self.cfg.anti_spoof_min_face_px
        dets = [
            d
            for d in (scrfd_dets + mtcnn_dets)
            if (d.bbox[2] - d.bbox[0]) >= min_px and (d.bbox[3] - d.bbox[1]) >= min_px
        ]

        # 3. Group detections across models.
        groups = fuse_detections(dets, self.cfg.ensemble_iou)

        # 4. Embed with each available recognizer (once per unique bbox).
        #    We embed against the _fused_ group bbox by picking the
        #    highest-confidence det per group.
        anchors = self._group_anchors(groups, dets)

        embeddings: list[Embedding] = []
        embeddings.extend(embed_all(self._arcface, frame_in, anchors))
        if self._facenet is not None:
            embeddings.extend(embed_all(self._facenet, frame_in, anchors))

        # 5. Attach embeddings back to their groups.
        assign_embeddings(groups, embeddings, self.cfg.ensemble_iou)

        # 6. Vote across recognizers.
        predictions = vote(
            groups,
            self.store_manager.stores,
            self._weights,
            self.cfg,
        )

        return FrameResult(
            predictions=predictions,
            enhanced=was_enhanced,
            detections=dets,
        )

    # ── Enrolment ──────────────────────────────────────────────────────
    def enrol_student(
        self,
        account_id: int,
        images: list[np.ndarray],
    ) -> dict[str, int]:
        """Extract ArcFace (and optionally FaceNet) embeddings from a set
        of enrolment photos, average them, write one row per model to
        Supabase, and refresh the in-memory stores."""

        vectors: dict[str, list[np.ndarray]] = {ARCFACE_MODEL_NAME: []}
        if self._facenet is not None:
            vectors[FACENET_MODEL_NAME] = []

        for img in images:
            frame_in, _ = maybe_enhance(img, self.enhancer, self.cfg)
            scrfd_dets = self._scrfd.detect(frame_in)
            if not scrfd_dets:
                continue
            # Use the biggest face in the enrolment photo.
            best = max(
                scrfd_dets,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            )

            arc_vec = self._arcface.embed(frame_in, best)
            if arc_vec is not None:
                vectors[ARCFACE_MODEL_NAME].append(arc_vec)

            if self._facenet is not None:
                fn_vec = self._facenet.embed(frame_in, best)
                if fn_vec is not None:
                    vectors[FACENET_MODEL_NAME].append(fn_vec)

        written = self.store_manager.enrol_account(
            account_id=account_id,
            vectors_by_model=vectors,
            retention_days=self.cfg.embedding_retention_days,
            model_versions={
                ARCFACE_MODEL_NAME: ARCFACE_MODEL_VERSION,
                FACENET_MODEL_NAME: FACENET_MODEL_VERSION,
            },
        )
        return written

    # ── Visualisation ───────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        out = frame.copy()
        if result.enhanced:
            cv2.putText(out, "enhanced", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        for p in result.predictions:
            x1, y1, x2, y2 = p.bbox
            colour = (0, 255, 0) if p.recognised else (0, 0, 255)
            label_1 = (
                f"{p.student_id or p.full_name or f'acc#{p.account_id}'}"
                if p.recognised
                else "Unknown"
            )
            label_2 = f"fused={p.score:.2f}  det={p.det_score:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, label_1, (x1, max(0, y1 - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(out, label_2, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return out

    # ── Internals ───────────────────────────────────────────────────────
    @staticmethod
    def _group_anchors(
        groups, detections: list[Detection]
    ) -> list[Detection]:
        """Pick the single best detection per group to feed the recognizers."""
        from .detectors import bbox_iou  # local to avoid cycles

        anchors: list[Detection] = []
        for g in groups:
            best: tuple[float, Detection | None] = (-1.0, None)
            for d in detections:
                if bbox_iou(d.bbox, g.bbox) == 0:
                    continue
                # Prefer SCRFD (aligned kps + lets ArcFace reuse Face obj),
                # and higher det_score.
                score = d.det_score + (0.1 if d.source == "scrfd" else 0.0)
                if score > best[0]:
                    best = (score, d)
            if best[1] is not None:
                anchors.append(best[1])
        return anchors
