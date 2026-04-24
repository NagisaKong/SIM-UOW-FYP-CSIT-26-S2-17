"""Centralised configuration for the AI pipeline.

Everything the pipeline needs to start up is read from environment variables
(populated from the repo-root `.env` by `python-dotenv`), with sensible
defaults suitable for a prototype demo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    # repo root = parent of ai/
    _repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    return float(v) if v else default


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    return int(v) if v else default


@dataclass
class AIConfig:
    """Prototype AI configuration.

    Reads every tunable knob from environment variables with a prefix of `AI_`
    so operators can override the pipeline behaviour without touching code.
    """

    # ── Database ────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "")
    )

    # ── Hardware ────────────────────────────────────────────────────────
    # ctx_id: -1 = CPU, >=0 = that GPU index (InsightFace convention).
    ctx_id: int = field(default_factory=lambda: _env_int("AI_CTX_ID", -1))
    device: str = field(default_factory=lambda: os.getenv("AI_DEVICE", "cpu"))

    # ── Detection ───────────────────────────────────────────────────────
    det_size: tuple[int, int] = (640, 640)
    det_thresh: float = field(default_factory=lambda: _env_float("AI_DET_THRESH", 0.5))
    use_mtcnn: bool = field(default_factory=lambda: _env_bool("AI_USE_MTCNN", True))

    # ── Recognition ─────────────────────────────────────────────────────
    # Cosine-similarity cut-off used to decide "match vs. unknown".
    # ArcFace embeddings are L2-normalised so [-1, 1]; FaceNet likewise.
    arcface_threshold: float = field(
        default_factory=lambda: _env_float("AI_ARCFACE_THRESHOLD", 0.40)
    )
    facenet_threshold: float = field(
        default_factory=lambda: _env_float("AI_FACENET_THRESHOLD", 0.55)
    )
    use_facenet: bool = field(default_factory=lambda: _env_bool("AI_USE_FACENET", True))

    # ── Enhancement (GAN / CLAHE) ───────────────────────────────────────
    use_enhancer: bool = field(default_factory=lambda: _env_bool("AI_USE_ENHANCER", True))
    enhancer_kind: str = field(default_factory=lambda: os.getenv("AI_ENHANCER", "auto"))
    # "auto" = try GFPGAN/RealESRGAN, fall back to CLAHE; "clahe" = force baseline;
    # "gfpgan" / "realesrgan" = force that model (requires the optional dep).
    enhance_only_low_light: bool = field(
        default_factory=lambda: _env_bool("AI_ENHANCE_LOW_LIGHT_ONLY", True)
    )
    low_light_mean_threshold: float = field(
        default_factory=lambda: _env_float("AI_LOW_LIGHT_MEAN", 80.0)
    )

    # ── Ensemble ────────────────────────────────────────────────────────
    # Weight given to each recognizer when fusing cosine scores.
    arcface_weight: float = field(
        default_factory=lambda: _env_float("AI_ARCFACE_WEIGHT", 0.65)
    )
    facenet_weight: float = field(
        default_factory=lambda: _env_float("AI_FACENET_WEIGHT", 0.35)
    )
    # IoU cut-off to consider two detections "the same face" across models.
    ensemble_iou: float = field(default_factory=lambda: _env_float("AI_ENSEMBLE_IOU", 0.4))

    # ── Anti-spoof (simple heuristic placeholder) ───────────────────────
    anti_spoof_min_face_px: int = field(
        default_factory=lambda: _env_int("AI_ANTISPOOF_MIN_FACE_PX", 50)
    )

    # ── PDPC / retention ────────────────────────────────────────────────
    embedding_retention_days: int = field(
        default_factory=lambda: _env_int("AI_EMBEDDING_RETENTION_DAYS", 365)
    )

    def log_summary(self) -> str:
        return (
            f"AIConfig(ctx_id={self.ctx_id}, device={self.device}, "
            f"det_thresh={self.det_thresh}, "
            f"arcface(th={self.arcface_threshold}, w={self.arcface_weight}), "
            f"facenet(enabled={self.use_facenet}, th={self.facenet_threshold}, "
            f"w={self.facenet_weight}), "
            f"mtcnn={self.use_mtcnn}, enhancer={self.enhancer_kind}"
            f"{'(on)' if self.use_enhancer else '(off)'})"
        )


def load_config() -> AIConfig:
    return AIConfig()
