"""Image enhancement for faces captured under poor lighting.

Two implementations share a common interface:

* `ClaheEnhancer` — lightweight, dependency-free baseline (CLAHE in LAB
  space + an unsharp-mask pass). Always available.
* `GanEnhancer`  — wraps a real face-restoration GAN (GFPGAN by default,
  optionally Real-ESRGAN). Imported lazily so the prototype still runs
  if these optional deps are missing.

`build_enhancer` picks the best available backend based on `AIConfig`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import cv2
import numpy as np

from .config import AIConfig


class EnhancerProto(Protocol):
    name: str

    def enhance(self, frame: np.ndarray) -> np.ndarray: ...


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def is_low_light(frame: np.ndarray, mean_threshold: float) -> bool:
    """Cheap brightness heuristic — if the V channel mean is below the
    threshold we treat it as low-light and run the enhancer. Good enough
    for a prototype; swap in an exposure/histogram model later."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean()) < mean_threshold


# ────────────────────────────────────────────────────────────────────────────
# CLAHE baseline
# ────────────────────────────────────────────────────────────────────────────

class _BaseEnhancer(ABC):
    name: str = "base"

    @abstractmethod
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        ...


class ClaheEnhancer(_BaseEnhancer):
    """CLAHE on the L channel of LAB space, then a mild unsharp mask.

    This is the fallback when no GAN weights are available. It gives a
    measurable recognition boost on dim webcam captures with effectively
    zero runtime cost.
    """

    name = "clahe"

    def __init__(self, clip_limit: float = 2.5, tile_grid: tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        lab = cv2.merge((l_ch, a_ch, b_ch))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Mild unsharp mask to recover micro-contrast lost to CLAHE.
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
        out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
        return out


# ────────────────────────────────────────────────────────────────────────────
# GAN face restoration (optional — GFPGAN / Real-ESRGAN)
# ────────────────────────────────────────────────────────────────────────────

class GanEnhancer(_BaseEnhancer):
    """Thin wrapper around a face-restoration GAN.

    Tries GFPGAN first (face-specific, ideal for recognition), then falls
    back to Real-ESRGAN if it's the only thing installed. If neither is
    available, `build_enhancer` won't select this class.
    """

    name = "gan"

    def __init__(self, backend: str = "auto", device: str = "cpu"):
        self._backend_name, self._runner = self._load_backend(backend, device)
        self.name = f"gan/{self._backend_name}"

    @staticmethod
    def _load_backend(backend: str, device: str):
        # Deferred imports — we don't want to force these heavy deps on
        # anyone who's happy with CLAHE.
        if backend in ("auto", "gfpgan"):
            try:
                from gfpgan import GFPGANer  # type: ignore

                runner = GFPGANer(
                    model_path=None,  # downloads weights to ~/.cache on first call
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )

                def _run(img: np.ndarray) -> np.ndarray:
                    _, _, restored = runner.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True
                    )
                    return restored if restored is not None else img

                return "gfpgan", _run
            except Exception:
                if backend == "gfpgan":
                    raise

        if backend in ("auto", "realesrgan"):
            try:
                from realesrgan import RealESRGANer  # type: ignore
                from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=2)
                runner = RealESRGANer(
                    scale=2,
                    model_path=None,
                    model=model,
                    tile=0,
                    half=False,
                    device=device,
                )

                def _run(img: np.ndarray) -> np.ndarray:
                    out, _ = runner.enhance(img, outscale=1)
                    return out

                return "realesrgan", _run
            except Exception:
                if backend == "realesrgan":
                    raise

        raise ImportError(
            "No GAN backend available. Install `gfpgan` or `realesrgan`, "
            "or set AI_ENHANCER=clahe to use the lightweight baseline."
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        try:
            return self._runner(frame)
        except Exception:
            # Never let enhancement failures sink the whole pipeline —
            # fall through to the original frame.
            return frame


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────

def build_enhancer(cfg: AIConfig) -> _BaseEnhancer | None:
    if not cfg.use_enhancer:
        return None

    kind = cfg.enhancer_kind.lower()
    if kind == "clahe":
        return ClaheEnhancer()

    if kind in ("gfpgan", "realesrgan"):
        try:
            return GanEnhancer(backend=kind, device=cfg.device)
        except ImportError as exc:
            print(f"[enhancer] {exc} — using CLAHE instead.")
            return ClaheEnhancer()

    # auto
    try:
        return GanEnhancer(backend="auto", device=cfg.device)
    except ImportError:
        return ClaheEnhancer()


# ────────────────────────────────────────────────────────────────────────────
# Convenience: smart per-frame apply with low-light gating
# ────────────────────────────────────────────────────────────────────────────

def maybe_enhance(
    frame: np.ndarray,
    enhancer: _BaseEnhancer | None,
    cfg: AIConfig,
) -> tuple[np.ndarray, bool]:
    """Apply the enhancer if one is configured and the frame looks dim.

    Returns `(frame_out, was_enhanced)`.
    """
    if enhancer is None:
        return frame, False
    if cfg.enhance_only_low_light and not is_low_light(
        frame, cfg.low_light_mean_threshold
    ):
        return frame, False
    return enhancer.enhance(frame), True
