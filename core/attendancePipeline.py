"""Service Class: AttendancePipeline.
Absorbed from previous standalone files:
  • config.py             →  AIConfig dataclass + env-loading helpers
  • database_manager.py   →  EmbeddingRow dataclass + EmbeddingRepo
  • (formerly in facialImage.py)  ScrfdDetector / MtcnnDetector /
        ArcFaceRecognizer / FaceNetRecognizer / Image enhancers /
        Ensemble voting / EmbeddingStore / SupabaseEmbeddingStore /
        AttendancePipeline

Layout:
   1. Env helpers + AIConfig
   2. EmbeddingRow + EmbeddingRepo  (DB persistence)
   3. Constants (model identifiers)
   4. Lightweight data classes (Detection, Embedding, FaceGroup,
      Prediction, StudentInfo, FrameResult)
   5. Pure helpers (_normalise, bbox_iou, is_low_light, embed_all,
      fuse_detections, assign_embeddings, vote, build_enhancer,
      maybe_enhance)
   6. Image enhancers  (ClaheEnhancer, GanEnhancer)
   7. Detectors        (ScrfdDetector, MtcnnDetector)
   8. Recognizers      (ArcFaceRecognizer, FaceNetRecognizer)
   9. Embedding stores (EmbeddingStore, SupabaseEmbeddingStore)
  10. Pipeline         (AttendancePipeline)  — main service class
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import cv2
import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


# Make the pip-installed NVIDIA CUDA/cuDNN DLLs discoverable by onnxruntime-gpu.
# InsightFace (SCRFD + ArcFace) builds its onnxruntime session here, which loads
# onnxruntime_providers_cuda.dll -> cublasLt64_12.dll / cudnn64_9.dll. Those ship
# in the ``nvidia-*-cu12`` wheels under ``site-packages/nvidia/*/bin`` but are not
# on the Windows DLL search path by default, so onnxruntime-gpu silently falls
# back to CPU. Prepend those dirs to PATH before the session is built. No-op when
# the wheels are absent (CPU-only machines), so this is safe for the whole team.
def _register_cuda_libs() -> None:
    import glob
    import site

    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())

    bin_dirs: list[str] = []
    for root in roots:
        bin_dirs += glob.glob(os.path.join(root, "nvidia", "*", "bin"))
    if not bin_dirs:
        return

    os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")
    for d in bin_dirs:
        try:
            os.add_dll_directory(d)  # type: ignore[attr-defined]  # Windows-only
        except (AttributeError, OSError):
            pass


_register_cuda_libs()

# Load .env once when this module is imported (was in config.py).
try:
    from dotenv import load_dotenv  # type: ignore

    _repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(_repo_root / ".env", override=False)
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════════════
# 1. AIConfig
# ════════════════════════════════════════════════════════════════════════
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


def _env_det_size(key: str, default: int) -> tuple[int, int]:
    """SCRFD detection input size. Accepts a single int (square, e.g. "1280")
    or "WxH" (e.g. "1280x1280"). Larger = better at detecting small / distant
    faces, at the cost of more compute per frame."""
    v = os.getenv(key)
    if not v:
        return (default, default)
    v = v.strip().lower()
    if "x" in v:
        w, h = v.split("x", 1)
        return (int(w), int(h))
    s = int(v)
    return (s, s)


def _env_tiles(key: str, default: tuple[int, int]) -> tuple[int, int]:
    """Detection tiling grid. "2x2" = 2 cols × 2 rows; a single int "2" = 2x2.
    1x1 (default) disables tiling."""
    v = os.getenv(key)
    if not v:
        return default
    v = v.strip().lower()
    if "x" in v:
        c, r = v.split("x", 1)
        return (max(1, int(c)), max(1, int(r)))
    n = max(1, int(v))
    return (n, n)


def _onnx_providers(ctx_id: int) -> list[str]:
    """ONNX Runtime execution providers for InsightFace. ctx_id >= 0 requests
    GPU: prefer CUDA with CPU fallback, but only if the installed runtime
    actually exposes CUDA — a CPU-only `onnxruntime` wheel will not, so we fall
    back to CPU rather than silently pretending to use the GPU."""
    if ctx_id < 0:
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    providers = []
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


@dataclass
class AIConfig:
    """Prototype AI configuration.

    Reads every tunable knob from environment variables with an `AI_` prefix
    so operators can override pipeline behaviour without touching code.
    """

    # ── Database ────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "")
    )

    # ── Hardware ────────────────────────────────────────────────────────
    ctx_id: int = field(default_factory=lambda: _env_int("AI_CTX_ID", -1))
    device: str = field(default_factory=lambda: os.getenv("AI_DEVICE", "cpu"))

    # ── Detection ───────────────────────────────────────────────────────
    # Larger det_size lets SCRFD see smaller / farther faces. Default 1280
    # (vs the old 640) roughly doubles the usable range for a 1080p camera.
    # Override with AI_DET_SIZE (e.g. "960", "1280", or "1920x1080").
    det_size: tuple[int, int] = field(
        default_factory=lambda: _env_det_size("AI_DET_SIZE", 1280)
    )
    det_thresh: float = field(default_factory=lambda: _env_float("AI_DET_THRESH", 0.5))
    use_mtcnn: bool = field(default_factory=lambda: _env_bool("AI_USE_MTCNN", True))
    # Tiled detection for long range: split each frame into cols×rows tiles and
    # run the detector on each, so distant (small) faces become large enough to
    # detect. Multiplies detection compute by cols*rows. "2x2" ≈ doubles range.
    # Default 1x1 = off (keeps CPU deployments fast). Set AI_DETECT_TILES=2x2.
    detect_tiles: tuple[int, int] = field(
        default_factory=lambda: _env_tiles("AI_DETECT_TILES", (1, 1))
    )
    detect_tile_overlap: float = field(
        default_factory=lambda: _env_float("AI_DETECT_TILE_OVERLAP", 0.15)
    )

    # ── Recognition ─────────────────────────────────────────────────────
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
    enhance_only_low_light: bool = field(
        default_factory=lambda: _env_bool("AI_ENHANCE_LOW_LIGHT_ONLY", True)
    )
    low_light_mean_threshold: float = field(
        default_factory=lambda: _env_float("AI_LOW_LIGHT_MEAN", 80.0)
    )

    # ── Match confidence ────────────────────────────────────────────────
    # Passing the similarity threshold is not enough: with a small gallery a
    # live face often sits just above it for SEVERAL people, and plain argmax
    # then reports the runner-up's neighbour with full confidence. Requiring
    # the winner to beat the second-best *different* account by this margin
    # turns those ambiguous cases into "Unknown" instead of a wrong name.
    # Set AI_MATCH_MARGIN=0 to restore the old argmax-only behaviour.
    match_margin: float = field(
        default_factory=lambda: _env_float("AI_MATCH_MARGIN", 0.05)
    )
    # Synthetic (GAN / seeded demo) embeddings are useful for calibration and
    # for populating demo screens, but they must not compete with real people
    # during a live scan. Set AI_MATCH_SYNTHETIC=true to include them.
    match_synthetic: bool = field(
        default_factory=lambda: _env_bool("AI_MATCH_SYNTHETIC", False)
    )

    # ── Ensemble ────────────────────────────────────────────────────────
    arcface_weight: float = field(
        default_factory=lambda: _env_float("AI_ARCFACE_WEIGHT", 0.65)
    )
    facenet_weight: float = field(
        default_factory=lambda: _env_float("AI_FACENET_WEIGHT", 0.35)
    )
    ensemble_iou: float = field(default_factory=lambda: _env_float("AI_ENSEMBLE_IOU", 0.4))

    # ── Anti-spoof (simple heuristic placeholder) ───────────────────────
    anti_spoof_min_face_px: int = field(
        default_factory=lambda: _env_int("AI_ANTISPOOF_MIN_FACE_PX", 50)
    )

    # ── PDPC / retention ────────────────────────────────────────────────
    embedding_retention_days: int = field(
        default_factory=lambda: _env_int("AI_EMBEDDING_RETENTION_DAYS", 365)
    )

    # ── Behaviour analysis (CR-06 / U32–U35) ────────────────────────────
    # Master switch for the BehaviourAnalysisService (MediaPipe FaceMesh +
    # YOLO phone detection). Default OFF so CPU deployments (Railway) pay
    # nothing; enable on the GPU rig with AI_BEHAVIOUR=true.
    behaviour_enabled: bool = field(
        default_factory=lambda: _env_bool("AI_BEHAVIOUR", False)
    )
    # Eyes count as closed below this EAR; a drowsiness episode needs the
    # signal to persist for ear_consec_seconds before it is recorded.
    ear_threshold: float = field(
        default_factory=lambda: _env_float("AI_EAR_THRESHOLD", 0.21)
    )
    ear_consec_seconds: float = field(
        default_factory=lambda: _env_float("AI_EAR_CONSEC_SECONDS", 2.0)
    )
    mar_threshold: float = field(
        default_factory=lambda: _env_float("AI_MAR_THRESHOLD", 0.6)
    )
    # Absolute head-down cutoff, used only when adaptive_headpose is off.
    # Measured 2026-08-04 (docs/evidence/behaviour_st_analysis.md): a ~40 deg
    # head-down posture is estimated by the 6-point solvePnP model as only
    # 12-20 deg, against a 6-8 deg upright baseline. The former default of 30
    # could therefore never fire — the head-pose signal was dead. 12 is the
    # highest cutoff that still recalls the labelled head-down segment while
    # keeping false alarms on the negative segments at zero.
    headpose_pitch_deg: float = field(
        default_factory=lambda: _env_float("AI_HEADPOSE_PITCH_DEG", 12.0)
    )
    # Where head pose comes from. "insightface" uses the 3D-landmark pose
    # from the detector that already found the face; "solvepnp" uses the
    # 6-point MediaPipe estimate. The latter is kept as a fallback for
    # deployments that cannot afford the extra model.
    headpose_source: str = field(
        default_factory=lambda: os.getenv("AI_HEADPOSE_SOURCE", "insightface").lower()
    )
    # Upright pitch differs per person and per camera mounting, exactly as
    # eye aperture does, so the same adaptive treatment applies: learn each
    # student's own upright pitch and flag departures from it.
    adaptive_headpose: bool = field(
        default_factory=lambda: _env_bool("AI_ADAPTIVE_HEADPOSE", True)
    )
    headpose_baseline_samples: int = field(
        default_factory=lambda: _env_int("AI_HEADPOSE_BASELINE_SAMPLES", 20)
    )
    # Degrees below the personal baseline that count as head-down. Swept
    # against the labelled clips: 10 recalls 100% of the head-down segment
    # with zero false alarms on all three negative segments. Lower values
    # start catching the small nod that accompanies eye closure; 15 and above
    # begins losing genuine head-down samples.
    headpose_delta_deg: float = field(
        default_factory=lambda: _env_float("AI_HEADPOSE_DELTA_DEG", 10.0)
    )
    # YOLO confidence for a detection to count as a phone. Measured against
    # labelled clips at 0.50: yolov8n and yolov8m both reach recall 1.00 with
    # zero false alarms, at either 1280 or 1920.
    #
    # Do NOT raise this to compensate for a noisy model. The 44% false-alarm
    # rate originally observed came from yolov8s specifically — it scores a
    # wall-mounted switch panel as a phone at up to 0.59, which neither
    # yolov8n nor yolov8m does at any input size. Raising the floor to 0.80
    # masks that for yolov8s but costs yolov8n three quarters of its recall
    # (1.00 -> 0.25), because a genuine phone does not always score high.
    # Pick a clean model instead; see phone_model.
    phone_conf: float = field(
        default_factory=lambda: _env_float("AI_PHONE_CONF", 0.50)
    )
    # A phone box that never moves is furniture, not a phone. Detections whose
    # position is essentially unchanged across this many samples are treated
    # as a fixed fixture and suppressed. Temporal debouncing cannot do this:
    # a persistent false positive is exactly what it confirms into an event.
    phone_static_samples: int = field(
        default_factory=lambda: _env_int("AI_PHONE_STATIC_SAMPLES", 45)
    )
    phone_static_iou: float = field(
        default_factory=lambda: _env_float("AI_PHONE_STATIC_IOU", 0.80)
    )
    # ── Drowsiness robustness (see behaviourAnalysis) ────────────────────
    # A fixed EAR cut-off is unreliable across people: eye aperture differs
    # more between individuals than between one person's alert and drowsy
    # states, so 0.21 marks some students as permanently asleep and never
    # trips for others. With adaptive_ear on, each student's own open-eye
    # median becomes the reference and closure is judged relative to it.
    adaptive_ear: bool = field(
        default_factory=lambda: _env_bool("AI_ADAPTIVE_EAR", True)
    )
    ear_baseline_samples: int = field(
        default_factory=lambda: _env_int("AI_EAR_BASELINE_SAMPLES", 20)
    )
    ear_baseline_ratio: float = field(
        default_factory=lambda: _env_float("AI_EAR_BASELINE_RATIO", 0.75)
    )
    # PERCLOS: share of samples with closed eyes over a rolling window. The
    # standard drowsiness measure — far steadier at ~1 fps than reacting to
    # any single frame, which is what a blink looks like.
    perclos_window_seconds: float = field(
        default_factory=lambda: _env_float("AI_PERCLOS_WINDOW_SECONDS", 60.0)
    )
    perclos_threshold: float = field(
        default_factory=lambda: _env_float("AI_PERCLOS_THRESHOLD", 0.40)
    )
    # Facial landmarks below this face size are noise; such samples are left
    # unanalysed (reported as "inconclusive") instead of guessed at.
    behaviour_min_face_px: int = field(
        default_factory=lambda: _env_int("AI_BEHAVIOUR_MIN_FACE_PX", 64)
    )
    # At ~1 fps sampling, N consecutive samples ≈ N seconds of phone use
    # before an episode is confirmed (debounces YOLO flicker).
    phone_consec_samples: int = field(
        default_factory=lambda: _env_int("AI_PHONE_CONSEC", 5)
    )
    # YOLO weights for phone detection. Measured on labelled clips at
    # conf 0.50 (recall / false-alarm rate on negative segments):
    #
    #   yolov8n  1.00 / 0.000   yolov8m  1.00 / 0.000
    #   yolov8s  1.00 / 0.267   yolov8l  1.00 / 0.106
    #
    # Bigger is NOT reliably better here: yolov8s and yolov8l both mistake a
    # wall switch panel for a phone, and yolov8s does so confidently. yolov8n
    # is the right default and yolov8m the right upgrade on a GPU rig.
    phone_model: str = field(
        default_factory=lambda: os.getenv("AI_PHONE_MODEL", "yolov8n.pt")
    )
    # YOLO inference size. The ultralytics default (640) halves a 720p frame
    # and makes phones beyond ~3 m too small to detect; 1280 keeps them at
    # native resolution at ~4x the (still tiny) nano compute cost.
    phone_imgsz: int = field(
        default_factory=lambda: _env_int("AI_PHONE_IMGSZ", 1280)
    )
    heatmap_grid: tuple[int, int] = field(
        default_factory=lambda: _env_tiles("AI_HEATMAP_GRID", (8, 6))
    )
    heatmap_flush_seconds: int = field(
        default_factory=lambda: _env_int("AI_HEATMAP_FLUSH_SECONDS", 60)
    )

    def log_summary(self) -> str:
        return (
            f"AIConfig(ctx_id={self.ctx_id}, device={self.device}, "
            f"det_thresh={self.det_thresh}, "
            f"arcface(th={self.arcface_threshold}, w={self.arcface_weight}), "
            f"facenet(enabled={self.use_facenet}, th={self.facenet_threshold}, "
            f"w={self.facenet_weight}), "
            f"mtcnn={self.use_mtcnn}, enhancer={self.enhancer_kind}"
            f"{'(on)' if self.use_enhancer else '(off)'}, "
            f"behaviour={'on' if self.behaviour_enabled else 'off'})"
        )


# ════════════════════════════════════════════════════════════════════════
# 2. EmbeddingRow + EmbeddingRepo
# ════════════════════════════════════════════════════════════════════════
@dataclass
class EmbeddingRow:
    face_id: int
    account_id: int
    student_id: str | None
    full_name: str | None
    model_name: str
    model_version: str
    dimension: int
    vector: np.ndarray


def vector_to_np(value) -> np.ndarray:
    """Coerce a pgvector column value to float32 across pgvector-python
    versions: 0.2/0.3 hand back np.ndarray, 0.4+ hands back a Vector
    object (with .to_numpy()), and without register_vector it's the raw
    '[0.1,0.2,…]' text."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=np.float32)
    if isinstance(value, str):
        return np.asarray(
            [float(x) for x in value.strip()[1:-1].split(",")], dtype=np.float32
        )
    return np.asarray(value, dtype=np.float32)


class EmbeddingRepo:
    """Thin repository over the FACE_EMBEDDING + PERSONAL_INFO tables."""

    def __init__(self, database_url: str):
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set; add it to .env before starting the AI pipeline."
            )
        self._database_url = database_url

    @contextlib.contextmanager
    def _conn(self) -> Iterator[psycopg2.extensions.connection]:
        conn = psycopg2.connect(self._database_url)
        try:
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ping(self) -> bool:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1

    def load_active_embeddings(
        self, model_name: str, include_synthetic: bool = False,
    ) -> list[EmbeddingRow]:
        """Gallery rows for one model.

        Synthetic rows (GAN-generated or demo-seeded, ``is_synthetic``) are
        excluded by default: they exist to populate screens and to calibrate
        thresholds, and letting them compete with real students during a live
        scan can only produce wrong identifications.
        """
        sql = """
            SELECT f.faceid, f.accountid, p.student_id, p.full_name,
                   f.model_name, f.model_version, f.dimension,
                   f.embedding_vector
            FROM face_embedding f
            LEFT JOIN personal_info p ON p.accountid = f.accountid
            WHERE f.is_active = TRUE AND f.model_name = %s
        """
        if not include_synthetic:
            sql += " AND f.is_synthetic = FALSE"
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (model_name,))
            rows = cur.fetchall()
        print(f"  [db] model={model_name}  rows_fetched={len(rows)}")
        out: list[EmbeddingRow] = []
        for face_id, account_id, student_id, full_name, m_name, m_ver, dim, vec in rows:
            arr = vector_to_np(vec)
            if arr.size != dim:
                print(f"  [db] WARNING: skipping faceid={face_id} "
                      f"— vector has {arr.size} floats, expected {dim}")
                continue
            out.append(EmbeddingRow(
                face_id=face_id, account_id=account_id,
                student_id=student_id, full_name=full_name,
                model_name=m_name, model_version=m_ver,
                dimension=dim, vector=arr,
            ))
        return out

    def load_student_info(self, account_id: int) -> "StudentInfo | None":
        """Name and student number for one account, or None if unknown.

        Used when a face is enrolled while the server is running, so the new
        gallery entry is labelled straight away instead of staying anonymous
        until the next restart re-hydrates the store.
        """
        sql = """
            SELECT accountid, student_id, full_name
            FROM personal_info WHERE accountid = %s LIMIT 1
        """
        try:
            with self._conn() as c, c.cursor() as cur:
                cur.execute(sql, (account_id,))
                row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001 — enrolment must not fail here
            print(f"  [db] could not load personal_info for {account_id}: {exc}")
            return None
        if not row:
            return None
        return StudentInfo(account_id=row[0], student_id=row[1], full_name=row[2])

    def find_account_by_student_id(self, student_id: str) -> int | None:
        sql = "SELECT accountid FROM personal_info WHERE student_id = %s LIMIT 1"
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (student_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def deactivate_embeddings(self, account_id: int, model_name: str) -> int:
        sql = """
            UPDATE face_embedding
            SET is_active = FALSE, updated_at = NOW()
            WHERE accountid = %s AND model_name = %s AND is_active = TRUE
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (account_id, model_name))
            return cur.rowcount

    def insert_embedding(
        self, account_id: int, vec: np.ndarray,
        model_name: str, model_version: str, retention_days: int,
    ) -> int:
        sql = """
            INSERT INTO face_embedding
              (accountid, embedding_vector, model_name, model_version, dimension,
               is_active, consent_given_at, retention_until)
            VALUES (%s, %s, %s, %s, %s, TRUE, NOW(), %s)
            RETURNING faceid
        """
        vec = np.asarray(vec, dtype=np.float32)
        retention = dt.date.today() + dt.timedelta(days=retention_days)
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (
                account_id, vec, model_name, model_version,
                int(vec.size), retention,
            ))
            return cur.fetchone()[0]

    def mark_attendance(
        self, session_id: int, account_id: int, status: str,
    ) -> None:
        """Best-effort insert of an attendance record, ignoring duplicates."""
        sql = """
            INSERT INTO attendance_record (attendancesessionid, accountid, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (attendancesessionid, accountid) DO NOTHING
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (session_id, account_id, status))

    def get_or_open_session(self, course_id: int) -> int:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                """SELECT attendancesessionid FROM attendance_session
                   WHERE courseid = %s AND status = 'active'
                   ORDER BY start_time DESC LIMIT 1""",
                (course_id,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute(
                """INSERT INTO attendance_session (courseid, start_time, status)
                   VALUES (%s, NOW(), 'active') RETURNING attendancesessionid""",
                (course_id,),
            )
            return cur.fetchone()[0]


# ════════════════════════════════════════════════════════════════════════
# 3. Constants
# ════════════════════════════════════════════════════════════════════════
ARCFACE_MODEL_NAME = "arcface"
ARCFACE_MODEL_VERSION = "buffalo_l/r100"
ARCFACE_DIM = 512

FACENET_MODEL_NAME = "facenet"
FACENET_MODEL_VERSION = "vggface2"
FACENET_DIM = 512


# ════════════════════════════════════════════════════════════════════════
# 4. Lightweight data classes
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    bbox: np.ndarray
    det_score: float
    kps: np.ndarray | None
    source: str  # "scrfd" | "mtcnn"
    raw: Any | None = None


@dataclass
class Embedding:
    detection: Detection
    vector: np.ndarray
    model_name: str
    model_version: str


@dataclass
class FaceGroup:
    bbox: np.ndarray
    det_scores: dict[str, float] = field(default_factory=dict)
    embeddings: list[Embedding] = field(default_factory=list)

    def merged_bbox(self) -> np.ndarray:
        return self.bbox

    def average_det_score(self) -> float:
        if not self.det_scores:
            return 0.0
        return sum(self.det_scores.values()) / len(self.det_scores)

    def head_pitch(self) -> float | None:
        """Pitch in degrees from the detector's 3D landmarks, if available.

        Present only when the `landmark_3d_68` module was loaded (behaviour
        analysis on); None otherwise, and the caller falls back to the
        MediaPipe estimate. Negative means head tilted downward.
        """
        for emb in self.embeddings:
            raw = getattr(emb.detection, "raw", None)
            pose = getattr(raw, "pose", None) if raw is not None else None
            if pose is not None:
                return float(pose[0])
        return None


@dataclass
class Prediction:
    bbox: list[int]
    recognised: bool
    account_id: int | None
    student_id: str | None
    full_name: str | None
    score: float
    per_model: dict[str, dict]
    det_score: float
    # Head pitch in degrees from the detector's 3D landmarks (negative =
    # head down). None when that module is not loaded; the behaviour branch
    # then falls back to its own MediaPipe estimate.
    head_pitch: float | None = None


@dataclass
class StudentInfo:
    account_id: int
    student_id: str | None
    full_name: str | None


@dataclass
class FrameResult:
    predictions: list[Prediction]
    enhanced: bool
    detections: list[Detection]


class MultipleFacesError(Exception):
    """Raised during enrolment when more than one face is in the photo."""

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"{count} faces detected")


# ════════════════════════════════════════════════════════════════════════
# 5. Pure helpers
# ════════════════════════════════════════════════════════════════════════
def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


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


def is_low_light(frame: np.ndarray, mean_threshold: float) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean()) < mean_threshold


def embed_all(
    recognizer, frame: np.ndarray, detections: Iterable[Detection],
) -> list[Embedding]:
    out: list[Embedding] = []
    for d in detections:
        vec = recognizer.embed(frame, d)
        if vec is None:
            continue
        out.append(Embedding(
            detection=d, vector=vec,
            model_name=recognizer.model_name,
            model_version=recognizer.model_version,
        ))
    return out


def fuse_detections(
    detections: list[Detection], iou_threshold: float,
) -> list[FaceGroup]:
    groups: list[FaceGroup] = []
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
            groups.append(FaceGroup(
                bbox=det.bbox.copy(),
                det_scores={det.source: det.det_score},
            ))
    return groups


def assign_embeddings(
    groups: list[FaceGroup], embeddings: list[Embedding], iou_threshold: float,
) -> None:
    for emb in embeddings:
        best: tuple[float, FaceGroup | None] = (0.0, None)
        for g in groups:
            iou = bbox_iou(emb.detection.bbox, g.bbox)
            if iou > best[0]:
                best = (iou, g)
        if best[1] is not None and best[0] >= iou_threshold * 0.75:
            best[1].embeddings.append(emb)


def vote(
    groups: list[FaceGroup],
    stores: dict[str, "EmbeddingStore"],
    weights: dict[str, float],
    cfg: AIConfig,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for group in groups:
        per_model: dict[str, dict] = {}
        vote_bag: dict[int, float] = {}
        info_bag: dict[int, StudentInfo] = {}
        weight_bag: dict[int, float] = {}

        for emb in group.embeddings:
            store = stores.get(emb.model_name)
            if store is None:
                continue
            account_id, score = store.best_match(emb.vector)
            # Runner-up scores for the "Match diagnostics" teacher view — lets
            # an "Unknown" box be read as either "nobody enrolled is close"
            # (low scores across the board) or "margin rule refusing to guess
            # between look-alikes" (two close, similar scores), instead of an
            # opaque rejection.
            candidates = []
            for cand_id, cand_score in store.rank(emb.vector, top_k=2):
                cand_info = store.info_for(cand_id)
                candidates.append({
                    "account_id": cand_id,
                    "score": cand_score,
                    "student_id": cand_info.student_id if cand_info else None,
                    "full_name": cand_info.full_name if cand_info else None,
                })
            per_model[emb.model_name] = {
                "matched": account_id is not None,
                "account_id": account_id,
                "score": score,
                "threshold": store.threshold,
                "candidates": candidates,
            }
            if account_id is None:
                continue
            w = weights.get(emb.model_name, 1.0)
            vote_bag[account_id] = vote_bag.get(account_id, 0.0) + w * score
            weight_bag[account_id] = weight_bag.get(account_id, 0.0) + w
            info = store.info_for(account_id)
            if info is not None and account_id not in info_bag:
                info_bag[account_id] = info

        if not vote_bag:
            predictions.append(Prediction(
                bbox=group.bbox.tolist(), recognised=False,
                account_id=None, student_id=None, full_name=None,
                score=0.0, per_model=per_model,
                det_score=group.average_det_score(),
                head_pitch=group.head_pitch(),
            ))
            continue

        winner = max(vote_bag.items(), key=lambda kv: kv[1])
        account_id = winner[0]
        fused = vote_bag[account_id] / max(weight_bag[account_id], 1e-6)
        info = info_bag.get(account_id)
        predictions.append(Prediction(
            bbox=group.bbox.tolist(), recognised=True,
            account_id=account_id,
            student_id=info.student_id if info else None,
            full_name=info.full_name if info else None,
            score=fused, per_model=per_model,
            det_score=group.average_det_score(),
            head_pitch=group.head_pitch(),
        ))
    return predictions


# ════════════════════════════════════════════════════════════════════════
# 6. Image enhancers
# ════════════════════════════════════════════════════════════════════════
class EnhancerProto(Protocol):
    name: str

    def enhance(self, frame: np.ndarray) -> np.ndarray: ...


class _BaseEnhancer(ABC):
    name: str = "base"

    @abstractmethod
    def enhance(self, frame: np.ndarray) -> np.ndarray: ...


class ClaheEnhancer(_BaseEnhancer):
    """CLAHE on the L channel of LAB space, then a mild unsharp mask."""

    name = "clahe"

    def __init__(self, clip_limit: float = 2.5, tile_grid: tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        lab = cv2.merge((l_ch, a_ch, b_ch))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
        out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
        return out


class GanEnhancer(_BaseEnhancer):
    """Thin wrapper around a face-restoration GAN (GFPGAN / Real-ESRGAN)."""

    name = "gan"

    def __init__(self, backend: str = "auto", device: str = "cpu"):
        self._backend_name, self._runner = self._load_backend(backend, device)
        self.name = f"gan/{self._backend_name}"

    @staticmethod
    def _load_backend(backend: str, device: str):
        if backend in ("auto", "gfpgan"):
            try:
                from gfpgan import GFPGANer  # type: ignore

                runner = GFPGANer(
                    model_path=None, upscale=1, arch="clean",
                    channel_multiplier=2, bg_upsampler=None,
                )

                def _run(img: np.ndarray) -> np.ndarray:
                    _, _, restored = runner.enhance(
                        img, has_aligned=False, only_center_face=False,
                        paste_back=True,
                    )
                    return restored if restored is not None else img

                return "gfpgan", _run
            except Exception:
                if backend == "gfpgan":
                    raise

        if backend in ("auto", "realesrgan"):
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
                from realesrgan import RealESRGANer  # type: ignore

                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2,
                )
                runner = RealESRGANer(
                    scale=2, model_path=None, model=model,
                    tile=0, half=False, device=device,
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
            return frame


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


def maybe_enhance(
    frame: np.ndarray,
    enhancer: _BaseEnhancer | None,
    cfg: AIConfig,
) -> tuple[np.ndarray, bool]:
    if enhancer is None:
        return frame, False
    if cfg.enhance_only_low_light and not is_low_light(
        frame, cfg.low_light_mean_threshold
    ):
        return frame, False
    return enhancer.enhance(frame), True


# ════════════════════════════════════════════════════════════════════════
# 7. Detectors
# ════════════════════════════════════════════════════════════════════════
def _build_face_analysis(cfg: AIConfig):
    """Construct the shared InsightFace app (SCRFD detection + ArcFace
    recognition) with explicit providers, and log which device each model
    actually runs on so a CPU-only onnxruntime install can't hide as 'GPU'."""
    from insightface.app import FaceAnalysis  # deferred

    providers = _onnx_providers(cfg.ctx_id)
    modules = ["detection", "recognition"]
    if cfg.behaviour_enabled and cfg.headpose_source == "insightface":
        # 3D landmarks give a head pose that survives what the behaviour
        # branch actually needs to detect. Measured 2026-08-04: on frames of
        # a student with their head down, MediaPipe returned no landmarks at
        # all for 60% of them, while this model produced a pose for 100% and
        # separated upright (+4.9 deg) from head-down (-19.9 deg) four times
        # more widely. It costs one extra inference per face, so it is loaded
        # only when behaviour analysis is switched on.
        modules.append("landmark_3d_68")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=modules,
        providers=providers,
    )
    app.prepare(ctx_id=cfg.ctx_id, det_size=cfg.det_size, det_thresh=cfg.det_thresh)
    try:
        active = {
            task: m.session.get_providers()[0]
            for task, m in app.models.items()
            if getattr(m, "session", None) is not None
        }
        on_gpu = any("CUDA" in p for p in active.values())
        print(f"[insightface] ctx_id={cfg.ctx_id} requested={providers} "
              f"active={active} -> {'GPU' if on_gpu else 'CPU'}")
        if cfg.ctx_id >= 0 and not on_gpu:
            print("[insightface] WARNING: GPU requested but running on CPU. "
                  "Install onnxruntime-gpu (and a matching CUDA/cuDNN) — the "
                  "plain `onnxruntime` wheel is CPU-only.")
    except Exception:
        pass
    return app


class ScrfdDetector:
    name = "scrfd"

    def __init__(self, cfg: AIConfig, shared_app=None):
        self.app = shared_app if shared_app is not None else _build_face_analysis(cfg)

    @staticmethod
    def _to_detection(f, ox: int = 0, oy: int = 0) -> Detection:
        """Wrap an InsightFace face as a Detection, shifting bbox/kps by the
        tile origin (ox, oy) so tiled coords map back to the full frame. The
        face's `raw.normed_embedding` is alignment-invariant, so recognition
        still works from a tile crop."""
        bbox = f.bbox.astype(int).copy()
        bbox[0] += ox
        bbox[1] += oy
        bbox[2] += ox
        bbox[3] += oy
        kps = None
        if getattr(f, "kps", None) is not None:
            kps = np.asarray(f.kps, dtype=float).copy()
            kps[:, 0] += ox
            kps[:, 1] += oy
        return Detection(bbox=bbox, det_score=float(f.det_score),
                         kps=kps, source="scrfd", raw=f)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [self._to_detection(f) for f in self.app.get(frame)]

    def detect_tiled(
        self, frame: np.ndarray, cols: int = 1, rows: int = 1, overlap: float = 0.15,
    ) -> list[Detection]:
        """Run detection on a cols×rows grid of (overlapping) tiles and merge.
        Overlapping tiles avoid missing faces straddling a tile boundary;
        duplicate detections are deduplicated later by fuse_detections."""
        if cols <= 1 and rows <= 1:
            return self.detect(frame)
        h, w = frame.shape[:2]
        tw, th = w / cols, h / rows
        ox_pad, oy_pad = int(tw * overlap), int(th * overlap)
        out: list[Detection] = []
        for r in range(rows):
            for c in range(cols):
                x0 = max(0, int(c * tw) - ox_pad)
                y0 = max(0, int(r * th) - oy_pad)
                x1 = min(w, int((c + 1) * tw) + ox_pad)
                y1 = min(h, int((r + 1) * th) + oy_pad)
                tile = frame[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                for f in self.app.get(tile):
                    out.append(self._to_detection(f, x0, y0))
        return out


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
            keep_all=True, device=device, post_process=False,
            min_face_size=cfg.anti_spoof_min_face_px,
            thresholds=[0.6, 0.7, 0.7],
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)
        if boxes is None:
            return []
        detections: list[Detection] = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            if box is None or prob is None:
                continue
            x1, y1, x2, y2 = box
            detections.append(Detection(
                bbox=np.array([int(x1), int(y1), int(x2), int(y2)], dtype=int),
                det_score=float(prob),
                kps=np.asarray(lm, dtype=float) if lm is not None else None,
                source="mtcnn",
            ))
        return detections


# ════════════════════════════════════════════════════════════════════════
# 8. Recognizers
# ════════════════════════════════════════════════════════════════════════
class ArcFaceRecognizer:
    model_name = ARCFACE_MODEL_NAME
    model_version = ARCFACE_MODEL_VERSION
    dim = ARCFACE_DIM

    def __init__(self, cfg: AIConfig, shared_app=None):
        if shared_app is None:
            self.app = _build_face_analysis(cfg)
        else:
            self.app = shared_app

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
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
        face = (face - 127.5) / 128.0
        t = self._torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self._model(t).cpu().numpy()[0]
        return _normalise(emb)


# ════════════════════════════════════════════════════════════════════════
# 9. Embedding stores
# ════════════════════════════════════════════════════════════════════════
class EmbeddingStore:
    """In-memory matrix of (N, dim) for a single recognition model."""

    def __init__(
        self, model_name: str, dim: int, threshold: float, margin: float = 0.0,
    ):
        self.model_name = model_name
        self.dim = dim
        self.threshold = threshold
        # Minimum lead the winner must have over the runner-up account.
        self.margin = margin
        self._ids: list[int] = []
        self._info: dict[int, StudentInfo] = {}
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def load_rows(self, rows: list[EmbeddingRow]) -> None:
        if not rows:
            self._ids = []
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            return
        self._ids = [r.account_id for r in rows]
        self._matrix = np.stack(
            [_normalise(r.vector) for r in rows], axis=0
        ).astype(np.float32)
        for r in rows:
            self._info[r.account_id] = StudentInfo(
                account_id=r.account_id,
                student_id=r.student_id, full_name=r.full_name,
            )

    def upsert(
        self, account_id: int, vector: np.ndarray,
        info: StudentInfo | None = None, replace: bool = True,
    ) -> None:
        """Add a gallery entry.

        ``replace=True`` (default) keeps one vector per account — the
        behaviour for "update my photo". ``replace=False`` appends another
        entry, letting a student's gallery cover several angles; matching
        then takes the best-scoring entry per account (see rank()).
        """
        vec = _normalise(vector).astype(np.float32)
        if replace and account_id in self._ids:
            idx = self._ids.index(account_id)
            self._matrix[idx] = vec
        else:
            self._ids.append(account_id)
            self._matrix = np.vstack([self._matrix, vec[None, :]])
        if info is not None:
            self._info[account_id] = info

    def rank(self, query: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        """Best score per ACCOUNT, highest first.

        A student may hold several gallery entries (different angles and
        lighting), so scores are collapsed per account — otherwise one
        well-enrolled person could occupy every top slot and mask the real
        runner-up that the margin test below has to consider.
        """
        if self._matrix.shape[0] == 0:
            return []
        q = _normalise(query).astype(np.float32)
        scores = self._matrix @ q
        best_per_account: dict[int, float] = {}
        for account_id, score in zip(self._ids, scores):
            s = float(score)
            if s > best_per_account.get(account_id, -2.0):
                best_per_account[account_id] = s
        ranked = sorted(best_per_account.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def best_match(self, query: np.ndarray) -> tuple[int | None, float]:
        """Identify a face, or return (None, best_score) when unsure.

        Two conditions must hold, not one:
          1. the top score clears `threshold` (is this anyone we know?), and
          2. it beats the runner-up account by `margin` (are we sure WHICH
             one?). Without (2) a face that scores 0.45 against student A and
             0.42 against its true owner B is reported confidently as A.
        """
        ranked = self.rank(query, top_k=2)
        if not ranked:
            return None, 0.0
        account_id, score = ranked[0]
        if score < self.threshold:
            return None, score
        if len(ranked) > 1 and (score - ranked[1][1]) < self.margin:
            # Too close to call — an unknown face resembling two students is
            # far more likely than a genuine tie between them.
            return None, score
        return account_id, score

    def info_for(self, account_id: int) -> StudentInfo | None:
        return self._info.get(account_id)

    def __len__(self) -> int:
        return len(self._ids)


@dataclass
class SupabaseEmbeddingStore:
    """Per-model stores hydrated from the Supabase FACE_EMBEDDING table."""

    repo: EmbeddingRepo
    stores: dict[str, EmbeddingStore] = field(default_factory=dict)
    # Whether synthetic gallery rows take part in live matching (see AIConfig).
    include_synthetic: bool = False

    def register_store(self, store: EmbeddingStore) -> None:
        self.stores[store.model_name] = store

    def hydrate_all(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for name, store in self.stores.items():
            rows = self.repo.load_active_embeddings(
                name, include_synthetic=self.include_synthetic
            )
            store.load_rows(rows)
            summary[name] = len(rows)
        return summary

    def reload(self) -> dict[str, int]:
        return self.hydrate_all()

    def get(self, model_name: str) -> EmbeddingStore:
        return self.stores[model_name]

    def enrol_account(
        self,
        account_id: int,
        vectors_by_model: dict[str, list[np.ndarray]],
        retention_days: int,
        model_versions: dict[str, str],
        deactivate_previous: bool = True,
    ) -> dict[str, int]:
        written: dict[str, int] = {}
        # Looked up once per enrolment (a rare, interactive operation) so the
        # in-memory gallery carries the same name the database holds.
        info = self.repo.load_student_info(account_id)
        for model_name, vectors in vectors_by_model.items():
            if not vectors:
                continue
            if deactivate_previous:
                self.repo.deactivate_embeddings(account_id, model_name)
            avg = _normalise(np.mean(vectors, axis=0))
            face_id = self.repo.insert_embedding(
                account_id=account_id, vec=avg,
                model_name=model_name,
                model_version=model_versions.get(model_name, "unknown"),
                retention_days=retention_days,
            )
            written[model_name] = face_id
            store = self.stores.get(model_name)
            if store is not None:
                # Appending adds a gallery entry rather than replacing the
                # account's existing one, so the in-memory matrix must gain a
                # row too — upsert() would overwrite the previous angle.
                #
                # The name has to come along. Without it the account is
                # matchable but anonymous, and every label falls back to
                # "acc#<id>" until the process restarts and re-hydrates from
                # the database — which looks like the recogniser failing when
                # it is actually working perfectly.
                store.upsert(account_id, avg, info=info,
                             replace=deactivate_previous)
        return written


# ════════════════════════════════════════════════════════════════════════
# 10. AttendancePipeline
# ════════════════════════════════════════════════════════════════════════
class AttendancePipeline:
    """End-to-end inference pipeline (Service Class).

    Loaded once at app startup (main_api.lifespan) and stashed on
    app.state.pipeline so every request reuses the same model weights.

    Per frame:
        GAN/CLAHE enhance (low-light only)
          └► SCRFD detect ──►┐
          └► MTCNN detect ──►┤  fuse by IoU
                             ├► ArcFace embed ──►┐
                             └► FaceNet embed ──►┤  weighted vote
                                                 ▼
                                          Prediction list
    """

    def __init__(self, cfg: AIConfig, store_manager: SupabaseEmbeddingStore):
        self.cfg = cfg
        self.store_manager = store_manager
        self.enhancer = build_enhancer(cfg)
        # SCRFD + ArcFace share one FaceAnalysis instance.
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
        manager = SupabaseEmbeddingStore(
            repo=repo, include_synthetic=cfg.match_synthetic
        )
        manager.register_store(
            EmbeddingStore(
                ARCFACE_MODEL_NAME, ARCFACE_DIM,
                cfg.arcface_threshold, cfg.match_margin,
            )
        )
        if cfg.use_facenet:
            manager.register_store(
                EmbeddingStore(
                    FACENET_MODEL_NAME, FACENET_DIM,
                    cfg.facenet_threshold, cfg.match_margin,
                )
            )
        loaded = manager.hydrate_all()
        print(f"[pipeline] DB hydrate: {loaded}")
        # U25: an admin-deployed similarity threshold (MODEL_CONFIGS) takes
        # precedence over the env default so deployments survive restarts.
        from core.trainConfiguration import load_deployed_threshold

        for name, store in manager.stores.items():
            deployed = load_deployed_threshold(cfg.database_url, name)
            if deployed is not None:
                store.threshold = deployed
                print(f"[pipeline] {name}: deployed threshold {deployed} applied")
        return cls(cfg, manager)

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        frame_in, was_enhanced = maybe_enhance(frame, self.enhancer, self.cfg)
        cols, rows = self.cfg.detect_tiles
        scrfd_dets = self._scrfd.detect_tiled(
            frame_in, cols, rows, self.cfg.detect_tile_overlap,
        )
        mtcnn_dets = self._mtcnn.detect(frame_in) if self._mtcnn is not None else []
        min_px = self.cfg.anti_spoof_min_face_px
        dets = [
            d for d in (scrfd_dets + mtcnn_dets)
            if (d.bbox[2] - d.bbox[0]) >= min_px
               and (d.bbox[3] - d.bbox[1]) >= min_px
        ]
        groups = fuse_detections(dets, self.cfg.ensemble_iou)
        anchors = self._group_anchors(groups, dets)
        embeddings: list[Embedding] = []
        embeddings.extend(embed_all(self._arcface, frame_in, anchors))
        if self._facenet is not None:
            embeddings.extend(embed_all(self._facenet, frame_in, anchors))
        assign_embeddings(groups, embeddings, self.cfg.ensemble_iou)
        predictions = vote(
            groups, self.store_manager.stores, self._weights, self.cfg,
        )
        return FrameResult(
            predictions=predictions, enhanced=was_enhanced, detections=dets,
        )

    def enrol_student(
        self, account_id: int, images: list[np.ndarray],
        reject_multiple: bool = False, append: bool = False,
    ) -> dict[str, int]:
        """Enrol a student's face.

        ``append=False`` (default) replaces the student's gallery — the right
        behaviour for "update my photo". ``append=True`` keeps the existing
        entries and adds another one, which is how a student builds a gallery
        covering several angles and lighting conditions. A single-entry
        gallery is the main driver of look-alike confusion: a live frame taken
        under different conditions drifts away from that one vector, and the
        person whose single photo happens to match the room best then wins
        everyone else's comparisons too.
        """
        vectors: dict[str, list[np.ndarray]] = {ARCFACE_MODEL_NAME: []}
        if self._facenet is not None:
            vectors[FACENET_MODEL_NAME] = []
        for img in images:
            frame_in, _ = maybe_enhance(img, self.enhancer, self.cfg)
            scrfd_dets = self._scrfd.detect(frame_in)
            if not scrfd_dets:
                continue
            if reject_multiple and len(scrfd_dets) > 1:
                raise MultipleFacesError(len(scrfd_dets))
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
            deactivate_previous=not append,
        )
        return written

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
                if p.recognised else "Unknown"
            )
            label_2 = f"fused={p.score:.2f}  det={p.det_score:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, label_1, (x1, max(0, y1 - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(out, label_2, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return out

    @staticmethod
    def _group_anchors(
        groups: list[FaceGroup], detections: list[Detection],
    ) -> list[Detection]:
        # Prefer a SCRFD detection whenever one overlaps the group: SCRFD
        # carries ArcFace's embedding for free via raw.normed_embedding, and
        # FaceNet can crop from any bbox. Picking by raw det_score alone makes
        # MTCNN (probabilities ~0.99) always win over SCRFD (~0.8-0.9), which
        # leaves ArcFace with a raw=None box, forces a fragile re-detect that
        # fails, and silently degrades the ensemble to FaceNet-only. Only fall
        # back to the best MTCNN box for faces SCRFD missed entirely.
        anchors: list[Detection] = []
        for g in groups:
            scrfd_best: tuple[float, Detection | None] = (-1.0, None)
            mtcnn_best: tuple[float, Detection | None] = (-1.0, None)
            for d in detections:
                if bbox_iou(d.bbox, g.bbox) == 0:
                    continue
                bucket = scrfd_best if d.source == "scrfd" else mtcnn_best
                if d.det_score > bucket[0]:
                    if d.source == "scrfd":
                        scrfd_best = (d.det_score, d)
                    else:
                        mtcnn_best = (d.det_score, d)
            pick = scrfd_best[1] if scrfd_best[1] is not None else mtcnn_best[1]
            if pick is not None:
                anchors.append(pick)
        return anchors
