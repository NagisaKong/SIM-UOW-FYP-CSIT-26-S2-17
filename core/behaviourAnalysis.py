"""Class 8: behaviourAnalysis  (renamed from inClassBehaviour)

Covers UC:
  U32 View Classroom Behaviour Analysis Report (viewBehaviourReport)
  U33 View Classroom Activity Heatmap          (viewHeatmap)
  U35 Configure Classroom Behaviour Analysis   (enableBehaviourAnalysis /
                                                disableBehaviourAnalysis)
  CR-06 Behaviour detection service            (BehaviourAnalysisService)

Layout:
  1. Config loader + InClassBehaviour business class (U32/U33/U35 — reads
     behaviour_event / heatmap_snapshot, toggles behaviour_config).
  2. Pure geometry helpers (EAR / MAR / head pose) — dependency-free and
     unit-testable.
  3. Detectors: DrowsinessDetector (MediaPipe FaceMesh), PhoneDetector
     (Ultralytics YOLO, COCO class 67 "cell phone"). Both lazy-load their
     models on first frame and degrade to disabled if the package is absent.
  4. Temporal state: _EpisodeTracker debounces per-student signals so one
     event row spans one episode (no per-frame spam).
  5. BehaviourAnalysisService — the per-frame entry point wired to
     POST /teacher/sessions/{id}/behaviour-scan.

Identity attribution (CR-06): every behaviour observation is attached to a
student ALREADY recognised by AttendancePipeline in the same frame (bbox
association). There is no separate tracking subsystem.

Privacy (PDPC): frames are processed in memory only. This module never
writes image data anywhere — only derived event tuples (behaviour_event)
and aggregated grid intensities (heatmap_snapshot) are persisted.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import psycopg2
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from psycopg2.extras import Json
from pydantic import BaseModel

from core.userInformation import CurrentUser, require_role

if TYPE_CHECKING:  # avoid importing the heavy pipeline module at runtime
    from core.attendancePipeline import AIConfig, AttendancePipeline, Prediction


_DEFAULT_FLAGS = {
    "enabled": False,
    "drowsiness": False,
    "phone_usage": False,
    "heatmap": False,
}

# Per-course detection tuning (U35). NULL in the database means "inherit the
# server-wide AIConfig value", so None is preserved rather than defaulted here.
_TUNING_KEYS = (
    "ear_threshold",
    "mar_threshold",
    "headpose_pitch_deg",
    "headpose_delta_deg",
    "phone_conf",
    "drowsy_confirm_seconds",
)
# Boolean tuning flags, handled alongside _TUNING_KEYS wherever both are
# needed (column lists, request parsing) but kept separate because they are
# not FLOAT columns.
_TUNING_BOOL_KEYS = ("adaptive_ear", "adaptive_headpose")


def load_behaviour_config(database_url: str, course_id: int) -> dict[str, bool]:
    """Read the per-course behaviour-analysis switch from behaviour_config.

    Returns _DEFAULT_FLAGS (all-off) when no row exists or the table is
    missing so callers get a consistent dict shape.
    """
    try:
        with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT enabled, drowsiness, phone_usage, heatmap,
                          adaptive_ear, adaptive_headpose, {", ".join(_TUNING_KEYS)}
                   FROM behaviour_config WHERE courseid = %s""",
                (course_id,),
            )
            row = cur.fetchone()
            if row:
                flags = {
                    "enabled": bool(row[0]),
                    "drowsiness": bool(row[1]),
                    "phone_usage": bool(row[2]),
                    "heatmap": bool(row[3]),
                    "adaptive_ear": bool(row[4]),
                    "adaptive_headpose": bool(row[5]),
                }
                # None is meaningful: it means "no override, use AIConfig".
                for offset, key in enumerate(_TUNING_KEYS, start=6):
                    flags[key] = None if row[offset] is None else float(row[offset])
                return flags
    except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn,
            psycopg2.OperationalError) as exc:
        # Older database without the tuning columns — fall back to the basic
        # switches so behaviour analysis still works before the migration.
        print(f"[behaviour] tuning columns unavailable ({exc}); using base config")
        return _load_basic_flags(database_url, course_id)
    except Exception as exc:  # noqa: BLE001 — config read must never 500
        print(f"[behaviour] config read failed, defaulting to off: {exc}")
    return dict(_DEFAULT_FLAGS)


def _load_basic_flags(database_url: str, course_id: int) -> dict[str, Any]:
    """Pre-migration fallback: the four original on/off switches only."""
    try:
        with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT enabled, drowsiness, phone_usage, heatmap
                   FROM behaviour_config WHERE courseid = %s""",
                (course_id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "enabled": bool(row[0]), "drowsiness": bool(row[1]),
                    "phone_usage": bool(row[2]), "heatmap": bool(row[3]),
                }
    except Exception as exc:  # noqa: BLE001
        print(f"[behaviour] base config read failed: {exc}")
    return dict(_DEFAULT_FLAGS)


class InClassBehaviour:
    """Classroom behaviour analytics entity (U32/U33/U35).

    Per-course toggles are persisted in behaviour_config (U35).
    Event/heatmap queries read live from behaviour_event /
    heatmap_snapshot (U32/U33).
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        # Class diagram attribute: + behaviour (loose dict of recent events)
        self.behaviour: dict[str, Any] = {}
        # Class diagram attribute: + isAnalysisEnabled: bool
        # (course-scoped — set per request via _refresh_enabled_for)
        self.isAnalysisEnabled: bool = False

    def _refresh_enabled_for(self, course_id: int) -> dict[str, bool]:
        flags = load_behaviour_config(self.database_url, course_id)
        self.isAnalysisEnabled = flags["enabled"]
        return flags

    # ── U32 viewBehaviourReport ─────────────────────────────────────
    def viewBehaviourReport(self, session_id: int) -> dict[str, Any]:
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                # Every student who was in the room (so a student with no
                # events still appears, and can be told apart from one who
                # could never be analysed — see analysis_status below).
                cur.execute(
                    """SELECT r.accountid, pi.full_name,
                              COUNT(be.behavioureventid)
                                FILTER (WHERE be.event_type='drowsiness')        AS drowsy,
                              COUNT(be.behavioureventid)
                                FILTER (WHERE be.event_type='phone')             AS phone,
                              COALESCE(SUM(be.duration_seconds) FILTER
                                       (WHERE be.event_type='drowsiness'), 0)    AS drowsy_seconds,
                              COALESCE(SUM(be.duration_seconds) FILTER
                                       (WHERE be.event_type='phone'), 0)         AS phone_seconds
                       FROM attendance_record r
                       LEFT JOIN personal_info pi ON pi.accountid = r.accountid
                       LEFT JOIN behaviour_event be
                              ON be.attendancesessionid = r.attendancesessionid
                             AND be.accountid = r.accountid
                       WHERE r.attendancesessionid = %s
                         AND r.status IN ('present', 'late', 'early_left')
                       GROUP BY r.accountid, pi.full_name
                       ORDER BY pi.full_name NULLS LAST""",
                    (session_id,),
                )
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]

                # Analysis coverage (optional table — absent before the
                # migration runs, in which case no student is flagged).
                coverage: dict[int, tuple[int, int]] = {}
                try:
                    cur.execute(
                        """SELECT accountid, samples_total, samples_analysed
                           FROM behaviour_coverage WHERE attendancesessionid = %s""",
                        (session_id,),
                    )
                    coverage = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()

                # U32 alt-flow 2: a student the analyser never got usable
                # landmarks for is *inconclusive*, not well-behaved. Only
                # applied when the session actually produced behaviour data.
                session_analysed = bool(coverage) or any(
                    (r["drowsy"] or 0) + (r["phone"] or 0) > 0 for r in rows
                )
                for row in rows:
                    total, analysed = coverage.get(row["accountid"], (0, 0))
                    row["samples_total"] = total
                    row["samples_analysed"] = analysed
                    if not session_analysed:
                        row["analysis_status"] = "not_analysed"
                    elif analysed > 0 or (row["drowsy"] or 0) + (row["phone"] or 0) > 0:
                        row["analysis_status"] = "analysed"
                    else:
                        row["analysis_status"] = "inconclusive"
                # Timeline for the frontend chart: newest 500 raw events.
                cur.execute(
                    """SELECT be.behavioureventid, be.accountid, pi.full_name,
                              be.event_type, be.detected_at,
                              be.duration_seconds, be.confidence, be.metadata
                       FROM behaviour_event be
                       LEFT JOIN personal_info pi ON pi.accountid = be.accountid
                       WHERE be.attendancesessionid = %s
                       ORDER BY be.detected_at DESC
                       LIMIT 500""",
                    (session_id,),
                )
                cols = [c[0] for c in cur.description]
                timeline = [dict(zip(cols, r)) for r in cur.fetchall()]
            self.behaviour = {"events_by_student": rows}
            return {
                "success": True, "session_id": session_id,
                "students": rows, "timeline": timeline,
                "inconclusive_count": sum(
                    1 for r in rows if r.get("analysis_status") == "inconclusive"
                ),
            }
        except psycopg2.errors.UndefinedTable:
            return {"success": True, "session_id": session_id, "students": [],
                    "timeline": [],
                    "note": "behaviour_event table not yet created (run schema.sql)"}
        except Exception as exc:  # noqa: BLE001 — report view must not 500
            print(f"[behaviour] viewBehaviourReport failed: {exc}")
            return {"success": True, "session_id": session_id,
                    "students": [], "timeline": []}

    # ── U33 viewHeatmap ─────────────────────────────────────────────
    def viewHeatmap(
        self, session_id: int,
        time_from: str | None = None, time_to: str | None = None,
    ) -> dict[str, Any]:
        """Zone intensities for a session. `time_from` / `time_to` (ISO
        timestamps) narrow the view to one part of the lesson (U33 step 4),
        so the teacher can compare, e.g., the first and last 20 minutes."""
        clauses = ["attendancesessionid = %s"]
        params: list[Any] = [session_id]
        if time_from:
            clauses.append("captured_at >= %s")
            params.append(time_from)
        if time_to:
            clauses.append("captured_at <= %s")
            params.append(time_to)
        sql = f"""SELECT zone_x, zone_y, intensity, captured_at
                  FROM heatmap_snapshot
                  WHERE {" AND ".join(clauses)}
                  ORDER BY captured_at"""
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                # Full extent of the session's data, so the UI can seed its
                # time-range pickers even when a narrower window is applied.
                cur.execute(
                    """SELECT MIN(captured_at), MAX(captured_at)
                       FROM heatmap_snapshot WHERE attendancesessionid = %s""",
                    (session_id,),
                )
                first, last = cur.fetchone()
            return {
                "success": True, "session_id": session_id, "zones": rows,
                "captured_from": first, "captured_to": last,
            }
        except psycopg2.errors.UndefinedTable:
            return {"success": True, "session_id": session_id, "zones": [],
                    "captured_from": None, "captured_to": None,
                    "note": "heatmap_snapshot table not yet created (run schema.sql)"}
        except Exception as exc:  # noqa: BLE001 — heatmap view must not 500
            print(f"[behaviour] viewHeatmap failed: {exc}")
            return {"success": True, "session_id": session_id, "zones": [],
                    "captured_from": None, "captured_to": None}

    # ── U35 enable / disable (DB-backed) ────────────────────────────
    def enableBehaviourAnalysis(
        self, course_id: int,
        drowsiness: bool = True, phone_usage: bool = True, heatmap: bool = True,
        updated_by: int | None = None, tuning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._upsert_config(
            course_id,
            enabled=True,
            drowsiness=drowsiness,
            phone_usage=phone_usage,
            heatmap=heatmap,
            updated_by=updated_by,
            tuning=tuning,
        )

    def disableBehaviourAnalysis(
        self, course_id: int, updated_by: int | None = None,
    ) -> dict[str, Any]:
        return self._upsert_config(
            course_id, enabled=False, updated_by=updated_by,
        )

    # ── internal ────────────────────────────────────────────────────
    @staticmethod
    def _apply_tuning(cur, course_id: int, tuning: dict[str, Any] | None) -> None:
        """Write per-course detection thresholds (U35).

        Only keys actually supplied are touched, and an explicit ``None``
        clears the override so the course falls back to the server default —
        that distinction is what lets an administrator undo a bad setting
        without knowing what the global value is.
        """

        """  if not tuning:
        if not tuning:
            return
        columns = [k for k in (*_TUNING_KEYS, *_TUNING_BOOL_KEYS) if k in tuning]
        if not columns:
            return
        assignments = ", ".join(f"{c} = %s" for c in columns)
        params = [tuning[c] for c in columns]
        params.append(course_id)
        try:
            cur.execute(
                f"UPDATE behaviour_config SET {assignments}, updated_at = NOW() "
                "WHERE courseid = %s",
                params,
            )
        except (psycopg2.errors.UndefinedColumn, psycopg2.errors.UndefinedTable):
            # Pre-migration database: the on/off switches above still applied.
            raise HTTPException(
                503,
                "Detection thresholds need the latest schema — run "
                "database/schema.sql, then save again.",
            )"""
            )

    def _upsert_config(
        self, course_id: int, enabled: bool,
        drowsiness: bool | None = None, phone_usage: bool | None = None,
        heatmap: bool | None = None, updated_by: int | None = None,
        tuning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            with psycopg2.connect(self.database_url) as conn:
                try:
                    with conn.cursor() as cur:
                        # 1. Verify the course exists (FK protection).
                        cur.execute("SELECT 1 FROM course WHERE courseid = %s", (course_id,))
                        if not cur.fetchone():
                            raise HTTPException(404, "Course not found")

                        # 2. UPSERT. When disabling we only flip `enabled`;
                        #    when enabling we may also flip sub-feature flags.
                        if enabled:
                            cur.execute(
                                """INSERT INTO behaviour_config
                                      (courseid, enabled, drowsiness, phone_usage,
                                       heatmap, updated_by, updated_at)
                                   VALUES (%s, TRUE, %s, %s, %s, %s, NOW())
                                   ON CONFLICT (courseid) DO UPDATE
                                     SET enabled    = EXCLUDED.enabled,
                                         drowsiness = EXCLUDED.drowsiness,
                                         phone_usage = EXCLUDED.phone_usage,
                                         heatmap    = EXCLUDED.heatmap,
                                         updated_by = EXCLUDED.updated_by,
                                         updated_at = NOW()""",
                                (course_id, drowsiness, phone_usage, heatmap, updated_by),
                            )
                            self._apply_tuning(cur, course_id, tuning)
                        else:
                            cur.execute(
                                """INSERT INTO behaviour_config
                                      (courseid, enabled, updated_by, updated_at)
                                   VALUES (%s, FALSE, %s, NOW())
                                   ON CONFLICT (courseid) DO UPDATE
                                     SET enabled    = FALSE,
                                         updated_by = EXCLUDED.updated_by,
                                         updated_at = NOW()""",
                                (course_id, updated_by),
                            )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        except psycopg2.errors.UndefinedTable:
            raise HTTPException(
                503, "behaviour_config table does not exist; please run schema.sql first",
            )

        flags = self._refresh_enabled_for(course_id)
        return {"success": True, "course_id": course_id, "config": flags}


# ════════════════════════════════════════════════════════════════════════
# 2. Pure geometry helpers (unit-testable, no ML dependency)
# ════════════════════════════════════════════════════════════════════════
# MediaPipe FaceMesh landmark indices.
_EYE_RIGHT = [33, 160, 158, 133, 153, 144]     # p1..p6 (subject's right eye)
_EYE_LEFT = [362, 385, 387, 263, 373, 380]     # p1..p6 (subject's left eye)
_MOUTH_V = (13, 14)                            # inner-lip top / bottom
_MOUTH_H = (61, 291)                           # mouth corners
# 6-point 3D face model (mm) for solvePnP head pose, index-matched below.
_POSE_MODEL = np.array([
    (0.0, 0.0, 0.0),           # 1   nose tip
    (0.0, -330.0, -65.0),      # 152 chin
    (-225.0, 170.0, -135.0),   # 33  right-eye outer corner
    (225.0, 170.0, -135.0),    # 263 left-eye outer corner
    (-150.0, -150.0, -125.0),  # 61  mouth right corner
    (150.0, -150.0, -125.0),   # 291 mouth left corner
], dtype=np.float64)
_POSE_IDX = [1, 152, 33, 263, 61, 291]


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR over 6 eye landmarks p1..p6 (Soukupová & Čech 2016).
    Low EAR (< ~0.21) == eye closed."""
    p1, p2, p3, p4, p5, p6 = pts
    horiz = float(np.linalg.norm(p1 - p4))
    if horiz <= 1e-6:
        return 0.0
    return (float(np.linalg.norm(p2 - p6)) + float(np.linalg.norm(p3 - p5))) / (2.0 * horiz)


def mouth_aspect_ratio(top: np.ndarray, bottom: np.ndarray,
                       left: np.ndarray, right: np.ndarray) -> float:
    """MAR = inner-lip gap / mouth width. High MAR (> ~0.6) == yawn."""
    width = float(np.linalg.norm(left - right))
    if width <= 1e-6:
        return 0.0
    return float(np.linalg.norm(top - bottom)) / width


def head_pitch_deg(landmarks_px: dict[int, tuple[float, float]],
                   frame_w: int, frame_h: int) -> float | None:
    """Approximate head pitch (deg) via solvePnP on 6 landmarks.
    Returns a value normalised to [-90, 90]; large |pitch| == head tilted
    away from frontal (used as the head-down drowsiness signal)."""
    try:
        img_pts = np.array([landmarks_px[i] for i in _POSE_IDX], dtype=np.float64)
    except KeyError:
        return None
    focal = float(frame_w)
    cam = np.array([[focal, 0, frame_w / 2.0],
                    [0, focal, frame_h / 2.0],
                    [0, 0, 1]], dtype=np.float64)
    ok, rvec, _tvec = cv2.solvePnP(
        _POSE_MODEL, img_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rot)
    pitch = float(angles[0])
    # Model-vs-image axis conventions can shift pitch by ±180°; normalise.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch


# ════════════════════════════════════════════════════════════════════════
# 3. Detectors (lazy-loaded; degrade to disabled if package missing)
# ════════════════════════════════════════════════════════════════════════
# FaceLandmarker model asset for the MediaPipe Tasks API (newer mediapipe
# wheels dropped the legacy mp.solutions API). Same 468-landmark topology as
# FaceMesh, so the EAR/MAR/pose indices above apply to both.
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class DrowsinessDetector:
    """MediaPipe face landmarks over a single face crop → {ear, mar, pitch}.

    Supports both mediapipe APIs: legacy ``mp.solutions.face_mesh`` (older
    wheels) and the Tasks-API ``FaceLandmarker`` (0.10.30+, where the legacy
    module was removed; the .task model downloads once on first use)."""

    def __init__(self):
        self._mp = None
        self._legacy = None   # mp.solutions FaceMesh instance
        self._tasks = None    # mp Tasks FaceLandmarker instance
        self._failed = False

    def _ensure(self) -> bool:
        if self._legacy is not None or self._tasks is not None:
            return True
        if self._failed:
            return False
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            if hasattr(mp, "solutions"):  # legacy API
                self._legacy = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.4,
                )
                return True
            # Tasks API — needs the FaceLandmarker model asset on disk.
            from mediapipe.tasks.python import vision  # type: ignore
            from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore

            model_path = os.getenv("AI_FACEMESH_MODEL", "models/face_landmarker.task")
            if not os.path.isfile(model_path):
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                print(f"[behaviour] downloading FaceLandmarker model → {model_path}")
                urllib.request.urlretrieve(_FACE_LANDMARKER_URL, model_path)
            self._tasks = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.4,
                )
            )
            return True
        except Exception as exc:  # noqa: BLE001 — missing wheel, download, GPU init, …
            print(f"[behaviour] drowsiness disabled (mediapipe unavailable): {exc}")
            self._failed = True
            return False

    def _landmarks(self, rgb: np.ndarray):
        """Return the 468-landmark list (objects with .x/.y) or None."""
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            if not res.multi_face_landmarks:
                return None
            return res.multi_face_landmarks[0].landmark
        img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        res = self._tasks.detect(img)
        if not res.face_landmarks:
            return None
        return res.face_landmarks[0]

    def observe(self, face_bgr: np.ndarray) -> dict[str, Any] | None:
        """Return {"ear", "mar", "pitch"} for the crop, or None if no face."""
        if not self._ensure():
            return None
        h, w = face_bgr.shape[:2]
        if h < 40 or w < 40:
            return None
        lm = self._landmarks(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        if lm is None:
            return None
        wanted = set(_EYE_LEFT + _EYE_RIGHT + list(_MOUTH_V) + list(_MOUTH_H) + _POSE_IDX)
        px = {i: (lm[i].x * w, lm[i].y * h) for i in wanted}
        arr = {i: np.array(v) for i, v in px.items()}
        ear = (eye_aspect_ratio(np.stack([arr[i] for i in _EYE_LEFT]))
               + eye_aspect_ratio(np.stack([arr[i] for i in _EYE_RIGHT]))) / 2.0
        mar = mouth_aspect_ratio(arr[_MOUTH_V[0]], arr[_MOUTH_V[1]],
                                 arr[_MOUTH_H[0]], arr[_MOUTH_H[1]])
        pitch = head_pitch_deg(px, w, h)
        return {"ear": round(ear, 3), "mar": round(mar, 3),
                "pitch": round(pitch, 1) if pitch is not None else None}


class PhoneDetector:
    """Ultralytics YOLO on the full frame; COCO class 67 = cell phone.

    model_name / imgsz come from AIConfig (AI_PHONE_MODEL / AI_PHONE_IMGSZ):
    inference at imgsz=1280 keeps a 720p frame at native scale so distant
    (>3 m) phones stay detectable; yolov8s+ further improves small-object
    recall on GPU rigs."""

    _COCO_CELL_PHONE = 67

    def __init__(self, conf_threshold: float,
                 model_name: str = "yolov8n.pt", imgsz: int = 1280,
                 want_gpu: bool = False):
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.imgsz = imgsz
        self.want_gpu = want_gpu
        self._model = None
        self._failed = False

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_name)  # weights download once on first use
            device = getattr(self._model, "device", "?")
            print(f"[behaviour] phone detector: {self.model_name} "
                  f"@ imgsz={self.imgsz} on {device}")
            # A CPU-only torch build runs happily and silently at roughly ten
            # times the latency, which at 1 fps shows up as dropped frames
            # rather than an error. Say so rather than letting it look normal.
            if self.want_gpu and "cuda" not in str(device).lower():
                print("[behaviour] WARNING: GPU requested but YOLO is on CPU. "
                      "The installed torch is a CPU build — reinstall with "
                      "--index-url https://download.pytorch.org/whl/cu126")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[behaviour] phone detection disabled (ultralytics unavailable): {exc}")
            self._failed = True
            return False

    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return [(bbox[x1,y1,x2,y2], confidence)] for phones in the frame."""
        if not self._ensure():
            return []
        results = self._model.predict(
            frame, verbose=False, conf=self.conf_threshold,
            classes=[self._COCO_CELL_PHONE], imgsz=self.imgsz,
        )
        out: list[tuple[np.ndarray, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy()):
                out.append((box.astype(int), float(conf)))
        return out


class HeatmapAccumulator:
    """Accumulate face-centre hits on a cols×rows grid; flush → normalised
    (zone_x, zone_y, intensity) rows for heatmap_snapshot (U33)."""

    def __init__(self, cols: int, rows: int):
        self.cols = max(1, cols)
        self.rows = max(1, rows)
        self._grid = np.zeros((self.rows, self.cols), dtype=np.float64)

    def add_point(self, cx: float, cy: float, frame_w: int, frame_h: int) -> None:
        if frame_w <= 0 or frame_h <= 0:
            return
        x = min(self.cols - 1, max(0, int(cx / frame_w * self.cols)))
        y = min(self.rows - 1, max(0, int(cy / frame_h * self.rows)))
        self._grid[y, x] += 1.0

    def snapshot_and_reset(self) -> list[tuple[int, int, float]]:
        peak = float(self._grid.max())
        cells: list[tuple[int, int, float]] = []
        if peak > 0:
            for y in range(self.rows):
                for x in range(self.cols):
                    v = float(self._grid[y, x])
                    if v > 0:
                        cells.append((x, y, round(v / peak, 4)))
        self._grid.fill(0.0)
        return cells


# ════════════════════════════════════════════════════════════════════════
# 4. Temporal state (episode debouncing)
# ════════════════════════════════════════════════════════════════════════
class _EpisodeTracker:
    """Debounce a boolean per-sample signal into episodes.

    * An episode must stay active >= confirm_seconds to count at all.
    * Signal dropouts shorter than release_seconds do not end the episode
      (YOLO / FaceMesh flicker tolerance at ~1 fps).
    * Episodes longer than chunk_seconds are emitted in chunks so a long
      episode is still recorded even if scanning stops mid-way.
    """

    def __init__(self, confirm_seconds: float,
                 release_seconds: float = 2.5, chunk_seconds: float = 60.0):
        self.confirm = confirm_seconds
        self.release = release_seconds
        self.chunk = chunk_seconds
        self.start: float | None = None
        self.last_true: float | None = None
        self.meta: dict[str, Any] = {}

    def _merge_meta(self, meta: dict[str, Any] | None) -> None:
        if not meta:
            return
        reasons = set(self.meta.get("reasons", [])) | set(meta.get("reasons", []))
        self.meta.update(meta)
        if reasons:
            self.meta["reasons"] = sorted(reasons)

    def _emit(self, end: float) -> dict[str, Any]:
        started = end if self.start is None else self.start
        return {
            "started": started,
            "duration": max(1, int(round(end - started))),
            "meta": dict(self.meta),
        }

    def is_confirmed(self, now: float) -> bool:
        return self.start is not None and (now - self.start) >= self.confirm

    def update(self, active: bool, now: float,
               meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Feed one sample; returns a finished episode dict or None."""
        if active:
            if self.start is None:
                self.start = now
                self.meta = {}
            self._merge_meta(meta)
            self.last_true = now
            if now - self.start >= self.chunk:
                episode = self._emit(now)
                self.start = now  # continue counting the next chunk
                return episode
            return None
        if self.start is None or self.last_true is None:
            return None
        if now - self.last_true <= self.release:
            return None  # short dropout — keep the episode open
        episode = None
        if self.last_true - self.start >= self.confirm:
            episode = self._emit(self.last_true)
        self.start = None
        self.last_true = None
        self.meta = {}
        return episode


class _EyeClosureModel:
    """Decides "eyes closed" per student, and aggregates it with PERCLOS.

    Two problems with comparing a raw eye-aspect ratio against one constant:

    * Eye aperture varies far more between people than between one person's
      alert and drowsy states. A literature constant (0.21, from datasets
      that are not representative of this cohort) marks some students as
      permanently asleep and never trips for others.
    * A single low sample is usually a blink, not sleep.

    So the first `baseline_samples` observations establish that student's own
    open-eye median (people are alert at the start of a lesson), closure is
    then judged relative to it, and the verdict is PERCLOS — the share of
    closed samples inside a rolling time window — rather than one frame.
    """

    def __init__(
        self, fixed_threshold: float, adaptive: bool, baseline_samples: int,
        baseline_ratio: float, window_seconds: float, perclos_threshold: float,
    ):
        self.fixed_threshold = fixed_threshold
        self.adaptive = adaptive
        self.baseline_samples = max(1, baseline_samples)
        self.baseline_ratio = baseline_ratio
        self.window_seconds = window_seconds
        self.perclos_threshold = perclos_threshold
        self._calibration: list[float] = []
        self.baseline: float | None = None
        # (timestamp, closed) inside the rolling window.
        self._window: list[tuple[float, bool]] = []

    @property
    def threshold(self) -> float:
        """The EAR value currently treated as 'closed' for this student."""
        if self.adaptive and self.baseline is not None:
            return self.baseline * self.baseline_ratio
        return self.fixed_threshold

    def observe(self, ear: float, now: float) -> bool:
        """Feed one sample; returns whether PERCLOS says the eyes are closed."""
        if self.adaptive and self.baseline is None:
            self._calibration.append(ear)
            if len(self._calibration) >= self.baseline_samples:
                # Median, so a few blinks during calibration cannot drag the
                # reference down and desensitise the student permanently.
                ordered = sorted(self._calibration)
                mid = len(ordered) // 2
                self.baseline = (
                    ordered[mid] if len(ordered) % 2
                    else (ordered[mid - 1] + ordered[mid]) / 2.0
                )
            # Judging with the fixed threshold while still learning would
            # re-introduce exactly the bias adaptation exists to remove: a
            # student whose open eyes sit below the constant would have every
            # calibration sample recorded as closed, poisoning the window
            # before their real baseline is even known. Observe, don't judge.
            return False

        closed = ear < self.threshold
        self._window.append((now, closed))
        cutoff = now - self.window_seconds
        self._window = [(t, c) for t, c in self._window if t >= cutoff]
        return self.perclos() >= self.perclos_threshold

    def perclos(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for _, closed in self._window if closed) / len(self._window)

    def calibrating(self) -> bool:
        return self.adaptive and self.baseline is None


class _HeadPoseModel:
    """Decides "head down" per student, relative to their own upright pose.

    An absolute pitch cutoff does not survive contact with real classrooms.
    The 6-point solvePnP model compresses the angle badly — a head lowered
    about 40 degrees is estimated at 12-20 — and the upright reading itself
    depends on where the camera is mounted and how the student sits, which
    measured between 6 and 8 degrees here. A single global cutoff therefore
    either never fires (the shipped 30) or fires constantly for anyone whose
    natural posture is slightly forward.

    So the first `baseline_samples` readings establish that student's own
    upright pitch, and head-down is judged as a departure from it.

    Readings are DIRECTIONAL: callers pass `down_deg`, increasing as the head
    tilts down. Comparing on magnitude instead would make a student who sits
    upright-and-slightly-back look identical to one slumped forward — which
    is measurably wrong, not merely imprecise: a segment recorded sitting
    back read +8 to +11 degrees and was flagged as head-down by an
    abs()-based test.
    """

    def __init__(self, fixed_threshold: float, adaptive: bool,
                 baseline_samples: int, delta_deg: float):
        self.fixed_threshold = fixed_threshold
        self.adaptive = adaptive
        self.baseline_samples = max(1, baseline_samples)
        self.delta_deg = delta_deg
        self._calibration: list[float] = []
        self.baseline: float | None = None

    def observe(self, down_deg: float) -> bool:
        """Feed one directional reading; returns whether the head is down."""
        if not self.adaptive:
            return down_deg > self.fixed_threshold
        if self.baseline is None:
            self._calibration.append(down_deg)
            if len(self._calibration) >= self.baseline_samples:
                ordered = sorted(self._calibration)
                mid = len(ordered) // 2
                self.baseline = (
                    ordered[mid] if len(ordered) % 2
                    else (ordered[mid - 1] + ordered[mid]) / 2.0
                )
            # Withhold a verdict while learning, for the same reason the eye
            # model does: judging against the global cutoff during
            # calibration reintroduces the bias adaptation exists to remove.
            return False
        return down_deg - self.baseline > self.delta_deg

    @property
    def threshold(self) -> float:
        """The |pitch| currently treated as head-down for this student."""
        if self.adaptive and self.baseline is not None:
            return self.baseline + self.delta_deg
        return self.fixed_threshold

    def calibrating(self) -> bool:
        return self.adaptive and self.baseline is None


class _StaticBoxFilter:
    """Suppress phone detections that never move.

    Wall fixtures — light switches, thermostats, sockets, projector panels —
    are small bright rectangles, which is close enough to a phone that YOLO
    scores them well above any usable confidence floor. Because they sit in
    the same pixels all lesson, they fall inside some student's reach region
    every single sample, and that student is blamed for the whole class.

    Temporal debouncing makes this worse rather than better: a detection that
    is present in every frame is precisely what `_EpisodeTracker` promotes
    into a confirmed episode. The distinguishing property is not confidence
    or duration but immobility, so that is what is tested here — a box whose
    position is essentially unchanged over a long run of samples is a fixture.

    A phone held still for a while is not misclassified: `persist_samples`
    defaults to ~45 s of continuous, near-pixel-identical position, and any
    real hand movement resets the track.
    """

    def __init__(self, persist_samples: int, iou_threshold: float):
        self.persist_samples = persist_samples
        self.iou_threshold = iou_threshold
        # (box, consecutive_hits) per candidate fixture.
        self._tracks: list[list[Any]] = []

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / float(area_a + area_b - inter)

    def update(self, boxes: list) -> None:
        """Feed every phone box seen in this sample."""
        matched = set()
        for track in self._tracks:
            hit = None
            for i, box in enumerate(boxes):
                if i in matched:
                    continue
                if self._iou(track[0], box) >= self.iou_threshold:
                    hit = i
                    break
            if hit is None:
                track[1] = 0  # moved or vanished — not a fixture
            else:
                matched.add(hit)
                track[1] += 1
                track[0] = boxes[hit]
        for i, box in enumerate(boxes):
            if i not in matched:
                self._tracks.append([box, 1])
        # Bound memory: only plausible fixtures are worth remembering.
        self._tracks = [t for t in self._tracks if t[1] > 0][-32:]

    def is_static(self, box) -> bool:
        for track_box, hits in self._tracks:
            if (hits >= self.persist_samples
                    and self._iou(track_box, box) >= self.iou_threshold):
                return True
        return False


@dataclass
class _StudentState:
    drowsy: _EpisodeTracker
    phone: _EpisodeTracker
    eyes: "_EyeClosureModel | None" = None
    head: "_HeadPoseModel | None" = None


@dataclass
class _SessionState:
    students: dict[int, _StudentState] = field(default_factory=dict)
    heatmap: HeatmapAccumulator | None = None
    # Fixture suppression is per-room, not per-student: a wall panel is the
    # same object no matter whose reach region it happens to fall in.
    static_phones: "_StaticBoxFilter | None" = None
    last_heatmap_flush: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # U32 coverage: account_id -> [samples_total, samples_analysed] since the
    # last flush. Lets the report tell "nothing detected" apart from
    # "never analysable" (insufficient camera coverage).
    coverage: dict[int, list[int]] = field(default_factory=dict)
    last_coverage_flush: float = field(default_factory=time.time)


# ════════════════════════════════════════════════════════════════════════
# 5. BehaviourAnalysisService (CR-06 per-frame entry point)
# ════════════════════════════════════════════════════════════════════════
_SESSION_STALE_SECONDS = 600  # drop per-session state after 10 min idle


def _tuned(flags: dict[str, Any], key: str, fallback: float) -> float:
    """Per-course override if the admin set one, else the server default."""
    value = (flags or {}).get(key)
    return fallback if value is None else float(value)


class BehaviourAnalysisService:
    """Per-frame behaviour analysis: drowsiness + phone + heatmap.

    Held once on app.state.behaviour (enabled via AI_BEHAVIOUR=true).
    Models load lazily on the first analysed frame so startup stays fast
    and CPU-only deployments that never enable the feature pay nothing.
    """

    def __init__(self, cfg: "AIConfig", database_url: str):
        self.cfg = cfg
        self.database_url = database_url
        self._drowsy = DrowsinessDetector()
        self._phone = PhoneDetector(cfg.phone_conf, cfg.phone_model,
                                    cfg.phone_imgsz, want_gpu=cfg.ctx_id >= 0)
        self._sessions: dict[int, _SessionState] = {}
        # Frame-drop guard: if a frame is still being processed when the
        # next arrives (CPU too slow for 1 fps), the new one is skipped.
        self._busy = threading.Lock()

    # ── public entry point ───────────────────────────────────────────
    def process_frame(
        self, session_id: int, frame: np.ndarray,
        pipeline: "AttendancePipeline", flags: dict[str, bool],
    ) -> dict[str, Any]:
        if not self._busy.acquire(blocking=False):
            return {"skipped": True, "reason": "previous frame still processing"}
        try:
            return self._process(session_id, frame, pipeline, flags)
        finally:
            self._busy.release()

    # ── internals ────────────────────────────────────────────────────
    def _state(self, session_id: int, now: float) -> _SessionState:
        # Evict stale sessions so memory stays bounded across a school day.
        for sid in [s for s, st in self._sessions.items()
                    if now - st.last_seen > _SESSION_STALE_SECONDS]:
            del self._sessions[sid]
        state = self._sessions.get(session_id)
        if state is None:
            cols, rows = self.cfg.heatmap_grid
            state = _SessionState(
                heatmap=HeatmapAccumulator(cols, rows),
                static_phones=_StaticBoxFilter(
                    self.cfg.phone_static_samples, self.cfg.phone_static_iou,
                ),
            )
            self._sessions[session_id] = state
        state.last_seen = now
        return state

    def _student(
        self, state: _SessionState, account_id: int,
        flags: dict[str, Any] | None = None,
    ) -> _StudentState:
        st = state.students.get(account_id)
        if st is None:
            flags = flags or {}
            confirm = _tuned(
                flags, "drowsy_confirm_seconds", self.cfg.ear_consec_seconds
            )
            st = _StudentState(
                drowsy=_EpisodeTracker(confirm),
                # ~1 fps sampling: N consecutive samples ≈ N seconds.
                phone=_EpisodeTracker(float(self.cfg.phone_consec_samples)),
                eyes=_EyeClosureModel(
                    fixed_threshold=_tuned(
                        flags, "ear_threshold", self.cfg.ear_threshold
                    ),
                    adaptive=bool(flags.get("adaptive_ear", self.cfg.adaptive_ear)),
                    baseline_samples=self.cfg.ear_baseline_samples,
                    baseline_ratio=self.cfg.ear_baseline_ratio,
                    window_seconds=self.cfg.perclos_window_seconds,
                    perclos_threshold=self.cfg.perclos_threshold,
                ),
                head=_HeadPoseModel(
                    fixed_threshold=_tuned(
                        flags, "headpose_pitch_deg", self.cfg.headpose_pitch_deg
                    ),
                    adaptive=bool(
                        flags.get("adaptive_headpose", self.cfg.adaptive_headpose)
                    ),
                    baseline_samples=self.cfg.headpose_baseline_samples,
                    delta_deg=_tuned(
                        flags, "headpose_delta_deg", self.cfg.headpose_delta_deg
                    ),
                ),
            )
            state.students[account_id] = st
        return st

    @staticmethod
    def _face_crop(frame: np.ndarray, bbox: list[int], pad: float = 0.25) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - pad * bw))
        cy1 = max(0, int(y1 - pad * bh))
        cx2 = min(w, int(x2 + pad * bw))
        cy2 = min(h, int(y2 + pad * bh))
        return frame[cy1:cy2, cx1:cx2]

    @staticmethod
    def _phone_owner(
        phone_box: np.ndarray, recognised: list["Prediction"],
    ) -> int | None:
        """Attribute a phone box to the nearest recognised face whose
        'reach region' contains the phone centre.

        The region starts at eye level and extends downwards, because a phone
        being looked at is held below the face — never above it. Anything
        higher (a screen behind the class, a poster, a reflection) belongs to
        nobody and is dropped rather than blamed on the nearest student.
        """
        pcx = (phone_box[0] + phone_box[2]) / 2.0
        pcy = (phone_box[1] + phone_box[3]) / 2.0
        best: tuple[float, int | None] = (float("inf"), None)
        for p in recognised:
            x1, y1, x2, y2 = p.bbox
            fw = max(1, x2 - x1)
            fh = max(1, y2 - y1)
            rx1, rx2 = x1 - fw, x2 + fw
            # From mid-face downwards only.
            ry1, ry2 = y1 + fh * 0.4, y2 + 2 * fw
            if not (rx1 <= pcx <= rx2 and ry1 <= pcy <= ry2):
                continue
            fcx, fcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = (pcx - fcx) ** 2 + (pcy - fcy) ** 2
            if d < best[0]:
                best = (d, p.account_id)
        return best[1]

    def _process(
        self, session_id: int, frame: np.ndarray,
        pipeline: "AttendancePipeline", flags: dict[str, bool],
    ) -> dict[str, Any]:
        now = time.time()
        state = self._state(session_id, now)
        result = pipeline.process_frame(frame)
        recognised = [p for p in result.predictions
                      if p.recognised and p.account_id is not None]
        frame_h, frame_w = frame.shape[:2]

        events: list[tuple[int, str, float, int, float | None, dict]] = []
        drowsy_active: list[int] = []
        phone_active: list[int] = []

        # ── heatmap: every detected face (recognised or not) counts ──
        if flags.get("heatmap") and state.heatmap is not None:
            for p in result.predictions:
                x1, y1, x2, y2 = p.bbox
                state.heatmap.add_point((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                                        frame_w, frame_h)

        # ── phone usage: YOLO on the full frame, attributed by bbox ──
        # Resolved before drowsiness because a student looking down at a phone
        # is looking down for a known reason; see the head_pose gate below.
        holders: dict[int, float] = {}
        if flags.get("phone_usage"):
            min_conf = _tuned(flags, "phone_conf", self.cfg.phone_conf)
            detections = [
                (box, conf) for box, conf in self._phone.detect(frame)
                # A per-course override may be stricter than YOLO's own floor.
                if conf >= min_conf
            ]
            # Track every surviving box so immobile ones can be recognised as
            # room fixtures, then drop those before anybody is blamed.
            if state.static_phones is not None:
                state.static_phones.update([b for b, _ in detections])
                detections = [
                    (b, c) for b, c in detections
                    if not state.static_phones.is_static(b)
                ]
            for box, conf in detections:
                owner = self._phone_owner(box, recognised)
                if owner is not None:
                    holders[owner] = max(holders.get(owner, 0.0), conf)

        # ── drowsiness: per recognised face (identity attribution) ───
        if flags.get("drowsiness"):
            for p in recognised:
                st = self._student(state, p.account_id, flags)
                # U32 coverage: this sample recognised the student; it only
                # counts as *analysed* when landmarks were actually usable.
                cov = state.coverage.setdefault(p.account_id, [0, 0])
                cov[0] += 1

                crop = self._face_crop(frame, p.bbox)
                # Landmarks on a tiny face are noise. Leaving the sample
                # unanalysed feeds the "inconclusive" report (U32) instead of
                # inventing a verdict from unreliable geometry.
                too_small = min(crop.shape[:2]) < self.cfg.behaviour_min_face_px
                obs = None if too_small else self._drowsy.observe(crop)

                # Head pose is taken from the detector that already found this
                # face, and is therefore still available on exactly the frames
                # the landmarker gives up on — which are the head-down frames
                # this signal exists to catch.
                #
                # The two sources disagree on sign: the detector reports pitch
                # negative-downward, while the landmarker's normalised
                # solvePnP estimate grows with any departure from frontal.
                # Both are converted to "degrees downward" so _HeadPoseModel
                # sees one convention.
                if p.head_pitch is not None:
                    down_deg = -p.head_pitch
                elif obs is not None and obs["pitch"] is not None:
                    down_deg = abs(obs["pitch"])
                else:
                    down_deg = None
                if obs is not None or down_deg is not None:
                    cov[1] += 1

                reasons = []
                head_down = (down_deg is not None and st.head is not None
                             and st.head.observe(down_deg))
                # A student bent over a phone is bent over for a reason that
                # is already being reported as its own behaviour. Counting it
                # as drowsiness too would double-report one posture as two
                # findings and, worse, put "drowsy" next to the name of
                # someone who is wide awake and texting. Measured: this alone
                # accounts for 40% of head-down samples on the phone segment.
                if head_down and p.account_id not in holders:
                    reasons.append("head_pose")
                if obs is not None:
                    # PERCLOS over this student's own baseline, not a frame
                    # against a global constant.
                    if st.eyes is not None and st.eyes.observe(obs["ear"], now):
                        reasons.append("eyes_closed")
                    if obs["mar"] > _tuned(flags, "mar_threshold", self.cfg.mar_threshold):
                        reasons.append("yawn")
                if obs is not None and st.eyes is not None:
                    obs = {
                        **obs,
                        "ear_threshold": round(st.eyes.threshold, 3),
                        "perclos": round(st.eyes.perclos(), 2),
                        "calibrating": st.eyes.calibrating(),
                    }
                    if st.head is not None:
                        obs["pitch_threshold"] = round(st.head.threshold, 1)
                elif down_deg is not None:
                    # Landmarks unavailable but pose was: record what the
                    # verdict was actually based on, so a head_pose event is
                    # never mistaken for a full facial-landmark analysis.
                    obs = {"down_deg": round(down_deg, 1),
                           "source": "detector_pose"}
                meta = {**obs, "reasons": reasons} if reasons else None
                episode = st.drowsy.update(bool(reasons), now, meta)
                if episode is not None:
                    events.append((p.account_id, "drowsiness", episode["started"],
                                   episode["duration"], None, episode["meta"]))
                if st.drowsy.is_confirmed(now):
                    drowsy_active.append(p.account_id)

        # ── phone episodes (detection itself ran before drowsiness) ──
        if flags.get("phone_usage"):
            for p in recognised:
                st = self._student(state, p.account_id, flags)
                conf = holders.get(p.account_id)
                meta = {"conf": round(conf, 3)} if conf is not None else None
                episode = st.phone.update(conf is not None, now, meta)
                if episode is not None:
                    events.append((p.account_id, "phone", episode["started"],
                                   episode["duration"],
                                   episode["meta"].get("conf"), episode["meta"]))
                if st.phone.is_confirmed(now):
                    phone_active.append(p.account_id)

        # ── persist derived tuples only (never frames) ────────────────
        written = self._write_events(session_id, events)
        heatmap_cells = 0
        if (flags.get("heatmap") and state.heatmap is not None
                and now - state.last_heatmap_flush >= self.cfg.heatmap_flush_seconds):
            heatmap_cells = self._write_heatmap(
                session_id, state.heatmap.snapshot_and_reset()
            )
            state.last_heatmap_flush = now
        if (state.coverage
                and now - state.last_coverage_flush >= self.cfg.heatmap_flush_seconds):
            self._write_coverage(session_id, state.coverage)
            state.coverage = {}
            state.last_coverage_flush = now

        return {
            "faces_in_frame": len(result.predictions),
            "recognised": len(recognised),
            "drowsy_active": drowsy_active,
            "phone_active": phone_active,
            "events_written": written,
            "heatmap_cells_written": heatmap_cells,
        }

    def _write_events(
        self, session_id: int,
        events: list[tuple[int, str, float, int, float | None, dict]],
    ) -> int:
        if not events:
            return 0
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                for account_id, etype, started, duration, conf, meta in events:
                    cur.execute(
                        """INSERT INTO behaviour_event
                              (attendancesessionid, accountid, event_type,
                               detected_at, duration_seconds, confidence, metadata)
                           VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s)""",
                        (session_id, account_id, etype, started,
                         duration, conf, Json(meta)),
                    )
            return len(events)
        except Exception as exc:  # noqa: BLE001 — a DB blip must not kill the scan loop
            print(f"[behaviour] event write failed ({len(events)} events): {exc}")
            return 0

    def _write_coverage(
        self, session_id: int, coverage: dict[int, list[int]],
    ) -> int:
        """Accumulate per-student analysis coverage (U32 'inconclusive')."""
        if not coverage:
            return 0
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                for account_id, (total, analysed) in coverage.items():
                    cur.execute(
                        """INSERT INTO behaviour_coverage
                              (attendancesessionid, accountid,
                               samples_total, samples_analysed)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (attendancesessionid, accountid) DO UPDATE
                             SET samples_total =
                                   behaviour_coverage.samples_total + EXCLUDED.samples_total,
                                 samples_analysed =
                                   behaviour_coverage.samples_analysed + EXCLUDED.samples_analysed,
                                 updated_at = NOW()""",
                        (session_id, account_id, total, analysed),
                    )
            return len(coverage)
        except Exception as exc:  # noqa: BLE001 — coverage is diagnostic only
            print(f"[behaviour] coverage write failed: {exc}")
            return 0

    def _write_heatmap(
        self, session_id: int, cells: list[tuple[int, int, float]],
    ) -> int:
        if not cells:
            return 0
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                for zone_x, zone_y, intensity in cells:
                    cur.execute(
                        """INSERT INTO heatmap_snapshot
                              (attendancesessionid, zone_x, zone_y, intensity)
                           VALUES (%s, %s, %s, %s)""",
                        (session_id, zone_x, zone_y, intensity),
                    )
            return len(cells)
        except Exception as exc:  # noqa: BLE001
            print(f"[behaviour] heatmap write failed ({len(cells)} cells): {exc}")
            return 0


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["behaviourAnalysis"])


def _svc(request: Request) -> InClassBehaviour:
    return InClassBehaviour(request.app.state.cfg.database_url)


async def _bytes_to_cv2(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unable to parse image file")
    return img


# Teacher: U32 / U33
@router.get("/teacher/sessions/{session_id}/behaviour")
def teacher_view_behaviour(
    session_id: int, request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewBehaviourReport(session_id)


@router.get("/teacher/sessions/{session_id}/heatmap")
def teacher_view_heatmap(
    session_id: int, request: Request,
    time_from: str | None = None, time_to: str | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewHeatmap(session_id, time_from, time_to)


# Teacher: CR-06 behaviour frame ingest (~1 fps from the scan page).
# Frames are analysed in memory and discarded; only derived event tuples
# and heatmap grid intensities are stored (PDPC).
@router.post("/teacher/sessions/{session_id}/behaviour-scan")
async def teacher_behaviour_scan(
    session_id: int,
    request: Request,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_role("teacher")),
):
    database_url = request.app.state.cfg.database_url
    with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT courseid, status FROM attendance_session WHERE attendancesessionid = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    course_id, status = row
    if status != "active":
        raise HTTPException(409, "Behaviour analysis only runs while the session is active")

    # U35 gate: the per-course toggle decides whether anything runs at all.
    flags = load_behaviour_config(database_url, course_id)
    if not flags["enabled"]:
        return {"success": True, "enabled": False,
                "message": "Behaviour analysis is disabled for this course (U35)"}

    svc: BehaviourAnalysisService | None = getattr(request.app.state, "behaviour", None)
    if svc is None:
        raise HTTPException(
            503,
            "Behaviour analysis service is not running on this server "
            "(set AI_BEHAVIOUR=true — GPU deployment recommended)",
        )

    img = await _bytes_to_cv2(file)
    out = svc.process_frame(session_id, img, request.app.state.pipeline, flags)
    return {"success": True, "enabled": True, "session_id": session_id, **out}


# Admin: U35
class BehaviourToggleBody(BaseModel):
    enable: bool
    drowsiness: bool = True
    phone_usage: bool = True
    heatmap: bool = True
    # Detection tuning. Omit a field to leave it as-is; send null to clear the
    # override and inherit the server default. Ranges are enforced by CHECK
    # constraints on behaviour_config.
    ear_threshold: float | None = None
    mar_threshold: float | None = None
    headpose_pitch_deg: float | None = None
    headpose_delta_deg: float | None = None
    phone_conf: float | None = None
    drowsy_confirm_seconds: float | None = None
    adaptive_ear: bool | None = None
    adaptive_headpose: bool | None = None


@router.patch("/admin/courses/{course_id}/behaviour-analysis")
def admin_toggle_behaviour(
    course_id: int, body: BehaviourToggleBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    svc = _svc(request)
    if body.enable:
        # Only fields present in the request body are written, so a client
        # that knows nothing about tuning cannot silently reset it.
        supplied = body.model_dump(exclude_unset=True)
        tuning = {
            k: supplied[k]
            for k in (*_TUNING_KEYS, *_TUNING_BOOL_KEYS)
            if k in supplied
        }
        return svc.enableBehaviourAnalysis(
            course_id, body.drowsiness, body.phone_usage, body.heatmap,
            updated_by=user.account_id, tuning=tuning,
        )
    return svc.disableBehaviourAnalysis(course_id, updated_by=user.account_id)


@router.get("/admin/courses/{course_id}/behaviour-analysis")
def admin_get_behaviour(
    course_id: int, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    flags = load_behaviour_config(request.app.state.cfg.database_url, course_id)
    return {"success": True, "course_id": course_id, "config": flags}
