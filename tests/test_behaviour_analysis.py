"""Unit tests for core/behaviourAnalysis.py (CR-06 / U32–U35).

Everything here is ML-free: the geometry helpers, episode debouncer,
heatmap accumulator and phone-owner attribution are pure functions, so
they run on any CI runner without mediapipe / ultralytics / a GPU.

The DB round-trip test runs only when DATABASE_URL is set (as it is in
the CI backend job, against the throwaway pgvector Postgres service) and
is skipped locally so it can never touch a production database.
"""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import behaviourAnalysis as ba
from core.attendancePipeline import AIConfig
from core.behaviourAnalysis import (
    BehaviourAnalysisService,
    HeatmapAccumulator,
    _EpisodeTracker,
    _EyeClosureModel,
    eye_aspect_ratio,
    head_pitch_deg,
    mouth_aspect_ratio,
)


def _eye_model(**overrides):
    params = {
        "fixed_threshold": 0.21, "adaptive": True, "baseline_samples": 5,
        "baseline_ratio": 0.75, "window_seconds": 60.0, "perclos_threshold": 0.4,
    }
    params.update(overrides)
    return _EyeClosureModel(**params)


# ── geometry helpers ─────────────────────────────────────────────────
def test_ear_open_higher_than_closed():
    open_eye = np.array([[0, 0], [2, -1], [4, -1], [6, 0], [4, 1], [2, 1]], float)
    closed = np.array([[0, 0], [2, -0.1], [4, -0.1], [6, 0], [4, 0.1], [2, 0.1]], float)
    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed) > 0


def test_ear_degenerate_zero_width():
    pts = np.zeros((6, 2), float)
    assert eye_aspect_ratio(pts) == 0.0


def test_mar_ratio():
    mar = mouth_aspect_ratio(
        np.array([0, 0.0]), np.array([0, 3.0]),
        np.array([-2, 1.5]), np.array([2, 1.5]),
    )
    assert mar == pytest.approx(0.75)


def test_head_pitch_missing_landmarks_returns_none():
    assert head_pitch_deg({1: (0, 0)}, 640, 480) is None


def test_head_pitch_frontal_face_near_zero():
    # Project the reference 3D model with zero rotation → solvePnP must
    # recover a (near-)frontal pitch.
    w, h = 640, 480
    focal = float(w)
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], float)
    rvec = np.zeros(3)
    tvec = np.array([0.0, 0.0, 1000.0])
    projected, _ = cv2.projectPoints(ba._POSE_MODEL, rvec, tvec, cam, None)
    landmarks = {
        idx: tuple(projected[i][0]) for i, idx in enumerate(ba._POSE_IDX)
    }
    pitch = head_pitch_deg(landmarks, w, h)
    assert pitch is not None
    assert abs(pitch) < 10.0


# ── episode debouncing ───────────────────────────────────────────────
def test_episode_confirm_then_release():
    t = _EpisodeTracker(confirm_seconds=2.0, release_seconds=2.5)
    now = 1000.0
    assert t.update(True, now) is None
    assert t.update(True, now + 1) is None
    assert t.update(True, now + 3) is None
    assert t.is_confirmed(now + 3)
    # 1s dropout is within release tolerance — the episode stays open.
    assert t.update(False, now + 4) is None
    ep = t.update(False, now + 7)  # gap > release → emit
    assert ep is not None
    assert ep["duration"] == 3
    assert ep["started"] == now


def test_episode_below_confirm_never_emits():
    t = _EpisodeTracker(confirm_seconds=3.0)
    t.update(True, 0.0)
    t.update(True, 1.0)
    assert t.update(False, 10.0) is None  # only 1s active — discarded


def test_episode_meta_reasons_union():
    t = _EpisodeTracker(confirm_seconds=1.0)
    t.update(True, 0.0, {"reasons": ["eyes_closed"], "ear": 0.15})
    t.update(True, 2.0, {"reasons": ["head_pose"], "pitch": 40.0})
    ep = t.update(False, 10.0)
    assert ep["meta"]["reasons"] == ["eyes_closed", "head_pose"]
    assert ep["meta"]["ear"] == 0.15 and ep["meta"]["pitch"] == 40.0


def test_episode_chunking_emits_long_episodes():
    t = _EpisodeTracker(confirm_seconds=2.0, chunk_seconds=60.0)
    t.update(True, 0.0)
    ep = t.update(True, 61.0)  # crossed the chunk boundary while active
    assert ep is not None
    assert ep["duration"] == 61
    # The episode keeps running: the next chunk starts at 61.
    assert t.start == 61.0
    t.update(True, 70.0)               # more activity in the second chunk
    ep2 = t.update(False, 75.0)        # ends → second (partial) chunk emitted
    assert ep2 is not None
    assert ep2["duration"] == 9        # 61 → 70 (last active sample)


# ── adaptive eye-closure model (EAR baseline + PERCLOS) ──────────────
def test_baseline_is_each_student_own_median():
    """A narrow-eyed student must not be judged by a global constant."""
    model = _eye_model()
    # Alert, but consistently below the 0.21 literature threshold.
    for i in range(5):
        model.observe(0.18, float(i))
    assert model.baseline == pytest.approx(0.18)
    # Their "closed" cut-off is now relative to their own eyes …
    assert model.threshold == pytest.approx(0.135)
    # … so staying at 0.18 is NOT read as drowsy, unlike with a fixed 0.21.
    assert model.perclos() == 0.0


def test_baseline_median_ignores_blinks_during_calibration():
    model = _eye_model()
    for i, ear in enumerate([0.30, 0.05, 0.31, 0.29, 0.30]):  # one blink
        model.observe(ear, float(i))
    assert model.baseline == pytest.approx(0.30)


def test_perclos_needs_sustained_closure_not_one_blink():
    model = _eye_model(adaptive=False)
    now = 1000.0
    for i in range(10):  # establish an open-eye window
        assert model.observe(0.30, now + i) is False
    assert model.observe(0.05, now + 10) is False  # single blink → not drowsy
    # Sustained closure eventually crosses the 40% PERCLOS threshold.
    drowsy = any(model.observe(0.05, now + 11 + i) for i in range(12))
    assert drowsy
    assert model.perclos() >= 0.4


def test_perclos_window_forgets_old_samples():
    model = _eye_model(adaptive=False, window_seconds=10.0)
    for i in range(10):
        model.observe(0.05, float(i))          # all closed
    assert model.perclos() == 1.0
    for i in range(10):
        model.observe(0.30, 100.0 + i)         # much later, all open
    assert model.perclos() == 0.0              # old window dropped


def test_fixed_threshold_used_until_calibrated():
    model = _eye_model(baseline_samples=100)
    assert model.calibrating()
    assert model.threshold == pytest.approx(0.21)
    model.observe(0.30, 0.0)
    assert model.calibrating()


# ── heatmap ──────────────────────────────────────────────────────────
def test_heatmap_normalises_to_peak_and_resets():
    hm = HeatmapAccumulator(8, 6)
    hm.add_point(100, 100, 1280, 720)
    hm.add_point(100, 100, 1280, 720)
    hm.add_point(1270, 710, 1280, 720)
    cells = hm.snapshot_and_reset()
    assert (0, 0, 1.0) in cells
    assert (7, 5, 0.5) in cells
    assert hm.snapshot_and_reset() == []  # reset happened


def test_heatmap_clamps_out_of_frame_points():
    hm = HeatmapAccumulator(4, 4)
    hm.add_point(-50, 99999, 640, 480)  # off-frame → clamped to a valid cell
    cells = hm.snapshot_and_reset()
    assert cells == [(0, 3, 1.0)]


# ── phone attribution ────────────────────────────────────────────────
@dataclass
class _P:
    bbox: list
    account_id: int


def test_phone_owner_prefers_nearest_face_in_reach():
    faces = [_P([100, 100, 200, 200], 11), _P([600, 100, 700, 200], 22)]
    below_face_1 = np.array([120, 260, 160, 300])
    assert BehaviourAnalysisService._phone_owner(below_face_1, faces) == 11
    below_face_2 = np.array([640, 300, 680, 340])
    assert BehaviourAnalysisService._phone_owner(below_face_2, faces) == 22


def test_phone_owner_none_when_out_of_reach():
    faces = [_P([100, 100, 200, 200], 11)]
    far_away = np.array([2000, 2000, 2050, 2050])
    assert BehaviourAnalysisService._phone_owner(far_away, faces) is None


def test_phone_above_the_face_is_not_attributed():
    """A phone is held below the face when looked at; anything above it is a
    screen, poster or reflection and must not be blamed on a student."""
    faces = [_P([100, 100, 200, 200], 11)]
    above = np.array([120, 40, 160, 80])
    assert BehaviourAnalysisService._phone_owner(above, faces) is None
    below = np.array([120, 260, 160, 300])
    assert BehaviourAnalysisService._phone_owner(below, faces) == 11


# ── privacy guarantee (PDPC) ─────────────────────────────────────────
def test_module_never_writes_image_data():
    """CR-06 privacy check: the behaviour module must only persist derived
    tuples — no image/video writing API may appear in its source."""
    src = Path(inspect.getfile(ba)).read_text(encoding="utf-8")
    for forbidden in ("imwrite", "VideoWriter", "imencode"):
        assert forbidden not in src, f"{forbidden} found in behaviourAnalysis.py"


# ── DB round-trip (CI only — needs DATABASE_URL + schema.sql applied) ─
# Guarded on CI=true, not just DATABASE_URL: importing `core` loads .env
# via attendancePipeline, so DATABASE_URL alone could silently point this
# test at the production Supabase when run on a developer machine.
@pytest.mark.skipif(
    not (os.getenv("CI") and os.getenv("DATABASE_URL")),
    reason="runs only in CI against the disposable pgvector Postgres service",
)
def test_event_and_heatmap_rows_round_trip():
    import psycopg2

    db = os.environ["DATABASE_URL"]
    # The real config, not a hand-rolled stub. A SimpleNamespace here only
    # duplicated AIConfig field for field, and drifted the moment
    # BehaviourAnalysisService.__init__ started reading cfg.ctx_id — a break
    # this test could not catch locally, since it only runs in CI. The values
    # are irrelevant to what is asserted below: _write_events() and
    # _write_heatmap() never touch cfg, and conftest pins AI_CTX_ID=-1 so the
    # phone detector is constructed for CPU (it loads its model lazily anyway).
    cfg = AIConfig()
    svc = BehaviourAnalysisService(cfg, db)

    with psycopg2.connect(db) as conn, conn.cursor() as cur:
        cur.execute("SELECT profileid FROM user_profiles WHERE role='student' LIMIT 1")
        profile_id = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO user_account (profileid, email, password_hash)
               VALUES (%s, %s, 'x') RETURNING accountid""",
            (profile_id, f"beh-test-{time.time()}@test.local"),
        )
        account_id = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO course (course_code, course_name)
               VALUES (%s, 'Behaviour Test') RETURNING courseid""",
            (f"BEH{int(time.time()) % 100000}",),
        )
        course_id = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO attendance_session (courseid, start_time, status)
               VALUES (%s, NOW(), 'active') RETURNING attendancesessionid""",
            (course_id,),
        )
        session_id = cur.fetchone()[0]

    try:
        n = svc._write_events(session_id, [
            (account_id, "drowsiness", time.time() - 30, 12, None,
             {"reasons": ["eyes_closed"], "ear": 0.15}),
            (account_id, "phone", time.time() - 10, 5, 0.7, {"conf": 0.7}),
        ])
        assert n == 2
        m = svc._write_heatmap(session_id, [(0, 0, 1.0), (3, 2, 0.5)])
        assert m == 2

        with psycopg2.connect(db) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT event_type, duration_seconds, metadata->>'reasons' IS NOT NULL
                   FROM behaviour_event WHERE attendancesessionid = %s
                   ORDER BY event_type""",
                (session_id,),
            )
            rows = cur.fetchall()
            assert [r[0] for r in rows] == ["drowsiness", "phone"]
            assert rows[0][1] == 12
            cur.execute(
                "SELECT COUNT(*) FROM heatmap_snapshot WHERE attendancesessionid = %s",
                (session_id,),
            )
            assert cur.fetchone()[0] == 2
    finally:
        # behaviour rows cascade with the session; then remove the fixtures.
        with psycopg2.connect(db) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM attendance_session WHERE attendancesessionid = %s", (session_id,))
            cur.execute("DELETE FROM course WHERE courseid = %s", (course_id,))
            cur.execute("DELETE FROM user_account WHERE accountid = %s", (account_id,))
