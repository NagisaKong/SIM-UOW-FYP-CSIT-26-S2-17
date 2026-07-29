"""Shared pytest fixtures for unit + System Testing suites.

- Makes the repo root importable.
- Loads `.env` (DATABASE_URL, JWT_SECRET, …) without overriding explicit env.
- Forces lightweight AI flags so TestClient startup does not load SCRFD/ArcFace.
- Provides a stub AttendancePipeline (deterministic fake embeddings) and an
  optional seeded multi-role world for API/DB system tests.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

# Keep app import / lifespan free of heavy models during tests.
for _k, _v in {
    "AI_USE_MTCNN": "false",
    "AI_USE_FACENET": "false",
    "AI_USE_ENHANCER": "false",
    "AI_BEHAVIOUR": "false",
    "AI_DEVICE": "cpu",
    "AI_CTX_ID": "-1",
    "APP_ENV": "development",
}.items():
    os.environ.setdefault(_k, _v)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_db: needs a reachable DATABASE_URL (CI Postgres or local .env)",
    )


def db_available() -> bool:
    return bool(os.getenv("DATABASE_URL"))


requires_db = pytest.mark.skipif(
    not db_available(),
    reason="DATABASE_URL not set — skip DB-backed system tests",
)


# ── Stub pipeline (no InsightFace / YOLO weights) ─────────────────────
def _fake_embed(image: np.ndarray) -> np.ndarray:
    """Deterministic 512-d unit vector from image bytes (same image → same vec)."""
    payload = image.tobytes()[:8192]
    digest = hashlib.sha256(payload).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    vec = rng.randn(512).astype(np.float32)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def build_stub_pipeline(cfg=None):
    """AttendancePipeline stand-in that persists/matches fake embeddings via DB."""
    import psycopg2

    from core.attendancePipeline import (
        ARCFACE_DIM,
        ARCFACE_MODEL_NAME,
        AIConfig,
        AttendancePipeline,
        Detection,
        EmbeddingRepo,
        EmbeddingStore,
        FrameResult,
        Prediction,
        SupabaseEmbeddingStore,
    )

    cfg = cfg or AIConfig()
    cfg.use_mtcnn = False
    cfg.use_facenet = False
    cfg.use_enhancer = False
    cfg.behaviour_enabled = False

    repo = EmbeddingRepo(cfg.database_url)
    manager = SupabaseEmbeddingStore(repo=repo)
    store = EmbeddingStore(ARCFACE_MODEL_NAME, ARCFACE_DIM, cfg.arcface_threshold)
    manager.register_store(store)
    with_suppress = True
    try:
        manager.hydrate_all()
    except Exception:
        if not with_suppress:
            raise

    pipe = object.__new__(AttendancePipeline)
    pipe.cfg = cfg
    pipe.store_manager = manager
    pipe.enhancer = None
    pipe._arcface = None
    pipe._scrfd = None
    pipe._mtcnn = None
    pipe._facenet = None
    pipe._weights = {ARCFACE_MODEL_NAME: 1.0}

    def enrol_student(
        account_id: int, images: list,
        reject_multiple: bool = False, append: bool = False,
    ):
        # Signature and append semantics must mirror
        # AttendancePipeline.enrol_student — otherwise the multi-angle
        # gallery path is never exercised by the suite.
        if not images:
            return {}
        # Simulate one face per enrolment photo.
        vec = _fake_embed(images[0])
        written = manager.enrol_account(
            account_id=account_id,
            vectors_by_model={ARCFACE_MODEL_NAME: [vec]},
            retention_days=cfg.embedding_retention_days,
            model_versions={ARCFACE_MODEL_NAME: "stub-test"},
            deactivate_previous=not append,
        )
        # Keep StudentInfo in sync so scan/identify labels resolve.
        try:
            with psycopg2.connect(cfg.database_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT student_id, full_name FROM personal_info WHERE accountid = %s",
                    (account_id,),
                )
                row = cur.fetchone()
            from core.attendancePipeline import StudentInfo

            info = StudentInfo(
                account_id=account_id,
                student_id=row[0] if row else None,
                full_name=row[1] if row else None,
            )
            # enrol_account() already inserted the vector; only the label is
            # missing. upsert() here would add a second copy in append mode.
            store.set_info(account_id, info)
        except Exception:
            pass
        return written

    def process_frame(frame: np.ndarray) -> FrameResult:
        h, w = frame.shape[:2]
        bbox = [int(w * 0.25), int(h * 0.2), int(w * 0.75), int(h * 0.85)]
        query = _fake_embed(frame)
        account_id, score = store.best_match(query)
        info = store.info_for(account_id) if account_id is not None else None
        recognised = account_id is not None
        pred = Prediction(
            bbox=bbox,
            recognised=recognised,
            account_id=account_id,
            student_id=info.student_id if info else None,
            full_name=info.full_name if info else None,
            score=float(score),
            per_model={ARCFACE_MODEL_NAME: {"score": float(score)}},
            det_score=0.99,
        )
        det = Detection(
            bbox=np.array(bbox, dtype=np.float32),
            det_score=0.99,
            kps=None,
            source="scrfd",
        )
        return FrameResult(predictions=[pred], enhanced=False, detections=[det])

    pipe.enrol_student = enrol_student  # type: ignore[method-assign]
    pipe.process_frame = process_frame  # type: ignore[method-assign]
    pipe.draw = lambda frame, result: frame  # type: ignore[method-assign]
    return pipe


def _png_bytes(seed: int = 0, size: int = 160) -> bytes:
    """Small PNG for multipart uploads (content varies with seed).

    Size defaults above the enrolment quality gate's minimum edge
    (AI_ENROL_MIN_IMAGE_PX, see core/facialImage.assess_image_quality) so
    fixture uploads exercise the real enrolment path rather than being
    rejected as too small. Random-noise content is sharp and mid-grey, so
    it passes the blur and exposure checks.

    Prefer Pillow; fall back to OpenCV; last resort a minimal hand-rolled PNG.
    """
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        pass
    try:
        import cv2

        ok, buf = cv2.imencode(".png", img)
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    # Minimal valid 1×1 PNG (content still varies via seed in IHDR unused — use
    # different raw RGB for different seeds via tEXt is overkill; embed seed in
    # a custom ancillary chunk isn't needed — vary by repeating encode of mean).
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(img[y, :, :].reshape(-1)) for y in range(size))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


@pytest.fixture(scope="session")
def png_bytes():
    return _png_bytes


@pytest.fixture(scope="session")
def client() -> Iterator[Any]:
    """FastAPI TestClient with stubbed pipeline (module lifespan runs once)."""
    with patch(
        "core.attendancePipeline.AttendancePipeline.from_env",
        side_effect=build_stub_pipeline,
    ), patch(
        "main_api.AttendancePipeline.from_env",
        side_effect=build_stub_pipeline,
    ):
        from fastapi.testclient import TestClient

        from main_api import app

        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="session")
def db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return url


@pytest.fixture(scope="session")
def st_suffix() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def st_world(client, db_url: str, st_suffix: str, png_bytes) -> Iterator[SimpleNamespace]:
    """Seed admin / teacher / 3 students + course + session; clean up afterwards.

    All rows are tagged with the unique st_suffix so cleanup is safe.
    """
    import psycopg2

    from core.userInformation import _hash_password

    # Purge leftover System-Testing rows from prior interrupted runs so
    # stub embeddings cannot collide across sessions.
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT accountid FROM user_account WHERE email LIKE 'st-%%@test.local'"
        )
        stale_ids = [r[0] for r in cur.fetchall()]
        if stale_ids:
            cur.execute(
                """SELECT attendancesessionid FROM attendance_session
                   WHERE courseid IN (
                     SELECT courseid FROM course WHERE course_code LIKE 'ST%%'
                   )"""
            )
            stale_sessions = [r[0] for r in cur.fetchall()]
            for sid in stale_sessions:
                for sql in (
                    "DELETE FROM behaviour_event WHERE attendancesessionid = %s",
                    "DELETE FROM heatmap_snapshot WHERE attendancesessionid = %s",
                    "DELETE FROM presence_check WHERE attendancesessionid = %s",
                    "DELETE FROM session_recording WHERE attendancesessionid = %s",
                    """DELETE FROM attendance_appeal WHERE attendancerecordid IN (
                         SELECT attendancerecordid FROM attendance_record
                         WHERE attendancesessionid = %s)""",
                    "DELETE FROM attendance_record WHERE attendancesessionid = %s",
                    "DELETE FROM leave_application WHERE attendancesessionid = %s",
                    "DELETE FROM attendance_session WHERE attendancesessionid = %s",
                ):
                    cur.execute(sql, (sid,))
            cur.execute("DELETE FROM course_enrollment WHERE courseid IN (SELECT courseid FROM course WHERE course_code LIKE 'ST%%')")
            cur.execute("DELETE FROM behaviour_config WHERE courseid IN (SELECT courseid FROM course WHERE course_code LIKE 'ST%%')")
            cur.execute("DELETE FROM course WHERE course_code LIKE 'ST%%'")
            cur.execute("DELETE FROM face_embedding WHERE accountid = ANY(%s)", (stale_ids,))
            cur.execute("DELETE FROM model_configs WHERE updated_by = ANY(%s)", (stale_ids,))
            # The singleton threshold config may reference an ST admin via
            # updated_by (ST-UF-34) — NULL it so the account can be deleted.
            cur.execute(
                "UPDATE attendance_threshold_config SET updated_by = NULL "
                "WHERE updated_by = ANY(%s)",
                (stale_ids,),
            )
            cur.execute(
                "UPDATE behaviour_config SET updated_by = NULL WHERE updated_by = ANY(%s)",
                (stale_ids,),
            )
            cur.execute("DELETE FROM personal_info WHERE accountid = ANY(%s)", (stale_ids,))
            cur.execute("DELETE FROM user_account WHERE accountid = ANY(%s)", (stale_ids,))
        conn.commit()

    # Clear in-memory gallery so stale vectors cannot win best_match.
    try:
        client.app.state.pipeline.store_manager.reload()
    except Exception:
        pass

    password = "StTestPass123!"
    pwd_hash = _hash_password(password)
    emails = {
        "admin": f"st-admin-{st_suffix}@test.local",
        "teacher": f"st-teacher-{st_suffix}@test.local",
        "s1": f"st-s1-{st_suffix}@test.local",
        "s2": f"st-s2-{st_suffix}@test.local",
        "s3": f"st-s3-{st_suffix}@test.local",
    }
    names = {
        "admin": f"ST Admin {st_suffix}",
        "teacher": f"ST Teacher {st_suffix}",
        "s1": f"ST Student1 {st_suffix}",
        "s2": f"ST Student2 {st_suffix}",
        "s3": f"ST Student3 {st_suffix}",
    }
    course_code = f"ST{st_suffix.upper()[:6]}"
    account_ids: dict[str, int] = {}
    course_id: int | None = None
    session_id: int | None = None

    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        for role_key, role in (
            ("admin", "admin"),
            ("teacher", "teacher"),
            ("s1", "student"),
            ("s2", "student"),
            ("s3", "student"),
        ):
            cur.execute(
                "SELECT profileid FROM user_profiles WHERE role = %s LIMIT 1", (role,)
            )
            row = cur.fetchone()
            if not row:
                pytest.skip("user_profiles seed missing — apply database/schema.sql")
            profile_id = row[0]
            cur.execute(
                """INSERT INTO user_account (profileid, email, password_hash)
                   VALUES (%s, %s, %s) RETURNING accountid""",
                (profile_id, emails[role_key], pwd_hash),
            )
            acc = cur.fetchone()[0]
            account_ids[role_key] = acc
            if role == "student":
                cur.execute(
                    """INSERT INTO personal_info (accountid, full_name, student_id, staff_id)
                       VALUES (%s, %s, %s, NULL)""",
                    (acc, names[role_key], f"ST{acc:05d}"),
                )
            else:
                cur.execute(
                    """INSERT INTO personal_info (accountid, full_name, student_id, staff_id)
                       VALUES (%s, %s, NULL, %s)""",
                    (acc, names[role_key], f"STA{acc:05d}"),
                )
        conn.commit()

    def login(role_key: str) -> str:
        r = client.post(
            "/auth/login",
            json={"email": emails[role_key], "password": password},
        )
        assert r.status_code == 200, r.text
        return r.json()["token"]

    tokens = {k: login(k) for k in emails}

    # Admin creates course + enrolments + session via API
    r = client.post(
        "/admin/courses",
        headers={"Authorization": f"Bearer {tokens['admin']}"},
        json={
            "course_code": course_code,
            "course_name": f"System Testing {st_suffix}",
            "teacher_id": account_ids["teacher"],
        },
    )
    assert r.status_code == 200, r.text
    course_id = r.json()["course_id"]

    for sk in ("s1", "s2", "s3"):
        r = client.post(
            f"/admin/courses/{course_id}/enrollments",
            headers={"Authorization": f"Bearer {tokens['admin']}"},
            json={"account_id": account_ids[sk]},
        )
        assert r.status_code == 200, r.text

    r = client.post(
        "/admin/sessions",
        headers={"Authorization": f"Bearer {tokens['admin']}"},
        json={
            "course_id": course_id,
            "start_time": "2030-01-15T09:00:00+08:00",
            "end_time": "2030-01-15T11:00:00+08:00",
            "status": "scheduled",
        },
    )
    assert r.status_code == 200, r.text
    session_id = r.json()["session_id"]

    world = SimpleNamespace(
        password=password,
        emails=emails,
        names=names,
        account_ids=account_ids,
        tokens=tokens,
        course_id=course_id,
        course_code=course_code,
        session_id=session_id,
        suffix=st_suffix,
        student_png=_png_bytes(seed=101),
        other_png=_png_bytes(seed=202),
        auth=lambda role: {"Authorization": f"Bearer {tokens[role]}"},
    )

    yield world

    # Cleanup (FK-safe) — wipe every session for this ST course, then the course.
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        ids = list(account_ids.values())
        session_ids: list[int] = []
        if course_id is not None:
            cur.execute(
                "SELECT attendancesessionid FROM attendance_session WHERE courseid = %s",
                (course_id,),
            )
            session_ids = [r[0] for r in cur.fetchall()]
        elif session_id is not None:
            session_ids = [session_id]

        for sid in session_ids:
            cur.execute(
                "DELETE FROM behaviour_event WHERE attendancesessionid = %s", (sid,)
            )
            cur.execute(
                "DELETE FROM heatmap_snapshot WHERE attendancesessionid = %s", (sid,)
            )
            cur.execute(
                "DELETE FROM presence_check WHERE attendancesessionid = %s", (sid,)
            )
            cur.execute(
                "DELETE FROM session_recording WHERE attendancesessionid = %s", (sid,)
            )
            cur.execute(
                """DELETE FROM attendance_appeal WHERE attendancerecordid IN (
                     SELECT attendancerecordid FROM attendance_record
                     WHERE attendancesessionid = %s)""",
                (sid,),
            )
            cur.execute(
                "DELETE FROM attendance_record WHERE attendancesessionid = %s", (sid,)
            )
            cur.execute(
                "DELETE FROM leave_application WHERE attendancesessionid = %s", (sid,)
            )
            cur.execute(
                "DELETE FROM attendance_session WHERE attendancesessionid = %s", (sid,)
            )

        if course_id is not None:
            cur.execute(
                "DELETE FROM course_enrollment WHERE courseid = %s", (course_id,)
            )
            cur.execute("DELETE FROM behaviour_config WHERE courseid = %s", (course_id,))
            cur.execute("DELETE FROM course WHERE courseid = %s", (course_id,))
        if ids:
            cur.execute("DELETE FROM face_embedding WHERE accountid = ANY(%s)", (ids,))
            cur.execute(
                "DELETE FROM model_configs WHERE updated_by = ANY(%s)", (ids,)
            )
            # Singleton/global config rows survive the run — NULL any
            # updated_by pointing at ST accounts (set by ST-UF-34 / U35).
            cur.execute(
                "UPDATE attendance_threshold_config SET updated_by = NULL "
                "WHERE updated_by = ANY(%s)",
                (ids,),
            )
            cur.execute(
                "UPDATE behaviour_config SET updated_by = NULL WHERE updated_by = ANY(%s)",
                (ids,),
            )
            cur.execute("DELETE FROM personal_info WHERE accountid = ANY(%s)", (ids,))
            cur.execute("DELETE FROM user_account WHERE accountid = ANY(%s)", (ids,))
        conn.commit()


# ════════════════════════════════════════════════════════════════════════
# System-test result reporting (Part D evidence)
# ════════════════════════════════════════════════════════════════════════
# The suite is written so every test maps 1:1 onto a case ID in the Final
# Technical Documentation (ST-UF-07 ↔ U07, and so on). pytest's default dot
# output loses that mapping, so filling in the "Pass?" column meant reading
# the source. These hooks print the results grouped by case ID and write the
# same table to docs/evidence/, which is what the document cites.

_ST_DOMAINS = {
    "BS": "Basic Settings",
    "TK": "Task-driven",
    "EX": "Exception",
    "DB": "Database accuracy",
    "UF": "User functional requirements",
    "SEC": "Network security",
    "ML": "ML algorithm accuracy",
    "BH": "Behaviour analysis",
}
_ST_ORDER = ["BS", "TK", "EX", "DB", "UF", "SEC", "ML", "BH"]
_ST_PATTERN = re.compile(r"_st_(bs|tk|ex|db|uf|sec|ml|bh)_(\d+)", re.IGNORECASE)

# nodeid -> {"id", "domain", "outcome", "duration", "detail"}
_st_results: dict[str, dict[str, Any]] = {}


def _case_id(nodeid: str) -> tuple[str, str] | None:
    """('UF', 'ST-UF-07') from a node id, or None for non-ST tests."""
    match = _ST_PATTERN.search(nodeid)
    if not match:
        return None
    domain = match.group(1).upper()
    return domain, f"ST-{domain}-{match.group(2).zfill(2)}"


def pytest_runtest_logreport(report) -> None:
    identified = _case_id(report.nodeid)
    if identified is None:
        return
    domain, case_id = identified

    # A case counts as failed if ANY phase failed: a passing test body whose
    # fixture teardown blew up has not really demonstrated the requirement.
    entry = _st_results.setdefault(
        report.nodeid,
        {"id": case_id, "domain": domain, "outcome": "passed",
         "duration": 0.0, "detail": ""},
    )
    entry["duration"] += report.duration
    if report.failed:
        entry["outcome"] = "failed"
        text = str(report.longrepr or "").strip().splitlines()
        detail = next(
            (ln.strip() for ln in reversed(text) if ln.strip().startswith("E ")), "",
        )
        entry["detail"] = (detail[2:].strip() or "see traceback")[:160]
    elif report.skipped and entry["outcome"] == "passed":
        entry["outcome"] = "skipped"


def _summary_rows() -> list[dict[str, Any]]:
    """One row per case ID, in document order."""
    by_case: dict[str, dict[str, Any]] = {}
    for entry in _st_results.values():
        current = by_case.get(entry["id"])
        # Duplicate IDs would be a numbering mistake; surface the worst result.
        if current is None or (current["outcome"] == "passed" != entry["outcome"]):
            by_case[entry["id"]] = entry
    def sort_key(e: dict[str, Any]) -> tuple[int, int]:
        num = e["id"].rsplit("-", 1)[-1]
        return (_ST_ORDER.index(e["domain"]), int(num) if num.isdigit() else 0)
    return sorted(by_case.values(), key=sort_key)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    rows = _summary_rows()
    if not rows:
        return

    tw = terminalreporter
    tw.write_line("")
    tw.write_sep("═", "SYSTEM TESTING RESULTS — FYP-26-S2-17", bold=True)

    totals = {"passed": 0, "failed": 0, "skipped": 0}
    for domain in _ST_ORDER:
        cases = [r for r in rows if r["domain"] == domain]
        if not cases:
            continue
        passed = sum(1 for c in cases if c["outcome"] == "passed")
        failed = sum(1 for c in cases if c["outcome"] == "failed")
        skipped = len(cases) - passed - failed
        totals["passed"] += passed
        totals["failed"] += failed
        totals["skipped"] += skipped

        pct = passed / len(cases)
        bar = "█" * round(pct * 10) + "░" * (10 - round(pct * 10))
        tw.write(f"  ST-{domain:<4}{_ST_DOMAINS[domain]:<32}")
        tw.write(f"{passed:>3}/{len(cases):<4}", green=failed == 0, red=failed > 0)
        tw.write_line(f" {bar} {pct:>4.0%}")

        # Only failures are itemised — a green run stays short.
        for case in cases:
            if case["outcome"] == "failed":
                tw.write_line(f"       ✗ {case['id']}  {case['detail']}", red=True)
            elif case["outcome"] == "skipped":
                tw.write_line(f"       – {case['id']}  skipped", yellow=True)

    tw.write_sep("─")
    tw.write(f"  {len(rows)} case(s): ")
    tw.write(f"{totals['passed']} passed", green=True)
    if totals["failed"]:
        tw.write(f" · {totals['failed']} failed", red=True)
    if totals["skipped"]:
        tw.write(f" · {totals['skipped']} skipped", yellow=True)
    tw.write_line("")

    path = _write_evidence(rows, totals)
    if path:
        tw.write_line(f"  Evidence written to {path}", cyan=True)
    tw.write_sep("═")


def _write_evidence(rows: list[dict[str, Any]], totals: dict[str, int]) -> str | None:
    """Persist the same table as markdown for the FTD's Pass? column."""
    import datetime as _dt

    try:
        out_dir = _REPO_ROOT / "docs" / "evidence"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now()
        path = out_dir / f"system_test_{stamp:%Y%m%d_%H%M}.md"
        lines = [
            "# System Testing results — FYP-26-S2-17",
            "",
            f"Generated {stamp:%Y-%m-%d %H:%M} by `pytest tests/`.",
            "Each row is produced by the automated suite; the ID matches the",
            "case tables in the Final Technical Documentation, Part D §3.3.",
            "",
            "| ID | Domain | Result | Duration | Detail |",
            "|---|---|---|---|---|",
        ]
        symbol = {"passed": "Y", "failed": "N", "skipped": "skipped"}
        for row in rows:
            lines.append(
                f"| {row['id']} | {_ST_DOMAINS[row['domain']]} | "
                f"{symbol[row['outcome']]} | {row['duration']:.2f}s | "
                f"{row['detail'] or ''} |"
            )
        lines += [
            "",
            f"**Totals — planned {len(rows)} · passed {totals['passed']} · "
            f"failed {totals['failed']} · skipped {totals['skipped']} · "
            f"pass rate {totals['passed'] / len(rows):.0%}**",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path.relative_to(_REPO_ROOT))
    except Exception as exc:  # noqa: BLE001 — reporting must never fail a run
        print(f"[evidence] could not write summary: {exc}")
        return None
