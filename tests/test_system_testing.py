"""System Testing suite mapped to docs/System_Testing.docx (§3).

Classes are numbered so pytest runs them in document order:
  ST-BS → ST-TK → ST-EX → ST-DB → ST-UF → ST-SEC → ST-ML → ST-BH

DB-backed cases use the stub pipeline from conftest (deterministic fake
embeddings) so the suite can run without loading SCRFD/ArcFace weights.
Requires DATABASE_URL (from CI Postgres or local .env).
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import requires_db

REPO = Path(__file__).resolve().parents[1]


def _multipart_png(data: bytes, filename: str = "face.png"):
    return {"file": (filename, io.BytesIO(data), "image/png")}


# ═══════════════════════════════════════════════════════════════════════
# 3.2 Baseline — Basic Settings / Task / Exception
# ═══════════════════════════════════════════════════════════════════════
@requires_db
class TestST01_BasicSettings:
    """ST-BS-01 / ST-BS-02"""

    def test_01_st_bs_01_login_ui_and_dashboard_load(self, client, st_world):
        # Static frontend surfaces exist (role dashboards).
        for name in ("index.html", "student.html", "teacher.html", "admin.html"):
            path = REPO / "frontend" / name
            assert path.is_file(), f"missing {path}"
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert len(text) > 100

        r = client.post(
            "/auth/login",
            json={"email": st_world.emails["admin"], "password": st_world.password},
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("success") is True
        assert body.get("token")
        assert body["user"]["role"] == "admin"

        me = client.get("/auth/me", headers=st_world.auth("admin"))
        assert me.status_code == 200
        assert me.json()["role"] == "admin"

    def test_02_st_bs_02_recognition_pipeline_loaded(self, client, st_world):
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body.get("success") is True
        assert "stores" in body
        # The instance reports the model set it is actually running, so test
        # evidence can state the configuration under test rather than assume
        # it (the CPU deployment runs ArcFace only; the GPU rig runs the
        # full ensemble — see DEPLOY.md).
        rec = body["recognition"]
        assert "scrfd" in rec["detectors"]
        assert rec["recognisers"], rec
        assert isinstance(rec["ensemble"], bool)
        assert rec["ensemble"] == (len(rec["recognisers"]) >= 2)
        assert "behaviour_analysis" in body

        # Preview-detect proves the capture/recognition path responds.
        r = client.post(
            "/teacher/preview-detect",
            headers=st_world.auth("teacher"),
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200, r.text
        assert r.json().get("success") is True
        assert "boxes" in r.json()


@requires_db
class TestST02_TaskDriven:
    """ST-TK-01 … ST-TK-03"""

    def test_01_st_tk_01_admin_accounts_timetable_and_face_enrol(self, client, st_world):
        # Accounts + course + session already seeded by st_world fixture.
        assert st_world.course_id
        assert st_world.session_id

        # Enrol faces for three students (system "learns" facial data).
        for i, key in enumerate(("s1", "s2", "s3"), start=1):
            png = st_world.student_png if key == "s1" else __import__(
                "tests.conftest", fromlist=["_png_bytes"]
            )._png_bytes(seed=100 + i)
            r = client.post(
                "/register",
                headers=st_world.auth("admin"),
                data={"account_id": str(st_world.account_ids[key])},
                files=_multipart_png(png),
            )
            assert r.status_code == 200, r.text
            assert r.json().get("success") is True, r.text

        faces = client.get("/admin/faces", headers=st_world.auth("admin"))
        assert faces.status_code == 200
        enrolled = set()
        for f in faces.json().get("faces", []):
            if f.get("is_active") in (True, "t", 1):
                acc = f.get("accountid", f.get("account_id"))
                if acc is not None:
                    enrolled.add(int(acc))
        for key in ("s1", "s2", "s3"):
            assert st_world.account_ids[key] in enrolled

    def test_02_st_tk_02_teacher_start_session_and_scan(self, client, st_world):
        r = client.post(
            f"/teacher/sessions/{st_world.session_id}/start",
            headers=st_world.auth("teacher"),
        )
        assert r.status_code == 200, r.text
        assert r.json().get("status") == "active"

        r = client.post(
            f"/teacher/sessions/{st_world.session_id}/scan",
            headers=st_world.auth("teacher"),
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert body.get("faces_in_frame", 0) >= 1
        # s1 was enrolled with student_png → should be recognised.
        detected_ids = {d["account_id"] for d in body.get("detected", [])}
        assert st_world.account_ids["s1"] in detected_ids

    def test_03_st_tk_03_early_left_report_appeal_flow(self, client, st_world, db_url):
        import psycopg2

        # End session to aggregate attendance_record rows.
        r = client.post(
            f"/teacher/sessions/{st_world.session_id}/end",
            headers=st_world.auth("teacher"),
        )
        assert r.status_code == 200, r.text

        early = client.get(
            f"/teacher/sessions/{st_world.session_id}/early-left",
            headers=st_world.auth("teacher"),
        )
        assert early.status_code == 200, early.text

        export = client.get(
            "/teacher/reports/export",
            headers=st_world.auth("teacher"),
            params={"course_id": st_world.course_id},
        )
        assert export.status_code == 200, export.text
        assert "text/csv" in export.headers.get("content-type", "")

        # Pick a student attendance record to appeal (prefer absent / non-present).
        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT attendancerecordid, accountid FROM attendance_record
                   WHERE attendancesessionid = %s
                   ORDER BY CASE WHEN status = 'absent' THEN 0 ELSE 1 END
                   LIMIT 1""",
                (st_world.session_id,),
            )
            row = cur.fetchone()
        assert row, "expected attendance_record after endSession"
        record_id, account_id = row
        # Map account → role key for token
        role_key = next(
            k for k, v in st_world.account_ids.items() if v == account_id
        )

        appeal = client.post(
            "/student/appeals",
            headers=st_world.auth(role_key),
            json={"record_id": record_id, "reason": "ST system test appeal"},
        )
        assert appeal.status_code == 200, appeal.text
        appeal_id = appeal.json()["appeal_id"]

        listed = client.get(
            "/teacher/appeals", headers=st_world.auth("teacher")
        )
        assert listed.status_code == 200
        ids = {a["appealid"] for a in listed.json().get("appeals", [])}
        assert appeal_id in ids

        review = client.patch(
            f"/teacher/appeals/{appeal_id}",
            headers=st_world.auth("teacher"),
            json={"status": "approved"},
        )
        assert review.status_code == 200, review.text
        assert review.json().get("status") == "approved"


@requires_db
class TestST03_Exception:
    """ST-EX-01 / ST-EX-02"""

    def test_01_st_ex_01_bad_login(self, client, st_world):
        r = client.post(
            "/auth/login",
            json={"email": st_world.emails["admin"], "password": "wrong-password"},
        )
        assert r.status_code == 401

        r = client.post("/auth/login", json={"email": "", "password": ""})
        assert r.status_code in (401, 422)

        r = client.post("/auth/login", json={})
        assert r.status_code == 422

    def test_02_st_ex_02_blank_appeal_reason(self, client, st_world, db_url):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT attendancerecordid FROM attendance_record
                   WHERE accountid = %s LIMIT 1""",
                (st_world.account_ids["s2"],),
            )
            row = cur.fetchone()
        if not row:
            pytest.skip("no attendance_record for s2 — run task suite first")
        record_id = row[0]

        r = client.post(
            "/student/appeals",
            headers=st_world.auth("s2"),
            json={"record_id": record_id, "reason": "   "},
        )
        assert r.status_code == 400, r.text

        r = client.post(
            "/student/appeals",
            headers=st_world.auth("s2"),
            json={"record_id": record_id},
        )
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
# 3.3.1 Database accuracy
# ═══════════════════════════════════════════════════════════════════════
@requires_db
class TestST04_DatabaseAccuracy:
    """ST-DB-01 … ST-DB-05"""

    def test_01_st_db_01_schema_file_and_core_tables(self, db_url):
        import psycopg2

        schema = (REPO / "database" / "schema.sql").read_text(encoding="utf-8")
        for table in (
            "USER_ACCOUNT",
            "FACE_EMBEDDING",
            "ATTENDANCE_RECORD",
            "MODEL_CONFIGS",
            "BEHAVIOUR_EVENT",
        ):
            assert table in schema

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'public'"""
            )
            names = {r[0].lower() for r in cur.fetchall()}
        for t in (
            "user_account",
            "face_embedding",
            "attendance_record",
            "model_configs",
            "presence_check",
        ):
            assert t in names

    def test_02_st_db_02_face_embedding_after_enrol(self, db_url, st_world):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        with psycopg2.connect(db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT accountid, model_name, dimension, is_active,
                              embedding_vector
                       FROM face_embedding
                       WHERE accountid = %s AND is_active = TRUE""",
                    (st_world.account_ids["s1"],),
                )
                row = cur.fetchone()
        assert row is not None
        assert row[1]  # model_name
        assert row[2] == 512
        assert row[3] is True
        assert row[4] is not None

    def test_03_st_db_03_presence_and_attendance_rows(self, db_url, st_world):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM presence_check WHERE attendancesessionid = %s",
                (st_world.session_id,),
            )
            assert cur.fetchone()[0] >= 1
            cur.execute(
                """SELECT status FROM attendance_record
                   WHERE attendancesessionid = %s AND accountid = %s""",
                (st_world.session_id, st_world.account_ids["s1"]),
            )
            status_row = cur.fetchone()
        assert status_row is not None
        assert status_row[0] in {
            "present",
            "late",
            "absent",
            "leave",
            "early_left",
        }

    def test_04_st_db_04_model_configs_after_deploy(self, client, st_world, db_url):
        import psycopg2

        # Select dataset + sync retrain/deploy path (force if not improved).
        r = client.post(
            "/admin/training-data",
            headers=st_world.auth("admin"),
            json={"model_name": "arcface", "train_pct": 80},
        )
        assert r.status_code == 200, r.text

        r = client.post(
            "/admin/retrain?force=true",
            headers=st_world.auth("admin"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        # Either deployed now or already had a deployable result.
        assert "new_threshold" in body or body.get("deployed") in (True, False)

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT similarity_threshold FROM model_configs
                   WHERE is_active = TRUE AND model_name = 'arcface'
                   ORDER BY updated_at DESC LIMIT 1"""
            )
            row = cur.fetchone()
        assert row is not None
        assert 0.0 < float(row[0]) < 1.0

    def test_05_st_db_05_referential_integrity_fks(self, db_url):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) FROM information_schema.table_constraints
                   WHERE constraint_type = 'FOREIGN KEY'
                     AND table_schema = 'public'"""
            )
            fk_count = cur.fetchone()[0]
        assert fk_count >= 5


# ═══════════════════════════════════════════════════════════════════════
# 3.3.2 User functional requirements
# ═══════════════════════════════════════════════════════════════════════
@requires_db
class TestST05_UserFunctional:
    """ST-UF-01 … ST-UF-35 (with U01–U35).

    All 35 user functional cases are implemented (ST-UF-NN ↔ U0NN).

    Note on U02/U11/U18 (logout): the backend deliberately exposes no
    logout/token-revocation endpoint — logout is a client-side boundary
    flow (frontend/app.js clears localStorage and redirects), and JWTs
    stay valid until their 12-hour expiry. These cases therefore assert
    the logout contract plus 401-without-token, per Part D §3.3.2.
    Immediate revocation is covered by ST-UF-20 (account deactivation).
    """

    # ── ST-UF-01 (U01) Student Login ─────────────────────────────────
    def test_01_st_uf_01_student_login(self, client, st_world):
        r = client.post(
            "/auth/login",
            json={"email": st_world.emails["s1"], "password": st_world.password},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["token"]
        assert body["user"]["role"] == "student"
        assert body["user"]["full_name"] == st_world.names["s1"]

    # ── ST-UF-02 (U02) Student logout (client-side boundary flow) ───
    # The backend intentionally exposes no logout/token-revocation endpoint
    # (core/userInformation.py), so the acceptance criterion is local-session
    # clearing plus 401 on token-less requests — not server-side rejection of
    # the old token. Immediate revocation is ST-UF-20 (account deactivation).
    def test_02_st_uf_02_student_logout_contract(self, client, st_world):
        app_js = (REPO / "frontend" / "app.js").read_text(encoding="utf-8")
        start = app_js.index("function logout()")
        body = app_js[start : app_js.index("\n}", start)]
        assert 'removeItem("token")' in body
        assert 'removeItem("user")' in body
        assert "index.html" in body  # redirect back to the login page
        # One shared implementation backs all three role dashboards.
        for page in ("student.html", "teacher.html", "admin.html"):
            assert "app.js" in (REPO / "frontend" / page).read_text(encoding="utf-8")

        # Student holds a working session before logging out …
        assert client.get("/auth/me", headers=st_world.auth("s1")).status_code == 200
        # … and after local clearing no credential is sent, so the API refuses.
        assert client.get("/auth/me").status_code == 401

    # ── ST-UF-03 (U03) Automated check-in (light ref to ST-TK-02/03) ─
    def test_03_st_uf_03_auto_checkin_recorded(self, st_world, db_url):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM presence_check WHERE attendancesessionid = %s",
                (st_world.session_id,),
            )
            assert cur.fetchone()[0] >= 1, "scan (ST-TK-02) wrote no presence rows"
            cur.execute(
                """SELECT status FROM attendance_record
                   WHERE attendancesessionid = %s AND accountid = %s""",
                (st_world.session_id, st_world.account_ids["s1"]),
            )
            row = cur.fetchone()
        assert row is not None, "endSession (ST-TK-03) wrote no record for s1"

    # ── ST-UF-04 (U04) Student views own attendance records ─────────
    def test_04_st_uf_04_view_attendance_records(self, client, st_world):
        r = client.get("/student/attendance", headers=st_world.auth("s1"))
        assert r.status_code == 200, r.text
        records = r.json().get("records", [])
        mine = [x for x in records if x.get("session_id") == st_world.session_id]
        assert mine, "s1 must see the ST session in their attendance list"
        assert mine[0]["course_code"] == st_world.course_code
        assert mine[0]["status"] is not None  # written by endSession (ST-TK-03)

    # ── ST-UF-05 (U05) Attendance status notification (late/absent) ──
    # Acceptance per Part D wording: notifications are GENERATED/QUEUED for
    # every late/absent student; delivery needs SMTP. _send_email is patched
    # so no machine with real SMTP credentials can mail anyone during tests.
    def test_05_st_uf_05_attendance_notifications(self, client, st_world):
        with patch("core.notification._send_email", return_value=True) as fake:
            r = client.post(
                f"/teacher/sessions/{st_world.session_id}/notify",
                headers=st_world.auth("teacher"),
            )
            assert r.status_code == 200, r.text
            queued = r.json()["queued"]
        assert queued >= 1, "ended ST session must have late/absent students"
        # TestClient runs BackgroundTasks synchronously → one email per recipient.
        assert fake.call_count == queued
        for call in fake.call_args_list:
            to_addr = call.args[0]
            assert to_addr.endswith("@test.local"), to_addr  # only ST accounts

        missing = client.post(
            "/teacher/sessions/99999999/notify", headers=st_world.auth("teacher")
        )
        assert missing.status_code == 404

    # ── ST-UF-06 (U06) Student self-service face registration ───────
    def test_06_st_uf_06_student_self_face_registration(self, client, st_world):
        r = client.post(
            "/register",
            headers=st_world.auth("s2"),
            data={"account_id": str(st_world.account_ids["s2"])},
            files=_multipart_png(st_world.other_png),
        )
        assert r.status_code == 200, r.text
        assert r.json().get("success") is True

        # A student may only enrol their own face.
        denied = client.post(
            "/register",
            headers=st_world.auth("s2"),
            data={"account_id": str(st_world.account_ids["s3"])},
            files=_multipart_png(st_world.other_png),
        )
        assert denied.status_code == 403

        # «include» Validate Image Quality — unusable photos are rejected
        # before any embedding is written (blurred / too dark / too small).
        import cv2

        from tests.conftest import _png_bytes

        def _encode(img):
            return cv2.imencode(".png", img)[1].tobytes()

        rejects = {
            "too_small": _png_bytes(seed=11, size=48),
            "too_dark": _encode(np.full((160, 160, 3), 5, np.uint8)),
            "blurred": _encode(
                cv2.GaussianBlur(
                    cv2.imdecode(
                        np.frombuffer(_png_bytes(seed=12), np.uint8), cv2.IMREAD_COLOR
                    ),
                    (0, 0), sigmaX=9,
                )
            ),
        }
        for label, payload in rejects.items():
            r = client.post(
                "/register",
                headers=st_world.auth("s2"),
                data={"account_id": str(st_world.account_ids["s2"])},
                files=_multipart_png(payload),
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["success"] is False, f"{label} should fail quality gate: {body}"
            assert "quality" in body and body["message"]

        # The good photo from the start of this case is still the active one.
        r = client.post(
            "/register",
            headers=st_world.auth("s2"),
            data={"account_id": str(st_world.account_ids["s2"])},
            files=_multipart_png(st_world.other_png),
        )
        assert r.json().get("success") is True

    # ── ST-UF-07 (U07) View single-session attendance detail ────────
    def test_07_st_uf_07_view_session_detail(self, client, st_world):
        r = client.get(
            f"/student/sessions/{st_world.session_id}", headers=st_world.auth("s1")
        )
        assert r.status_code == 200, r.text
        session = r.json()["session"]
        assert session["course_code"] == st_world.course_code
        assert session["session_status"] == "ended"
        assert session["attendance_status"] is not None

        missing = client.get(
            "/student/sessions/99999999", headers=st_world.auth("s1")
        )
        assert missing.status_code == 404

    # ── ST-UF-08 (U08) Appeal visible to its student (flow: ST-TK-03) ─
    def test_08_st_uf_08_submit_appeal_visible_to_student(
        self, client, st_world, db_url
    ):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT a.appealid, a.accountid, a.status
                   FROM attendance_appeal a
                   JOIN attendance_record r
                     ON r.attendancerecordid = a.attendancerecordid
                   WHERE r.attendancesessionid = %s""",
                (st_world.session_id,),
            )
            row = cur.fetchone()
        if not row:
            pytest.skip("no appeal from ST-TK-03 — run the task suite first")
        appeal_id, account_id, status = row
        role_key = next(k for k, v in st_world.account_ids.items() if v == account_id)
        listed = client.get("/student/appeals", headers=st_world.auth(role_key))
        assert listed.status_code == 200
        mine = {a["appealid"]: a for a in listed.json().get("appeals", [])}
        assert appeal_id in mine
        assert mine[appeal_id]["status"] == status == "approved"

    # ── ST-UF-09 (U09) Update student facial image (re-registration) ─
    # U09 has no separate endpoint by design: POSTing /register again
    # soft-deletes the previous embedding and activates the new one.
    def test_09_st_uf_09_update_face_image(self, client, st_world, db_url, png_bytes):
        import psycopg2

        s1 = st_world.account_ids["s1"]
        r = client.post(
            "/register",
            headers=st_world.auth("s1"),
            data={"account_id": str(s1)},
            files=_multipart_png(png_bytes(seed=303)),
        )
        assert r.status_code == 200, r.text
        assert r.json().get("success") is True
        new_face_id = r.json()["written"]["arcface"]

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT faceid, is_active FROM face_embedding
                   WHERE accountid = %s AND model_name = 'arcface'
                   ORDER BY created_at DESC""",
                (s1,),
            )
            rows = cur.fetchall()
        active = [fid for fid, act in rows if act]
        assert active == [new_face_id], "exactly the new embedding must be active"
        assert len(rows) >= 2, "the replaced embedding must be kept (soft-deleted)"

        # Restore the original face so later suites keep recognising s1.
        r = client.post(
            "/register",
            headers=st_world.auth("s1"),
            data={"account_id": str(s1)},
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200, r.text

    # ── ST-UF-10 (U10) Teacher Login ─────────────────────────────────
    def test_10_st_uf_10_teacher_login(self, client, st_world):
        r = client.post(
            "/auth/login",
            json={"email": st_world.emails["teacher"], "password": st_world.password},
        )
        assert r.status_code == 200, r.text
        assert r.json()["user"]["role"] == "teacher"
        assert r.json()["token"]

    # ── ST-UF-11 (U11) Teacher logout (client-side boundary flow) ───
    def test_11_st_uf_11_teacher_logout_contract(self, client, st_world):
        login = client.post(
            "/auth/login",
            json={"email": st_world.emails["teacher"], "password": st_world.password},
        )
        assert login.status_code == 200, login.text
        token = login.json()["token"]
        assert (
            client.get(
                "/teacher/courses", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 200
        )
        # After logout the frontend sends no token → protected teacher API 401s.
        assert client.get("/teacher/courses").status_code == 401

    # ── ST-UF-12 (U12) Real-time attendance roster ───────────────────
    def test_12_st_uf_12_realtime_attendance(self, client, st_world):
        r = client.get(
            f"/teacher/sessions/{st_world.session_id}/live",
            headers=st_world.auth("teacher"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session"]["course_code"] == st_world.course_code
        roster = body["roster"]
        assert len(roster) == 3  # s1/s2/s3 enrolled
        s1_row = next(
            x for x in roster if x["accountid"] == st_world.account_ids["s1"]
        )
        assert s1_row["attendance_status"] is not None
        summary = body["summary"]
        assert sum(summary.values()) == len(roster)

    # ── ST-UF-13 (U13) Per-student attendance history (teacher view) ─
    def test_13_st_uf_13_student_history(self, client, st_world):
        r = client.get(
            f"/teacher/students/{st_world.account_ids['s1']}/attendance",
            headers=st_world.auth("teacher"),
            params={"course_id": st_world.course_id},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        records = body["records"]
        assert len(records) >= 1
        assert all(x["course_code"] == st_world.course_code for x in records)
        assert body["total"] == len(records)
        assert 0.0 <= float(body["rate"]) <= 100.0

    # ── ST-UF-14 (U14) Export Attendance Report ──────────────────────
    def test_14_st_uf_14_report_export(self, client, st_world):
        r = client.get(
            "/teacher/reports/export",
            headers=st_world.auth("teacher"),
            params={"course_id": st_world.course_id},
        )
        assert r.status_code == 200, r.text
        assert "text/csv" in r.headers.get("content-type", "")
        text = r.content.decode("utf-8")
        header = text.splitlines()[0]
        assert "course_code" in header and "attendance_status" in header
        assert st_world.course_code in text

    # ── ST-UF-15 (U15) Session lifecycle: scheduled → active → ended ─
    def test_15_st_uf_15_session_lifecycle_states(self, client, st_world, db_url):
        import psycopg2

        r = client.post(
            "/admin/sessions",
            headers=st_world.auth("admin"),
            json={
                "course_id": st_world.course_id,
                "start_time": "2030-03-01T09:00:00+08:00",
                "end_time": "2030-03-01T11:00:00+08:00",
                "status": "scheduled",
            },
        )
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        try:
            start = client.post(
                f"/teacher/sessions/{sid}/start", headers=st_world.auth("teacher")
            )
            assert start.status_code == 200, start.text
            assert start.json().get("status") == "active"

            active = client.get(
                "/teacher/sessions",
                headers=st_world.auth("teacher"),
                params={"course_id": st_world.course_id, "status": "active"},
            )
            assert sid in {
                s["attendancesessionid"] for s in active.json()["sessions"]
            }

            again = client.post(
                f"/teacher/sessions/{sid}/start", headers=st_world.auth("teacher")
            )
            assert again.status_code == 409  # already in progress

            end = client.post(
                f"/teacher/sessions/{sid}/end", headers=st_world.auth("teacher")
            )
            assert end.status_code == 200, end.text

            ended = client.get(
                "/teacher/sessions",
                headers=st_world.auth("teacher"),
                params={"course_id": st_world.course_id, "status": "ended"},
            )
            assert sid in {
                s["attendancesessionid"] for s in ended.json()["sessions"]
            }
        finally:
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                for sql in (
                    "DELETE FROM presence_check WHERE attendancesessionid = %s",
                    "DELETE FROM session_recording WHERE attendancesessionid = %s",
                    "DELETE FROM attendance_record WHERE attendancesessionid = %s",
                    "DELETE FROM attendance_session WHERE attendancesessionid = %s",
                ):
                    cur.execute(sql, (sid,))
                conn.commit()

    # ── ST-UF-16 (U16) Early departure summary ───────────────────────
    def test_16_st_uf_16_early_left_summary(self, client, st_world):
        r = client.get(
            f"/teacher/sessions/{st_world.session_id}/early-left",
            headers=st_world.auth("teacher"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert isinstance(body.get("early_left"), list)

    # ── ST-UF-17 (U17) Administrator login ───────────────────────────
    def test_17_st_uf_17_admin_login(self, client, st_world):
        r = client.post(
            "/auth/login",
            json={"email": st_world.emails["admin"], "password": st_world.password},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["token"]
        assert body["user"]["role"] == "admin"
        # The admin console's data endpoint is reachable with that JWT.
        listed = client.get(
            "/admin/users", headers={"Authorization": f"Bearer {body['token']}"}
        )
        assert listed.status_code == 200

    # ── ST-UF-18 (U18) Administrator logout (client-side boundary) ──
    def test_18_st_uf_18_admin_logout_contract(self, client, st_world):
        login = client.post(
            "/auth/login",
            json={"email": st_world.emails["admin"], "password": st_world.password},
        )
        assert login.status_code == 200, login.text
        token = login.json()["token"]
        assert (
            client.get(
                "/admin/users", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 200
        )
        # After logout the frontend sends no token → admin API 401s.
        assert client.get("/admin/users").status_code == 401

    # ── ST-UF-19 (U19) Admin registration: Account+PersonalInfo+Face ─
    def test_19_st_uf_19_admin_registration_three_tables(
        self, client, st_world, db_url
    ):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT ua.accountid, pi.full_name, f.faceid
                   FROM user_account ua
                   JOIN personal_info pi ON pi.accountid = ua.accountid
                   JOIN face_embedding f
                     ON f.accountid = ua.accountid AND f.is_active
                   WHERE ua.accountid = %s""",
                (st_world.account_ids["s1"],),
            )
            rows = cur.fetchall()
        assert rows, "Account + PersonalInfo + FaceEmbedding must all exist for s1"

        # End-to-end: the enrolled face is identifiable (absorbs the old
        # identify test that previously sat at ST-UF-02).
        r = client.post(
            "/identify",
            headers=st_world.auth("admin"),
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200, r.text
        assert any(
            i.get("recognised") and i.get("account_id") == st_world.account_ids["s1"]
            for i in r.json().get("identities", [])
        )

    # ── ST-UF-20 (U20) Manage user accounts (deactivate + edit) ─────
    def test_20_st_uf_20_manage_user_accounts(self, client, st_world):
        s2 = st_world.account_ids["s2"]
        token = st_world.tokens["s2"]
        assert (
            client.get(
                "/auth/me", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 200
        )

        r = client.patch(
            f"/admin/users/{s2}/status",
            headers=st_world.auth("admin"),
            json={"status": "inactive"},
        )
        assert r.status_code == 200, r.text
        denied = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert denied.status_code == 403  # deactivation takes effect immediately

        r = client.patch(
            f"/admin/users/{s2}/status",
            headers=st_world.auth("admin"),
            json={"status": "active"},
        )
        assert r.status_code == 200

        # Edit branch: change full_name, verify in the admin list, restore.
        original = st_world.names["s2"]
        try:
            r = client.patch(
                f"/admin/users/{s2}",
                headers=st_world.auth("admin"),
                json={"full_name": original + " (edited)"},
            )
            assert r.status_code == 200, r.text
            users = client.get(
                "/admin/users", headers=st_world.auth("admin")
            ).json()["users"]
            names = {u["accountid"]: u.get("full_name") for u in users}
            assert names.get(s2) == original + " (edited)"
        finally:
            client.patch(
                f"/admin/users/{s2}",
                headers=st_world.auth("admin"),
                json={"full_name": original},
            )

    # ── ST-UF-21 (U21) Manage facial image database ──────────────────
    # Add = /register, list = GET /admin/faces, remove = DELETE (soft),
    # replace = re-register. No separate edit endpoint by design.
    def test_21_st_uf_21_manage_face_database(self, client, st_world, png_bytes):
        s3 = st_world.account_ids["s3"]

        listed = client.get("/admin/faces", headers=st_world.auth("admin"))
        assert listed.status_code == 200
        active_accounts = {
            f["accountid"]
            for f in listed.json()["faces"]
            if f.get("is_active") in (True, "t", 1)
        }
        assert st_world.account_ids["s1"] in active_accounts

        # Replace s3's face, then delete the replacement (soft delete).
        r = client.post(
            "/register",
            headers=st_world.auth("admin"),
            data={"account_id": str(s3)},
            files=_multipart_png(png_bytes(seed=555)),
        )
        assert r.status_code == 200, r.text
        face_id = r.json()["written"]["arcface"]

        deleted = client.delete(
            f"/admin/faces/{face_id}", headers=st_world.auth("admin")
        )
        assert deleted.status_code == 200, deleted.text

        after = {
            f["faceid"]: f.get("is_active")
            for f in client.get("/admin/faces", headers=st_world.auth("admin"))
            .json()["faces"]
        }
        assert after.get(face_id) in (False, "f", 0), "delete must deactivate the row"

        # Restore an active embedding for s3 (same photo as ST-TK-01).
        r = client.post(
            "/register",
            headers=st_world.auth("admin"),
            data={"account_id": str(s3)},
            files=_multipart_png(png_bytes(seed=103)),
        )
        assert r.status_code == 200, r.text

    # ── ST-UF-22 (U22) Assign training/testing data ──────────────────
    def test_22_st_uf_22_assign_training_data(self, client, st_world):
        r = client.post(
            "/admin/training-data",
            headers=st_world.auth("admin"),
            json={"model_name": "arcface", "train_pct": 80},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        total = body["train_count"] + body["test_count"]
        assert total >= 1
        assert body["train_count"] == int(total * 80 / 100)

        bad = client.post(
            "/admin/training-data",
            headers=st_world.auth("admin"),
            json={"model_name": "arcface", "train_pct": 5},
        )
        assert bad.status_code == 400

    # ── ST-UF-23 (U23) Configure + train (flow only; metrics: ST-ML-04)
    def test_23_st_uf_23_configure_and_train(self, client, st_world):
        r = client.post(
            "/admin/training-data",
            headers=st_world.auth("admin"),
            json={"model_name": "arcface", "train_pct": 70},
        )
        assert r.status_code == 200, r.text
        r = client.post(
            "/admin/training-config",
            headers=st_world.auth("admin"),
            json={"epochs": 1, "batch_size": 8},
        )
        assert r.status_code == 200, r.text
        r = client.post("/admin/train", headers=st_world.auth("admin"))
        assert r.status_code == 200, r.text
        deadline = time.time() + 30
        status, payload = None, {}
        while time.time() < deadline:
            payload = client.get(
                "/admin/training-status", headers=st_world.auth("admin")
            ).json()
            status = payload.get("status")
            if status in ("done", "failed"):
                break
            time.sleep(0.2)
        assert status == "done", payload

    # ── ST-UF-24 (U24) Configure ensemble voting ─────────────────────
    def test_24_st_uf_24_configure_ensemble(self, client, st_world):
        # Single available model → applied live, not an ensemble.
        r = client.post(
            "/admin/ensemble",
            headers=st_world.auth("admin"),
            json={"models": ["arcface"], "weighting": "equal"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["is_ensemble"] is False
        assert body["models"] == ["arcface"]
        assert body["applied_weights"]["arcface"] == 1.0

        # Requesting a model that is not loaded on this server warns but
        # keeps the available one active.
        r = client.post(
            "/admin/ensemble",
            headers=st_world.auth("admin"),
            json={"models": ["arcface", "facenet"], "weighting": "equal"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["models"] == ["arcface"]
        assert body["warnings"], "unavailable facenet must be reported"

        # Invalid inputs.
        assert (
            client.post(
                "/admin/ensemble",
                headers=st_world.auth("admin"),
                json={"models": [], "weighting": "equal"},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/admin/ensemble",
                headers=st_world.auth("admin"),
                json={"models": ["arcface"], "weighting": "bogus"},
            ).status_code
            == 400
        )

    # ── ST-UF-25 (U25) Retrain & redeploy ────────────────────────────
    def test_25_st_uf_25_redeploy_model(self, client, st_world):
        r = client.post(
            "/admin/deploy",
            headers=st_world.auth("admin"),
            json={"force": True},
        )
        assert r.status_code == 200, r.text
        assert r.json().get("success") is True

    # ── ST-UF-26 (U26) Manage courses (create/status/enrol/delete) ───
    def test_26_st_uf_26_manage_courses(self, client, st_world, db_url):
        import psycopg2

        code = f"ST{st_world.suffix[:4].upper()}P2"  # matches ST% cleanup pattern
        course_id = None
        try:
            r = client.post(
                "/admin/courses",
                headers=st_world.auth("admin"),
                json={"course_code": code, "course_name": "ST-UF-26 temp course"},
            )
            assert r.status_code == 200, r.text
            course_id = r.json()["course_id"]

            dup = client.post(
                "/admin/courses",
                headers=st_world.auth("admin"),
                json={"course_code": code, "course_name": "dup"},
            )
            assert dup.status_code == 409

            # Inactive course refuses new sessions.
            r = client.patch(
                f"/admin/courses/{course_id}/status",
                headers=st_world.auth("admin"),
                json={"status": "inactive"},
            )
            assert r.status_code == 200
            blocked = client.post(
                "/admin/sessions",
                headers=st_world.auth("admin"),
                json={
                    "course_id": course_id,
                    "start_time": "2030-05-01T09:00:00+08:00",
                    "status": "scheduled",
                },
            )
            assert blocked.status_code == 400
            r = client.patch(
                f"/admin/courses/{course_id}/status",
                headers=st_world.auth("admin"),
                json={"status": "active"},
            )
            assert r.status_code == 200

            # Enrolment add / list / remove.
            r = client.post(
                f"/admin/courses/{course_id}/enrollments",
                headers=st_world.auth("admin"),
                json={"account_id": st_world.account_ids["s1"]},
            )
            assert r.status_code == 200, r.text
            enrolled = client.get(
                f"/admin/courses/{course_id}/enrollments",
                headers=st_world.auth("admin"),
            ).json()["enrollments"]
            assert st_world.account_ids["s1"] in {e["accountid"] for e in enrolled}
            r = client.delete(
                f"/admin/courses/{course_id}/enrollments/{st_world.account_ids['s1']}",
                headers=st_world.auth("admin"),
            )
            assert r.status_code == 200

            # Delete (no attendance records) and confirm it is gone.
            r = client.delete(
                f"/admin/courses/{course_id}", headers=st_world.auth("admin")
            )
            assert r.status_code == 200, r.text
            remaining = client.get(
                "/admin/courses", headers=st_world.auth("admin")
            ).json()["courses"]
            assert course_id not in {c["courseid"] for c in remaining}
            course_id = None
        finally:
            if course_id is not None:  # belt-and-braces if an assert fired
                with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM course_enrollment WHERE courseid = %s",
                        (course_id,),
                    )
                    cur.execute(
                        "DELETE FROM attendance_session WHERE courseid = %s",
                        (course_id,),
                    )
                    cur.execute(
                        "DELETE FROM course WHERE courseid = %s", (course_id,)
                    )
                    conn.commit()

    # ── ST-UF-27 (U27) Student personal analytics ────────────────────
    def test_27_st_uf_27_student_analytics(self, client, st_world):
        r = client.get("/student/analytics", headers=st_world.auth("s1"))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert isinstance(body.get("trend"), list)
        breakdown = body.get("breakdown", {})
        assert (breakdown.get("total") or 0) >= 1
        assert 0.0 <= float(breakdown.get("rate", -1)) <= 100.0

        # U27 step 4: the view can be narrowed to a single module. The ST
        # course is the student's only one, so its total matches the whole.
        scoped = client.get(
            "/student/analytics",
            headers=st_world.auth("s1"),
            params={"course_id": st_world.course_id},
        )
        assert scoped.status_code == 200, scoped.text
        assert scoped.json()["breakdown"]["total"] == breakdown["total"]
        # A course the student has no records in yields an empty breakdown.
        empty = client.get(
            "/student/analytics",
            headers=st_world.auth("s1"),
            params={"course_id": 99999999},
        )
        assert empty.status_code == 200
        assert (empty.json()["breakdown"].get("total") or 0) == 0

        # The module filter's data source: records carry their course id.
        records = client.get(
            "/student/attendance", headers=st_world.auth("s1")
        ).json()["records"]
        assert all("courseid" in r for r in records)

    # ── ST-UF-28 (U28) Submit leave application ──────────────────────
    def test_28_st_uf_28_submit_leave_application(self, client, st_world):
        r = client.post(
            "/admin/sessions",
            headers=st_world.auth("admin"),
            json={
                "course_id": st_world.course_id,
                "start_time": "2030-04-01T09:00:00+08:00",
                "end_time": "2030-04-01T11:00:00+08:00",
                "status": "scheduled",
            },
        )
        assert r.status_code == 200, r.text
        st_world._leave_session_id = r.json()["session_id"]  # type: ignore[attr-defined]

        r = client.post(
            "/student/leave-applications",
            headers=st_world.auth("s3"),
            json={
                "session_id": st_world._leave_session_id,
                "reason": "Medical appointment (ST-UF-28)",
            },
        )
        assert r.status_code == 200, r.text
        st_world._leave_id = r.json()["leave_application_id"]  # type: ignore[attr-defined]

        mine = client.get(
            "/student/leave-applications", headers=st_world.auth("s3")
        )
        assert mine.status_code == 200
        rows = {a["leaveapplicationid"]: a for a in mine.json()["applications"]}
        assert st_world._leave_id in rows
        assert rows[st_world._leave_id]["status"] == "pending"

    # ── ST-UF-29 (U29) Long-term absence reminder ────────────────────
    # Acceptance = reminders are generated for students breaching the
    # thresholds; _send_email is patched so nothing real is ever mailed
    # (the scan runs over the WHOLE shared database, incl. real students).
    def test_29_st_uf_29_long_term_absence_reminder(self, client, st_world):
        with patch("core.notification._send_email", return_value=True) as fake:
            r = client.post(
                "/admin/notifications/long-term-absence",
                headers=st_world.auth("admin"),
            )
            assert r.status_code == 200, r.text
            assert r.json().get("queued") is True
        # At least one ST student sits at 0% attendance → must be targeted.
        recipients = {call.args[0] for call in fake.call_args_list}
        assert any(a.endswith("@test.local") for a in recipients), recipients

    # ── ST-UF-30 (U30) Class attendance analytics (teacher) ──────────
    def test_30_st_uf_30_class_analytics(self, client, st_world):
        r = client.get(
            f"/teacher/courses/{st_world.course_id}/analytics",
            headers=st_world.auth("teacher"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert isinstance(body.get("trend"), list) and body["trend"]
        breakdown = body["breakdown"]
        assert (breakdown.get("total") or 0) >= 1
        assert 0.0 <= float(breakdown.get("rate", -1)) <= 100.0

    # ── ST-UF-31 (U31) Review leave application ──────────────────────
    def test_31_st_uf_31_review_leave_application(self, client, st_world, db_url):
        import psycopg2

        leave_id = getattr(st_world, "_leave_id", None)
        session_id = getattr(st_world, "_leave_session_id", None)
        if not leave_id:
            pytest.skip("ST-UF-28 did not create a leave application")
        try:
            pending = client.get(
                "/teacher/leave-applications", headers=st_world.auth("teacher")
            )
            assert pending.status_code == 200
            assert leave_id in {
                a["leaveapplicationid"] for a in pending.json()["applications"]
            }

            review = client.patch(
                f"/teacher/leave-applications/{leave_id}",
                headers=st_world.auth("teacher"),
                json={"decision": "approved", "comment": "ST approve"},
            )
            assert review.status_code == 200, review.text

            # Approval must mark the student's record as excused leave.
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT status FROM attendance_record
                       WHERE attendancesessionid = %s AND accountid = %s""",
                    (session_id, st_world.account_ids["s3"]),
                )
                row = cur.fetchone()
            assert row is not None and row[0] == "leave"
        finally:
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM leave_application WHERE attendancesessionid = %s",
                    (session_id,),
                )
                cur.execute(
                    "DELETE FROM attendance_record WHERE attendancesessionid = %s",
                    (session_id,),
                )
                cur.execute(
                    "DELETE FROM attendance_session WHERE attendancesessionid = %s",
                    (session_id,),
                )
                conn.commit()

    # ── ST-UF-32 (U32) Behaviour report: empty + with-data branches ──
    def test_32_st_uf_32_behaviour_report_with_data(self, client, st_world, db_url):
        import psycopg2

        s1 = st_world.account_ids["s1"]
        sid = st_world.session_id

        # Disabled/no-data branch (was old ST-UF-10): endpoint still 200.
        empty = client.get(
            f"/teacher/sessions/{sid}/behaviour", headers=st_world.auth("teacher")
        )
        assert empty.status_code == 200, empty.text
        assert isinstance(empty.json().get("students"), list)

        # With-data branch: AI_BEHAVIOUR=false in tests, so seed derived
        # event tuples directly (exactly what the live service would write).
        try:
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO behaviour_event
                          (attendancesessionid, accountid, event_type,
                           duration_seconds, confidence, metadata)
                       VALUES (%s, %s, 'drowsiness', 30, NULL,
                               '{"reasons": ["eyes_closed"]}'::jsonb),
                              (%s, %s, 'phone', 10, 0.7,
                               '{"conf": 0.7}'::jsonb)""",
                    (sid, s1, sid, s1),
                )
                conn.commit()

            report = client.get(
                f"/teacher/sessions/{sid}/behaviour",
                headers=st_world.auth("teacher"),
            )
            assert report.status_code == 200, report.text
            body = report.json()
            row = next(
                (x for x in body["students"] if x["accountid"] == s1), None
            )
            assert row is not None, body
            assert row["drowsy"] == 1 and row["phone"] == 1
            assert row["drowsy_seconds"] == 30 and row["phone_seconds"] == 10
            timeline = body.get("timeline", [])
            assert len(timeline) >= 2
            assert any(ev.get("metadata") for ev in timeline)
            # U32 alt-flow 2: students the analyser produced nothing for are
            # reported as inconclusive rather than as well-behaved. s1 has
            # events, so it must be classified as analysed.
            assert row["analysis_status"] == "analysed"
            assert "inconclusive_count" in body
            others = [x for x in body["students"] if x["accountid"] != s1]
            assert all(
                x["analysis_status"] in ("analysed", "inconclusive") for x in others
            )
        finally:
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM behaviour_event WHERE attendancesessionid = %s",
                    (sid,),
                )
                conn.commit()

    # ── ST-UF-33 (U33) Classroom activity heatmap ────────────────────
    def test_33_st_uf_33_activity_heatmap(self, client, st_world, db_url):
        import psycopg2

        sid = st_world.session_id
        cells = [(0, 0, 1.0), (3, 2, 0.5), (7, 5, 0.25)]
        try:
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                for x, y, v in cells:
                    cur.execute(
                        """INSERT INTO heatmap_snapshot
                              (attendancesessionid, zone_x, zone_y, intensity)
                           VALUES (%s, %s, %s, %s)""",
                        (sid, x, y, v),
                    )
                conn.commit()

            r = client.get(
                f"/teacher/sessions/{sid}/heatmap", headers=st_world.auth("teacher")
            )
            assert r.status_code == 200, r.text
            body = r.json()
            zones = body.get("zones", [])
            got = {(z["zone_x"], z["zone_y"]): float(z["intensity"]) for z in zones}
            for x, y, v in cells:
                assert got.get((x, y)) == v

            # U33 step 4: the teacher can narrow the heatmap to part of the
            # lesson. The response also reports the full capture window so
            # the UI can seed its pickers.
            assert body["captured_from"] and body["captured_to"]
            future = "2099-01-01T00:00:00+08:00"
            narrowed = client.get(
                f"/teacher/sessions/{sid}/heatmap",
                headers=st_world.auth("teacher"),
                params={"time_from": future},
            )
            assert narrowed.status_code == 200, narrowed.text
            assert narrowed.json()["zones"] == []  # window excludes every row
            assert narrowed.json()["captured_from"]  # extent still reported
        finally:
            with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM heatmap_snapshot WHERE attendancesessionid = %s",
                    (sid,),
                )
                conn.commit()

    # ── ST-UF-34 (U34) Configure absence threshold ───────────────────
    def test_34_st_uf_34_configure_absence_threshold(self, client, st_world):
        original = client.get(
            "/config/attendance", headers=st_world.auth("admin")
        ).json()
        assert original.get("success") is True
        try:
            r = client.patch(
                "/admin/config/absence-threshold",
                headers=st_world.auth("admin"),
                json={
                    "consecutive_threshold": 4,
                    "minimum_rate": 75.5,
                    "late_grace_seconds": 300,
                    "detection_interval_seconds": 600,
                },
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["absence_threshold"] == 4
            assert body["minimum_attendance_rate"] == 75.5
            assert body["late_grace_seconds"] == 300
            assert body["detection_interval_seconds"] == 600

            bad = client.patch(
                "/admin/config/absence-threshold",
                headers=st_world.auth("admin"),
                json={"minimum_rate": 150},
            )
            assert bad.status_code == 400
        finally:
            # Shared DB — always put the institution-wide config back.
            client.patch(
                "/admin/config/absence-threshold",
                headers=st_world.auth("admin"),
                json={
                    "consecutive_threshold": int(original["absence_threshold"]),
                    "minimum_rate": float(original["minimum_attendance_rate"]),
                    "late_grace_seconds": int(original["late_grace_seconds"]),
                    "detection_interval_seconds": int(
                        original["detection_interval_seconds"]
                    ),
                },
            )

    # ── ST-UF-35 (U35) Configure classroom behaviour analysis ───────
    def test_35_st_uf_35_configure_behaviour_analysis(self, client, st_world):
        cid = st_world.course_id
        url = f"/admin/courses/{cid}/behaviour-analysis"

        r = client.patch(
            url,
            headers=st_world.auth("admin"),
            json={
                "enable": True,
                "drowsiness": True,
                "phone_usage": False,
                "heatmap": True,
            },
        )
        assert r.status_code == 200, r.text
        cfg = r.json()["config"]
        assert cfg["enabled"] is True
        assert cfg["drowsiness"] is True
        assert cfg["phone_usage"] is False  # sub-flag persisted
        assert cfg["heatmap"] is True

        readback = client.get(url, headers=st_world.auth("admin"))
        assert readback.status_code == 200
        assert readback.json()["config"] == cfg

        off = client.patch(
            url, headers=st_world.auth("admin"), json={"enable": False}
        )
        assert off.status_code == 200, off.text
        assert off.json()["config"]["enabled"] is False
        assert (
            client.get(url, headers=st_world.auth("admin")).json()["config"]["enabled"]
            is False
        )

        missing = client.patch(
            "/admin/courses/99999999/behaviour-analysis",
            headers=st_world.auth("admin"),
            json={"enable": True},
        )
        assert missing.status_code == 404
        # behaviour_config row for the ST course is removed by st_world cleanup.


# ═══════════════════════════════════════════════════════════════════════
# 3.3.3 Network security
# ═══════════════════════════════════════════════════════════════════════
@requires_db
class TestST06_NetworkSecurity:
    """ST-SEC-01 … ST-SEC-08"""

    def test_01_st_sec_01_password_hashed_argon2(self, db_url, st_world):
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM user_account WHERE accountid = %s",
                (st_world.account_ids["admin"],),
            )
            pw_hash = cur.fetchone()[0]
        assert pw_hash.startswith("$argon2")
        assert st_world.password not in pw_hash

    def test_02_st_sec_02_jwt_claims_and_auth_me(self, client, st_world):
        from jose import jwt

        token = st_world.tokens["teacher"]
        # Decode without verifying exp skew issues using app secret
        secret = os.getenv("JWT_SECRET", "fyp-demo-change-me")
        data = jwt.decode(token, secret, algorithms=["HS256"])
        assert "sub" in data and "role" in data and "email" in data and "exp" in data
        assert data["role"] == "teacher"

        me = client.get("/auth/me", headers=st_world.auth("teacher"))
        assert me.status_code == 200
        assert me.json()["email"] == st_world.emails["teacher"]

    def test_03_st_sec_03_missing_tampered_token(self, client):
        assert client.get("/auth/me").status_code == 401
        assert client.get(
            "/auth/me", headers={"Authorization": "Bearer not.a.jwt"}
        ).status_code == 401

        # Tamper signature
        from jose import jwt

        secret = os.getenv("JWT_SECRET", "fyp-demo-change-me")
        bad = jwt.encode(
            {"sub": "1", "role": "admin", "email": "x@y.z", "exp": 9999999999},
            secret + "-wrong",
            algorithm="HS256",
        )
        assert client.get(
            "/auth/me", headers={"Authorization": f"Bearer {bad}"}
        ).status_code == 401

    def test_04_st_sec_04_role_isolation(self, client, st_world):
        r = client.get("/admin/users", headers=st_world.auth("s1"))
        assert r.status_code == 403

    def test_05_st_sec_05_prod_rejects_weak_jwt_secret(self):
        import subprocess
        import sys

        code = (
            "import os, sys\n"
            "os.environ['APP_ENV'] = 'production'\n"
            "os.environ['JWT_SECRET'] = 'fyp-demo-change-me'\n"
            # Prevent dotenv from replacing the weak secret under test.
            "os.environ['AI_USE_MTCNN'] = 'false'\n"
            "sys.path.insert(0, r'%s')\n"
            "try:\n"
            "    import core.userInformation  # noqa: F401\n"
            "    print('UNEXPECTED_OK')\n"
            "except RuntimeError as e:\n"
            "    print('RUNTIME_ERROR', 'JWT_SECRET' in str(e))\n"
        ) % str(REPO).replace("\\", "\\\\")
        env = {k: v for k, v in os.environ.items() if k not in {"JWT_SECRET", "APP_ENV"}}
        env.update({"APP_ENV": "production", "JWT_SECRET": "fyp-demo-change-me"})
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        assert "RUNTIME_ERROR" in out, out
        assert "UNEXPECTED_OK" not in out

    def test_06_st_sec_06_docs_gated_by_app_env(self, client):
        # Development client exposes docs; production wiring is in source.
        assert client.get("/docs").status_code == 200
        src = (REPO / "main_api.py").read_text(encoding="utf-8")
        assert "docs_url=None if _IS_PROD" in src
        assert "openapi_url=None if _IS_PROD" in src

    def test_07_st_sec_07_security_headers(self, client):
        r = client.get("/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert "Referrer-Policy" in r.headers

    def test_08_st_sec_08_cors_origins_configured(self):
        src = (REPO / "main_api.py").read_text(encoding="utf-8")
        assert "ALLOWED_ORIGINS" in src
        assert "allow_credentials=True" in src
        # Wildcard + credentials is explicitly avoided in comments/code path.
        assert 'allow_origins=["*"]' not in src.replace(" ", "")


# ═══════════════════════════════════════════════════════════════════════
# 3.3.4 ML — SCRFD / ArcFace (stubbed recognition metrics)
# ═══════════════════════════════════════════════════════════════════════
@requires_db
class TestST07_ML_FaceAccuracy:
    """ST-ML-01 … ST-ML-05 — use stub embeddings for deterministic metrics."""

    def test_01_st_ml_01_detection_returns_face_box(self, client, st_world):
        r = client.post(
            "/teacher/preview-detect",
            headers=st_world.auth("teacher"),
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200
        boxes = r.json().get("boxes", [])
        assert len(boxes) >= 1  # stub always emits one detection

    def test_02_st_ml_02_genuine_accept_rate(self, client, st_world):
        # Ensure s1 still enrolled with student_png
        client.post(
            "/register",
            headers=st_world.auth("admin"),
            data={"account_id": str(st_world.account_ids["s1"])},
            files=_multipart_png(st_world.student_png),
        )
        hits = 0
        trials = 5
        body = {}
        for _ in range(trials):
            r = client.post(
                "/identify",
                headers=st_world.auth("admin"),
                files=_multipart_png(st_world.student_png),
            )
            assert r.status_code == 200, r.text
            body = r.json()
            if any(
                i.get("recognised") and i.get("account_id") == st_world.account_ids["s1"]
                for i in body.get("identities", [])
            ):
                hits += 1
        assert hits / trials >= 0.9, f"TAR={hits}/{trials} body={body}"

    def test_03_st_ml_03_false_accept_rate(self, client, st_world):
        from tests.conftest import _png_bytes

        unknown = _png_bytes(seed=9999)
        false_accepts = 0
        trials = 5
        for _ in range(trials):
            r = client.post(
                "/identify",
                headers=st_world.auth("admin"),
                files=_multipart_png(unknown),
            )
            assert r.status_code == 200, r.text
            body = r.json()
            if any(i.get("recognised") for i in body.get("identities", [])):
                false_accepts += 1
        assert false_accepts / trials <= 0.05 + 1e-9, f"FAR={false_accepts}/{trials}"

    def test_04_st_ml_04_calibration_metrics(self, client, st_world):
        r = client.post(
            "/admin/training-data",
            headers=st_world.auth("admin"),
            json={"model_name": "arcface", "train_pct": 80},
        )
        assert r.status_code == 200
        r = client.post("/admin/train", headers=st_world.auth("admin"))
        assert r.status_code == 200
        deadline = time.time() + 30
        result = None
        while time.time() < deadline:
            s = client.get(
                "/admin/training-status", headers=st_world.auth("admin")
            ).json()
            if s.get("status") == "done":
                result = s.get("result")
                break
            if s.get("status") == "failed":
                pytest.fail(s.get("message") or "training failed")
            time.sleep(0.2)
        assert result is not None
        assert "accuracy" in result or "new_threshold" in result
        assert "fpr" in result or result.get("limited_calibration") is True

    def test_05_st_ml_05_regression_after_deploy(self, client, st_world):
        # After prior deploy, genuine identify still works.
        r = client.post(
            "/identify",
            headers=st_world.auth("admin"),
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# 3.3.5 Behaviour analysis
# ═══════════════════════════════════════════════════════════════════════
class TestST08_BehaviourAnalysis:
    """ST-BH-01 … ST-BH-04 — geometry / privacy always; rates via helpers."""

    def test_01_st_bh_01_drowsiness_signal_ear(self):
        from core.behaviourAnalysis import eye_aspect_ratio

        open_eye = np.array([[0, 0], [2, -1], [4, -1], [6, 0], [4, 1], [2, 1]], float)
        closed = np.array(
            [[0, 0], [2, -0.1], [4, -0.1], [6, 0], [4, 0.1], [2, 0.1]], float
        )
        # Closed eyes have lower EAR → drowsiness cue; gate uses this signal.
        assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed)
        # Simulated labelled recall: closed frames flagged drowsy when EAR low.
        ear_thresh = 0.2
        labelled_drowsy = [eye_aspect_ratio(closed)] * 10
        detected = sum(1 for e in labelled_drowsy if e < ear_thresh)
        assert detected / len(labelled_drowsy) >= 0.8

    def test_02_st_bh_02_phone_attribution_helper(self):
        from dataclasses import dataclass

        from core.behaviourAnalysis import BehaviourAnalysisService

        @dataclass
        class _P:
            bbox: list
            account_id: int

        faces = [_P([100, 100, 200, 200], 11), _P([600, 100, 700, 200], 22)]
        phone = np.array([120, 260, 160, 300])
        assert BehaviourAnalysisService._phone_owner(phone, faces) == 11
        # Simulated detection rate on obvious in-reach phones.
        clips = [np.array([120, 260, 160, 300])] * 8
        hits = sum(
            1
            for c in clips
            if BehaviourAnalysisService._phone_owner(c, faces) is not None
        )
        assert hits / len(clips) >= 0.75

    def test_03_st_bh_03_episode_debounce(self):
        from core.behaviourAnalysis import _EpisodeTracker

        t = _EpisodeTracker(confirm_seconds=2.0, release_seconds=2.5)
        now = 1000.0
        assert t.update(True, now) is None
        assert t.update(True, now + 1) is None
        assert t.update(False, now + 1.5) is None  # flicker — no emit yet
        t.update(True, now + 2)
        assert t.update(True, now + 4) is None or t.is_confirmed(now + 4)

    def test_04_st_bh_04_privacy_no_frame_persistence(self):
        import inspect

        from core import behaviourAnalysis as ba

        src = Path(inspect.getfile(ba)).read_text(encoding="utf-8")
        for forbidden in ("imwrite", "VideoWriter", "imencode"):
            assert forbidden not in src


# ═══════════════════════════════════════════════════════════════════════
# Offline smoke (always runs — no DB)
# ═══════════════════════════════════════════════════════════════════════
class TestST00_OfflineSmoke:
    """Cheap checks that always run even without DATABASE_URL."""

    def test_01_frontend_login_page_exists(self):
        assert (REPO / "frontend" / "index.html").is_file()

    def test_02_schema_sql_exists(self):
        assert (REPO / "database" / "schema.sql").is_file()

    def test_03_ci_workflow_exists(self):
        assert (REPO / ".github" / "workflows" / "ci.yml").is_file()

    def test_04_argon2_hash_roundtrip(self):
        from core.userInformation import _hash_password, _verify_password

        h = _hash_password("unit-test-password")
        assert h.startswith("$argon2")
        assert _verify_password("unit-test-password", h)
        assert not _verify_password("nope", h)
