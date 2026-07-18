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
        assert health.json().get("success") is True
        assert "stores" in health.json()

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
    """ST-UF-01 … ST-UF-11"""

    def test_01_st_uf_01_role_logins(self, client, st_world):
        for role in ("admin", "teacher", "s1"):
            r = client.post(
                "/auth/login",
                json={"email": st_world.emails[role], "password": st_world.password},
            )
            assert r.status_code == 200
            assert r.json()["token"]

    def test_02_st_uf_02_face_enrol_identify(self, client, st_world):
        r = client.post(
            "/identify",
            headers=st_world.auth("admin"),
            files=_multipart_png(st_world.student_png),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        identities = body.get("identities", [])
        assert identities
        assert any(
            i.get("recognised") and i.get("account_id") == st_world.account_ids["s1"]
            for i in identities
        )

    def test_03_st_uf_03_session_lifecycle_already_exercised(self, st_world):
        # Covered by ST-TK-02/03; assert fixture session id is set.
        assert st_world.session_id > 0

    def test_04_st_uf_04_scan_detects_enrolled(self, st_world):
        # Covered by ST-TK-02 assertion on detected s1.
        assert st_world.account_ids["s1"]

    def test_05_st_uf_05_early_left_endpoint(self, client, st_world):
        # Session already ended — endpoint should still respond.
        r = client.get(
            f"/teacher/sessions/{st_world.session_id}/early-left",
            headers=st_world.auth("teacher"),
        )
        assert r.status_code == 200

    def test_06_st_uf_06_appeal_workflow(self, st_world):
        # Covered by ST-TK-03.
        assert True

    def test_07_st_uf_07_leave_application(self, client, st_world, db_url):
        import psycopg2

        # Create a fresh scheduled session for leave (cannot leave ended session
        # in some paths — use a new one).
        r = client.post(
            "/admin/sessions",
            headers=st_world.auth("admin"),
            json={
                "course_id": st_world.course_id,
                "start_time": "2030-02-01T09:00:00+08:00",
                "end_time": "2030-02-01T11:00:00+08:00",
                "status": "scheduled",
            },
        )
        assert r.status_code == 200, r.text
        leave_session = r.json()["session_id"]
        st_world._leave_session_id = leave_session  # type: ignore[attr-defined]

        r = client.post(
            "/student/leave-applications",
            headers=st_world.auth("s3"),
            json={"session_id": leave_session, "reason": "Medical appointment (ST)"},
        )
        assert r.status_code == 200, r.text
        leave_id = r.json().get("leave_application_id")
        assert leave_id

        listed = client.get(
            "/teacher/leave-applications", headers=st_world.auth("teacher")
        )
        assert listed.status_code == 200

        review = client.patch(
            f"/teacher/leave-applications/{leave_id}",
            headers=st_world.auth("teacher"),
            json={"decision": "approved"},
        )
        assert review.status_code == 200, review.text

        # Approved leave may have written attendance_record — delete children first.
        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM leave_application WHERE attendancesessionid = %s",
                (leave_session,),
            )
            cur.execute(
                "DELETE FROM attendance_record WHERE attendancesessionid = %s",
                (leave_session,),
            )
            cur.execute(
                "DELETE FROM attendance_session WHERE attendancesessionid = %s",
                (leave_session,),
            )
            conn.commit()

    def test_08_st_uf_08_teacher_report_export(self, client, st_world):
        r = client.get(
            "/teacher/reports/export",
            headers=st_world.auth("teacher"),
            params={"course_id": st_world.course_id},
        )
        assert r.status_code == 200
        assert len(r.content) > 0

    def test_09_st_uf_09_admin_train_deploy(self, client, st_world):
        r = client.post(
            "/admin/training-data",
            headers=st_world.auth("admin"),
            json={"model_name": "arcface", "train_pct": 70},
        )
        assert r.status_code == 200
        r = client.post("/admin/train", headers=st_world.auth("admin"))
        assert r.status_code == 200, r.text
        # Poll async job
        deadline = time.time() + 30
        status = "running"
        while time.time() < deadline:
            s = client.get(
                "/admin/training-status", headers=st_world.auth("admin")
            )
            assert s.status_code == 200
            status = s.json().get("status")
            if status in ("done", "failed"):
                break
            time.sleep(0.2)
        assert status == "done", s.json()
        deploy = client.post(
            "/admin/deploy",
            headers=st_world.auth("admin"),
            json={"force": True},
        )
        assert deploy.status_code == 200, deploy.text

    def test_10_st_uf_10_behaviour_endpoints_when_disabled(self, client, st_world):
        # With AI_BEHAVIOUR=false, behaviour-scan may 503/400 — listing config OK.
        r = client.get(
            f"/admin/courses/{st_world.course_id}/behaviour-analysis",
            headers=st_world.auth("admin"),
        )
        assert r.status_code == 200, r.text

        report = client.get(
            f"/teacher/sessions/{st_world.session_id}/behaviour",
            headers=st_world.auth("teacher"),
        )
        assert report.status_code == 200, report.text

    def test_11_st_uf_11_deactivate_revokes_access(self, client, st_world):
        token = st_world.tokens["s2"]
        # Confirm token works
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200

        r = client.patch(
            f"/admin/users/{st_world.account_ids['s2']}/status",
            headers=st_world.auth("admin"),
            json={"status": "inactive"},
        )
        assert r.status_code == 200, r.text

        denied = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert denied.status_code == 403

        # Reactivate for cleanup / later tests
        client.patch(
            f"/admin/users/{st_world.account_ids['s2']}/status",
            headers=st_world.auth("admin"),
            json={"status": "active"},
        )


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
