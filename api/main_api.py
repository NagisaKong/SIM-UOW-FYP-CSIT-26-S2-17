"""FastAPI web entrypoint for the FYP-26-S2-17 attendance system.

Wraps `ai.attendance_pipeline.AttendancePipeline` (the full SCRFD + MTCNN +
ArcFace + FaceNet + GAN/CLAHE ensemble) behind HTTP endpoints so the
frontend can register students and run live identification.

Run:
    uvicorn api.main_api:app --host 127.0.0.1 --port 8000
    # or:
    python -m api.main_api
"""

from __future__ import annotations


import sys
from pathlib import Path

# Allow running this file directly (e.g. `python api/main_api.py` or from an
# IDE's Run button). Adds the repo root to sys.path so `ai.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


"StyleGAN portion"

from ai.training.calibrate import calibrate_threshold

"StyleGaN portion"


import contextlib
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
import psycopg2
import base64
import uuid
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.attendance_pipeline import AttendancePipeline
from ai.config import AIConfig

from api.auth import (
    CurrentUser,
    authenticate,
    create_token,
    get_current_user,
    require_role,
)


# ── Lifespan: load the pipeline once at startup ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AIConfig()
    print(cfg.log_summary())
    app.state.cfg = cfg
    app.state.pipeline = AttendancePipeline.from_env(cfg)
    yield


app = FastAPI(
    title="SIM-UOW Face Attendance System API",
    description="Face enrolment + identification + role-scoped endpoints for the demo frontend.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB helpers ────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _db():
    conn = psycopg2.connect(app.state.cfg.database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _dict_rows(cur) -> list[dict[str, Any]]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


async def _bytes_to_cv2(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片文件")
    return img


# ──────────────────────────────────────────────────────────────────────────
# Health + auth
# ──────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SIM-UOW Face Attendance API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "frontend": "http://127.0.0.1:5500",
    }


@app.get("/health")
def health():
    pipeline: AttendancePipeline = app.state.pipeline
    stores = {name: len(s) for name, s in pipeline.store_manager.stores.items()}
    return {"success": True, "stores": stores}


class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
def login(body: LoginBody):
    user = authenticate(app.state.cfg.database_url, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    token = create_token(user.account_id, user.role, user.email)
    return {
        "success": True,
        "token": token,
        "user": {
            "account_id": user.account_id,
            "role": user.role,
            "email": user.email,
            "full_name": user.full_name,
        },
    }


@app.get("/auth/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"account_id": user.account_id, "role": user.role, "email": user.email}


# ──────────────────────────────────────────────────────────────────────────
# Face endpoints (protected: user must be logged in)
# ──────────────────────────────────────────────────────────────────────────
@app.post("/register")
async def register_face(
    account_id: int = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Extract embeddings from the uploaded photo and write them to Supabase.

    Students may only re-register their own face; admins may register anyone.
    """
    if user.role != "admin" and user.account_id != account_id:
        raise HTTPException(status_code=403, detail="只能录入本人人脸")

    pipeline: AttendancePipeline = app.state.pipeline
    img = await _bytes_to_cv2(file)
    written = pipeline.enrol_student(account_id=account_id, images=[img])
    if not written:
        return {"success": False, "message": "未检测到人脸，请重新拍摄"}
    return {"success": True, "message": f"学生 {account_id} 录入成功", "written": written}


@app.post("/identify")
async def identify_face(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    pipeline: AttendancePipeline = app.state.pipeline
    img = await _bytes_to_cv2(file)
    result = pipeline.process_frame(img)

    identities = []
    for p in result.predictions:
        label = (
            p.full_name or p.student_id or (f"acc#{p.account_id}" if p.account_id else "Unknown")
            if p.recognised
            else "Unknown"
        )
        identities.append({
            "name": label,
            "confidence": round(float(p.score), 2),
            "recognised": p.recognised,
            "account_id": p.account_id,
            "student_id": p.student_id,
            "full_name": p.full_name,
            "score": round(float(p.score), 4),
            "det_score": round(float(p.det_score), 4),
            "bbox": list(p.bbox),
        })
    return {"success": True, "enhanced": result.enhanced, "identities": identities}


# ──────────────────────────────────────────────────────────────────────────
# Student endpoints
# ──────────────────────────────────────────────────────────────────────────
@app.post("/student/checkin")
async def student_checkin(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_role("student")),
):
    """Face-based attendance check-in.

    Flow:
      1. Run the ensemble pipeline on the uploaded frame.
      2. Require a recognised face matching the logged-in student
         (prevents someone else's face from marking you present).
      3. Find an active session in a course the student is enrolled in
         (prefers the most recently started one).
      4. Insert an attendance_record with status='present'. Re-check-ins
         are silently deduped by the UNIQUE constraint.
    """
    pipeline: AttendancePipeline = app.state.pipeline
    img = await _bytes_to_cv2(file)
    result = pipeline.process_frame(img)

    matched = next(
        (p for p in result.predictions if p.recognised and p.account_id == user.account_id),
        None,
    )
    if matched is None:
        faces = len(result.predictions)
        return {
            "success": False,
            "message": f"未能识别为本人（检测到 {faces} 张人脸，请正对摄像头）",
            "detections": faces,
        }

    with _db() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT s.attendancesessionid, c.course_code, c.course_name
            FROM attendance_session s
            JOIN course c ON c.courseid = s.courseid
            JOIN course_enrollment e
              ON e.courseid = s.courseid AND e.accountid = %s AND e.status = 'active'
            WHERE s.status = 'active'
              AND NOW() BETWEEN s.start_time AND COALESCE(s.end_time, NOW() + INTERVAL '1 day')
            ORDER BY s.start_time DESC
            LIMIT 1
            """,
            (user.account_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"success": False, "message": "当前没有进行中的课程"}
        session_id, course_code, course_name = row

        cur.execute(
            """
            INSERT INTO attendance_record (attendancesessionid, accountid, status)
            VALUES (%s, %s, 'present')
            ON CONFLICT (attendancesessionid, accountid) DO NOTHING
            RETURNING attendancerecordid
            """,
            (session_id, user.account_id),
        )
        new_row = cur.fetchone()
        already = new_row is None

    return {
        "success": True,
        "already_checked_in": already,
        "message": "已签到（重复打卡已忽略）" if already else f"签到成功：{course_code} {course_name}",
        "session_id": session_id,
        "course_code": course_code,
        "course_name": course_name,
        "confidence": round(float(matched.score), 3),
    }


@app.get("/student/attendance")
def student_attendance(user: CurrentUser = Depends(require_role("student"))):
    sql = """
        SELECT r.attendancerecordid AS record_id,
               r.attendancesessionid AS session_id,
               s.start_time, s.end_time,
               c.course_code, c.course_name,
               r.status, r.marked_at
        FROM attendance_record r
        JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
        JOIN course c ON c.courseid = s.courseid
        WHERE r.accountid = %s
        ORDER BY s.start_time DESC
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql, (user.account_id,))
        return {"success": True, "records": _dict_rows(cur)}


@app.get("/student/sessions/{session_id}")
def student_session_detail(
    session_id: int, user: CurrentUser = Depends(require_role("student"))
):
    sql = """
        SELECT s.attendancesessionid AS session_id,
               s.start_time, s.end_time, s.status AS session_status,
               c.course_code, c.course_name,
               r.attendancerecordid AS record_id, r.status AS attendance_status, r.marked_at
        FROM attendance_session s
        JOIN course c ON c.courseid = s.courseid
        LEFT JOIN attendance_record r
          ON r.attendancesessionid = s.attendancesessionid AND r.accountid = %s
        WHERE s.attendancesessionid = %s
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql, (user.account_id, session_id))
        rows = _dict_rows(cur)
    if not rows:
        raise HTTPException(404, "Session not found")
    return {"success": True, "session": rows[0]}


class AppealBody(BaseModel):
    record_id: int
    reason: str


@app.post("/student/appeals")
def student_create_appeal(
    body: AppealBody, user: CurrentUser = Depends(require_role("student"))
):
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT accountid FROM attendance_record WHERE attendancerecordid = %s",
            (body.record_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Record not found")
        if row[0] != user.account_id:
            raise HTTPException(403, "不能对他人记录申诉")
        cur.execute(
            """
            INSERT INTO attendance_appeal (attendancerecordid, accountid, reason)
            VALUES (%s, %s, %s) RETURNING appealid
            """,
            (body.record_id, user.account_id, body.reason),
        )
        appeal_id = cur.fetchone()[0]
    return {"success": True, "appeal_id": appeal_id}


@app.get("/student/appeals")
def student_list_appeals(user: CurrentUser = Depends(require_role("student"))):
    sql = """
        SELECT a.appealid, a.attendancerecordid, a.reason, a.status,
               a.created_at, a.reviewed_at
        FROM attendance_appeal a
        WHERE a.accountid = %s
        ORDER BY a.created_at DESC
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql, (user.account_id,))
        return {"success": True, "appeals": _dict_rows(cur)}


# ──────────────────────────────────────────────────────────────────────────
# Admin endpoints
# ──────────────────────────────────────────────────────────────────────────
@app.get("/admin/users")
def admin_list_users(user: CurrentUser = Depends(require_role("admin"))):
    sql = """
        SELECT ua.accountid, ua.email, up.role, up.status,
               pi.full_name, pi.student_id, pi.staff_id, ua.created_at
        FROM user_account ua
        JOIN user_profiles up ON up.profileid = ua.profileid
        LEFT JOIN personal_info pi ON pi.accountid = ua.accountid
        ORDER BY ua.accountid
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "users": _dict_rows(cur)}


class CreateUserBody(BaseModel):
    email: str
    password: str
    role: str  # student | admin
    full_name: str
    student_id: str | None = None
    staff_id: str | None = None


@app.post("/admin/users")
def admin_create_user(
    body: CreateUserBody, user: CurrentUser = Depends(require_role("admin"))
):
    from api.auth import hash_password

    if body.role not in ("student", "admin", "teacher"):
        raise HTTPException(400, "role 非法")
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT profileid FROM user_profiles WHERE role = %s LIMIT 1", (body.role,)
        )
        prow = cur.fetchone()
        if not prow:
            raise HTTPException(500, f"未找到角色 profile: {body.role}")
        profile_id = prow[0]

        cur.execute(
            """
            INSERT INTO user_account (profileid, email, password_hash)
            VALUES (%s, %s, %s) RETURNING accountid
            """,
            (profile_id, body.email, hash_password(body.password)),
        )
        account_id = cur.fetchone()[0]

        student_id = body.student_id or (None if body.role != "student" else f"S{account_id:05d}")
        staff_id = body.staff_id or (None if body.role == "student" else f"A{account_id:05d}")
        cur.execute(
            """
            INSERT INTO personal_info (accountid, full_name, student_id, staff_id)
            VALUES (%s, %s, %s, %s)
            """,
            (account_id, body.full_name, student_id, staff_id),
        )
    return {"success": True, "account_id": account_id}


class StatusBody(BaseModel):
    status: str  # active | inactive


@app.patch("/admin/users/{account_id}/status")
def admin_set_user_status(
    account_id: int,
    body: StatusBody,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("active", "inactive"):
        raise HTTPException(400, "status 非法")
    with _db() as c, c.cursor() as cur:
        cur.execute(
            """
            UPDATE user_profiles SET status = %s
            WHERE profileid = (SELECT profileid FROM user_account WHERE accountid = %s)
            """,
            (body.status, account_id),
        )
    return {"success": True}


@app.get("/admin/faces")
def admin_list_faces(user: CurrentUser = Depends(require_role("admin"))):
    sql = """
        SELECT f.faceid, f.accountid, pi.full_name, pi.student_id,
               f.model_name, f.model_version, f.dimension, f.is_active, f.created_at
        FROM face_embedding f
        LEFT JOIN personal_info pi ON pi.accountid = f.accountid
        ORDER BY f.accountid, f.model_name, f.created_at DESC
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "faces": _dict_rows(cur)}


@app.delete("/admin/faces/{face_id}")
def admin_delete_face(face_id: int, user: CurrentUser = Depends(require_role("admin"))):
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE face_embedding SET is_active = FALSE WHERE faceid = %s", (face_id,)
        )
    # Reload pipeline's in-memory store so change takes effect immediately.
    try:
        app.state.pipeline.store_manager.reload()
    except Exception:
        pass
    return {"success": True}


@app.get("/admin/attendance")
def admin_list_attendance(user: CurrentUser = Depends(require_role("admin"))):
    sql = """
        SELECT r.attendancerecordid, r.attendancesessionid,
               s.start_time, c.course_code, c.course_name,
               r.accountid, pi.full_name, pi.student_id,
               r.status, r.marked_at
        FROM attendance_record r
        JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
        JOIN course c ON c.courseid = s.courseid
        LEFT JOIN personal_info pi ON pi.accountid = r.accountid
        ORDER BY s.start_time DESC, r.marked_at DESC
        LIMIT 500
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "records": _dict_rows(cur)}


@app.get("/admin/appeals")
def admin_list_appeals(user: CurrentUser = Depends(require_role("admin"))):
    sql = """
        SELECT a.appealid, a.attendancerecordid, a.accountid,
               pi.full_name, pi.student_id,
               a.reason, a.status, a.created_at, a.reviewed_at
        FROM attendance_appeal a
        LEFT JOIN personal_info pi ON pi.accountid = a.accountid
        ORDER BY a.created_at DESC
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "appeals": _dict_rows(cur)}


class AppealReviewBody(BaseModel):
    status: str  # approved | rejected


@app.patch("/admin/appeals/{appeal_id}")
def admin_review_appeal(
    appeal_id: int,
    body: AppealReviewBody,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("approved", "rejected"):
        raise HTTPException(400, "status 非法")
    with _db() as c, c.cursor() as cur:
        cur.execute(
            """
            UPDATE attendance_appeal
            SET status = %s, reviewed_by = %s, reviewed_at = NOW()
            WHERE appealid = %s
            """,
            (body.status, user.account_id, appeal_id),
        )
    return {"success": True}


# ──────────────────────────────────────────────────────────────────────────
# Course management (U26)
# ──────────────────────────────────────────────────────────────────────────
@app.get("/admin/courses")
def admin_list_courses(user: CurrentUser = Depends(require_role("admin"))):
    sql = """
        SELECT c.courseid, c.course_code, c.course_name,
               COALESCE(c.status, 'active') AS status,
               (SELECT COUNT(*) FROM attendance_session s
                  WHERE s.courseid = c.courseid AND s.status = 'active') AS active_sessions
        FROM course c
        ORDER BY c.courseid
    """
    with _db() as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "courses": _dict_rows(cur)}


class CourseBody(BaseModel):
    course_code: str
    course_name: str


@app.post("/admin/courses")
def admin_create_course(
    body: CourseBody, user: CurrentUser = Depends(require_role("admin"))
):
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM course WHERE course_code = %s", (body.course_code,)
        )
        if cur.fetchone():
            raise HTTPException(409, f"课程代码 {body.course_code} 已存在")
        cur.execute(
            "INSERT INTO course (course_code, course_name) VALUES (%s, %s) RETURNING courseid",
            (body.course_code, body.course_name),
        )
        course_id = cur.fetchone()[0]
    return {"success": True, "course_id": course_id}


class CourseStatusBody(BaseModel):
    status: str  # active | inactive


@app.patch("/admin/courses/{course_id}/status")
def admin_set_course_status(
    course_id: int,
    body: CourseStatusBody,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("active", "inactive"):
        raise HTTPException(400, "status 非法")
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE course SET status = %s WHERE courseid = %s",
            (body.status, course_id),
        )
    return {"success": True}


# ──────────────────────────────────────────────────────────────────────────
# Attendance session scheduling (admin)
# ──────────────────────────────────────────────────────────────────────────
@app.get("/admin/sessions")
def admin_list_sessions(
    course_id: int | None = None,
    user: CurrentUser = Depends(require_role("admin")),
):
    sql = """
        SELECT s.attendancesessionid, s.courseid, c.course_code, c.course_name,
               s.start_time, s.end_time, s.status
        FROM attendance_session s
        JOIN course c ON c.courseid = s.courseid
        {where}
        ORDER BY s.start_time DESC
        LIMIT 500
    """
    where = "WHERE s.courseid = %s" if course_id else ""
    params = (course_id,) if course_id else ()
    with _db() as c, c.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        return {"success": True, "sessions": _dict_rows(cur)}


class SessionBody(BaseModel):
    course_id: int
    start_time: str  # ISO 8601, e.g. "2026-05-10T09:00:00+08:00"
    end_time: str | None = None
    status: str = "scheduled"  # scheduled | active | ended | cancelled


@app.post("/admin/sessions")
def admin_create_session(
    body: SessionBody, user: CurrentUser = Depends(require_role("admin"))
):
    if body.status not in ("scheduled", "active", "ended", "cancelled"):
        raise HTTPException(400, "status 非法")
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(status,'active') FROM course WHERE courseid = %s",
            (body.course_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "课程不存在")
        if row[0] == "inactive":
            raise HTTPException(400, "课程已停用，无法排课")
        cur.execute(
            """
            INSERT INTO attendance_session (courseid, start_time, end_time, status)
            VALUES (%s, %s, %s, %s)
            RETURNING attendancesessionid
            """,
            (body.course_id, body.start_time, body.end_time, body.status),
        )
        sid = cur.fetchone()[0]
    return {"success": True, "session_id": sid}


class SessionPatchBody(BaseModel):
    start_time: str | None = None
    end_time: str | None = None
    status: str | None = None


@app.patch("/admin/sessions/{session_id}")
def admin_update_session(
    session_id: int,
    body: SessionPatchBody,
    user: CurrentUser = Depends(require_role("admin")),
):
    fields, params = [], []
    if body.start_time is not None:
        fields.append("start_time = %s")
        params.append(body.start_time)
    if body.end_time is not None:
        fields.append("end_time = %s")
        params.append(body.end_time)
    if body.status is not None:
        if body.status not in ("scheduled", "active", "ended", "cancelled"):
            raise HTTPException(400, "status 非法")
        fields.append("status = %s")
        params.append(body.status)
    if not fields:
        raise HTTPException(400, "无可更新字段")
    params.append(session_id)
    with _db() as c, c.cursor() as cur:
        cur.execute(
            f"UPDATE attendance_session SET {', '.join(fields)} WHERE attendancesessionid = %s",
            params,
        )
    return {"success": True}


@app.delete("/admin/sessions/{session_id}")
def admin_delete_session(
    session_id: int, user: CurrentUser = Depends(require_role("admin"))
):
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance_record WHERE attendancesessionid = %s LIMIT 1",
            (session_id,),
        )
        if cur.fetchone():
            raise HTTPException(409, "该课时已有签到记录，无法删除")
        cur.execute(
            "DELETE FROM attendance_session WHERE attendancesessionid = %s",
            (session_id,),
        )
    return {"success": True}


# ──────────────────────────────────────────────────────────────────────────
# AI Model Governance (U22-U25)
# ──────────────────────────────────────────────────────────────────────────
class TrainingDataBody(BaseModel):
    train_pct: int
    model_name: str


@app.post("/admin/training-data")
def admin_assign_training_data(
    body: TrainingDataBody, user: CurrentUser = Depends(require_role("admin"))
):
    if not (10 <= body.train_pct <= 95):
        raise HTTPException(400, "train_pct 必须在 10-95 之间")
    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM face_embedding WHERE is_active = TRUE AND model_name = %s",
            (body.model_name,),
        )
        total = cur.fetchone()[0]
    if total == 0:
        raise HTTPException(400, "训练集为空，无法分配数据")
    train_count = int(total * body.train_pct / 100)
    test_count = total - train_count
    return {
        "success": True,
        "model_name": body.model_name,
        "train_count": train_count,
        "test_count": test_count,
    }


class EnsembleBody(BaseModel):
    use_arcface: bool
    use_facenet: bool
    weighting: str  # equal | confidence


@app.post("/admin/ensemble")
def admin_configure_ensemble(
    body: EnsembleBody, user: CurrentUser = Depends(require_role("admin"))
):
    selected = sum([body.use_arcface, body.use_facenet])
    if selected < 2:
        raise HTTPException(400, "Ensemble 至少需要两个模型 (U24)")
    if body.weighting not in ("equal", "confidence"):
        raise HTTPException(400, "weighting 非法")
    return {
        "success": True,
        "models": [
            m for m, on in [("arcface", body.use_arcface), ("facenet", body.use_facenet)] if on
        ],
        "weighting": body.weighting,
    }


@app.post("/admin/retrain")
async def admin_retrain_model(
    force: bool = False, user: CurrentUser = Depends(require_role("admin"))
):
    """Retrain & redeploy active model. Warns if new threshold deviates strongly
    from the previous one (U25 alternative flow #2)."""
    from ai.training.synthetic_gen import SyntheticDataGenerator

    with _db() as c, c.cursor() as cur:
        cur.execute(
            "SELECT similarity_threshold FROM model_configs WHERE model_name = %s",
            ("arcface_ensemble",),
        )
        row = cur.fetchone()
        old_threshold = float(row[0]) if row else None

    generator = SyntheticDataGenerator()
    synthetic_data, labels = generator.prepare_calibration_set()
    new_threshold = float(calibrate_threshold(synthetic_data, labels))

    if not force and old_threshold is not None and abs(new_threshold - old_threshold) > 0.15:
        return {
            "success": False,
            "warning": (
                f"New threshold {new_threshold:.3f} differs significantly from "
                f"current {old_threshold:.3f}; review before deploying."
            ),
            "new_threshold": new_threshold,
            "old_threshold": old_threshold,
        }

    with _db() as c, c.cursor() as cur:
        cur.execute(
            """
            UPDATE model_configs
            SET similarity_threshold = %s, updated_at = NOW(), updated_by = %s
            WHERE model_name = %s
            """,
            (new_threshold, user.account_id, "arcface_ensemble"),
        )
    return {"success": True, "new_threshold": new_threshold, "old_threshold": old_threshold}


@app.post("/admin/recalibrate")
async def recalibrate_models():
    #Initialize StyleGAN generator
    from ai.training.synthetic_gen import SyntheticDataGenerator
    generator = SyntheticDataGenerator()
    
    #Generate the data
    synthetic_data, labels = generator.prepare_calibration_set()
    
    #Calculate the new threshold
    new_threshold = calibrate_threshold(synthetic_data, labels) 
    
    #Save to Supabase database using EXISTING db helper
    with _db() as c, c.cursor() as cur:
        cur.execute(
            """
            UPDATE model_configs 
            SET similarity_threshold = %s, updated_at = NOW()
            WHERE model_name = %s
            """,
            (float(new_threshold), 'arcface_ensemble')
        )
    
    return {"status": "success", "new_threshold": float(new_threshold)}


#this is a global memory to keep track of webcam scan sessions. Each session has a unique tracking_id and stores counts of how many times each student was seen during the scan.
if not hasattr(app.state, "webcam_sessions"):
    app.state.webcam_sessions = {}

@app.post("/admin/start-webcam-scan")
def start_webcam_scan(user: CurrentUser = Depends(require_role("admin"))):
    """Initializes a new tracking dictionary for the incoming photo stream."""
    tracking_id = str(uuid.uuid4())
    app.state.webcam_sessions[tracking_id] = {}
    return {"success": True, "tracking_id": tracking_id}

class FrameBody(BaseModel):
    image: str
    tracking_id: str

@app.post("/admin/process-webcam-frame")
def process_webcam_frame(body: FrameBody, user: CurrentUser = Depends(require_role("admin"))):
    """Receives a single base64 snapshot from the webcam and counts the identities."""
    pipeline: AttendancePipeline = app.state.pipeline
    
    # 1. Decode the base64 Javascript image into an OpenCV numpy array
    encoded_data = body.image.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Run the image through your AI Pipeline
    result = pipeline.process_frame(img)
    
    # 3. Add +1 to the counter for every recognised student
    tracking_dict = app.state.webcam_sessions.get(body.tracking_id)
    if tracking_dict is not None:
        for p in result.predictions:
            if p.recognised and p.account_id:
                tracking_dict[p.account_id] = tracking_dict.get(p.account_id, 0) + 1
                
    return {"success": True, "faces_found": len(result.predictions)}

@app.post("/admin/finalize-webcam-scan")
def finalize_webcam_scan(tracking_id: str, total_scans: int, user: CurrentUser = Depends(require_role("admin"))):
    """Applies the 70% rule based on the final counts."""
    # Retrieve and delete the memory dictionary to free up RAM
    tracking_dict = app.state.webcam_sessions.pop(tracking_id, {})
    
    present_count = 0
    absent_count = 0
    
    with _db() as c, c.cursor() as cur:
        # Get all students currently marked 'present' or 'late' in an active class
        cur.execute("""
            SELECT r.attendancerecordid, r.accountid
            FROM attendance_record r
            JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
            WHERE s.status = 'active'
        """)
        active_records = cur.fetchall()
        
        for record in active_records:
            record_id = record['attendancerecordid'] if isinstance(record, dict) else record[0]
            account_id = record['accountid'] if isinstance(record, dict) else record[1]
            
            # The 70% Logic Math
            times_seen = tracking_dict.get(account_id, 0)
            presence_percentage = (times_seen / total_scans) * 100 if total_scans > 0 else 0
            
            if presence_percentage >= 70.0:
                new_status = 'present'
                present_count += 1
            else:
                new_status = 'absent'
                absent_count += 1
                
            cur.execute("""
                UPDATE attendance_record
                SET status = %s, marked_at = NOW()
                WHERE attendancerecordid = %s
            """, (new_status, record_id))
            
    return {
        "success": True, 
        "present_count": present_count, 
        "absent_count": absent_count
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
