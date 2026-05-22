"""Class 5: attendanceSession

Covers UC:
  U07 View Session Attendance Details (viewSessionDetail)
  U15 Start / End Attendance Session  (startSession / endSession)
  U26 Manage Courses                  (admin CRUD on courses + enrolments + session scheduling)

Attributes: sessionDetail* (start_time, end_time, status, courseid, …)
"""

from __future__ import annotations

import contextlib
from typing import Any

import psycopg2
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from core.notification import send_late_absent_emails
from core.userInformation import CurrentUser, require_role


@contextlib.contextmanager
def _db(database_url: str):
    conn = psycopg2.connect(database_url)
    try:
        yield conn; conn.commit()
    except Exception:
        conn.rollback(); raise
    finally:
        conn.close()


def _dict_rows(cur) -> list[dict[str, Any]]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


class AttendanceSession:
    """Session entity (start/end + view detail + course scheduling)."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    # ── U07 viewSessionDetail (student perspective) ──────────────────
    def viewSessionDetail(self, session_id: int, account_id: int) -> dict[str, Any]:
        sql = """
            SELECT s.attendancesessionid AS session_id,
                   s.start_time, s.end_time, s.status AS session_status,
                   c.course_code, c.course_name,
                   r.attendancerecordid AS record_id,
                   r.status AS attendance_status, r.marked_at
            FROM attendance_session s
            JOIN course c ON c.courseid = s.courseid
            LEFT JOIN attendance_record r
              ON r.attendancesessionid = s.attendancesessionid AND r.accountid = %s
            WHERE s.attendancesessionid = %s
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (account_id, session_id))
            rows = _dict_rows(cur)
        if not rows:
            raise HTTPException(404, "Session not found")
        return {"success": True, "session": rows[0]}

    # ── U15 startSession ─────────────────────────────────────────────
    def startSession(self, session_id: int) -> dict[str, Any]:
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT courseid, status FROM attendance_session WHERE attendancesessionid = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "课时不存在")
            course_id, current_status = row
            if current_status == "active":
                raise HTTPException(409, "该课时已处于进行中")
            if current_status == "ended":
                raise HTTPException(409, "该课时已结束，无法再次开始")
            cur.execute(
                """SELECT 1 FROM attendance_session
                   WHERE courseid = %s AND status = 'active'
                     AND attendancesessionid <> %s LIMIT 1""",
                (course_id, session_id),
            )
            if cur.fetchone():
                raise HTTPException(409, "该课程已有进行中的课时")
            cur.execute(
                """UPDATE attendance_session
                   SET status = 'active',
                       start_time = CASE WHEN start_time > NOW() THEN NOW() ELSE start_time END
                   WHERE attendancesessionid = %s""",
                (session_id,),
            )
        return {"success": True, "session_id": session_id, "status": "active"}

    # ── U15 endSession ───────────────────────────────────────────────
    def endSession(
        self, session_id: int, background_tasks: BackgroundTasks | None = None,
    ) -> dict[str, Any]:
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT courseid, status FROM attendance_session WHERE attendancesessionid = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "课时不存在")
            course_id, current_status = row
            if current_status == "ended":
                raise HTTPException(409, "该课时已结束")
            if current_status == "cancelled":
                raise HTTPException(409, "该课时已取消")
            cur.execute(
                """INSERT INTO attendance_record (attendancesessionid, accountid, status)
                   SELECT %s, e.accountid, 'absent'
                   FROM course_enrollment e
                   WHERE e.courseid = %s AND e.status = 'active'
                   ON CONFLICT (attendancesessionid, accountid) DO NOTHING""",
                (session_id, course_id),
            )
            absentees = cur.rowcount
            cur.execute(
                """UPDATE attendance_session
                   SET status = 'ended',
                       end_time = COALESCE(end_time, NOW())
                   WHERE attendancesessionid = %s""",
                (session_id,),
            )
        recipients = self._fetch_late_absent_recipients(session_id)
        if background_tasks is not None:
            background_tasks.add_task(send_late_absent_emails, recipients)
            queued = len(recipients)
        else:
            send_late_absent_emails(recipients)
            queued = len(recipients)
        return {
            "success": True, "session_id": session_id,
            "marked_absent": absentees, "notifications_queued": queued,
        }

    # ── internal ─────────────────────────────────────────────────────
    def _fetch_late_absent_recipients(self, session_id: int) -> list[dict[str, Any]]:
        sql = """
            SELECT ua.email, pi.full_name, r.status,
                   c.course_code, c.course_name, s.start_time
            FROM attendance_record r
            JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            JOIN user_account ua ON ua.accountid = r.accountid
            LEFT JOIN personal_info pi ON pi.accountid = r.accountid
            WHERE r.attendancesessionid = %s
              AND r.status IN ('late', 'absent')
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (session_id,))
            return _dict_rows(cur)


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["attendanceSession"])


def _svc(request: Request) -> AttendanceSession:
    return AttendanceSession(request.app.state.cfg.database_url)


# Student U07
@router.get("/student/sessions/{session_id}")
def student_session_detail(
    session_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("student")),
):
    return _svc(request).viewSessionDetail(session_id, user.account_id)


# Teacher U15
@router.post("/teacher/sessions/{session_id}/start")
def teacher_start_session(
    session_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).startSession(session_id)


@router.post("/teacher/sessions/{session_id}/end")
def teacher_end_session(
    session_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).endSession(session_id, background_tasks)


# Teacher: list sessions / courses (read-only views)
@router.get("/teacher/courses")
def teacher_list_courses(
    request: Request, user: CurrentUser = Depends(require_role("teacher"))
):
    sql = """
        SELECT c.courseid, c.course_code, c.course_name,
               COALESCE(c.status,'active') AS status,
               (SELECT COUNT(*) FROM course_enrollment e
                  WHERE e.courseid = c.courseid AND e.status='active') AS enrolled
        FROM course c
        WHERE COALESCE(c.status,'active') = 'active'
        ORDER BY c.course_code
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "courses": _dict_rows(cur)}


@router.get("/teacher/sessions")
def teacher_list_sessions(
    request: Request,
    course_id: int | None = None,
    status: str | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    clauses, params = [], []
    if course_id is not None:
        clauses.append("s.courseid = %s"); params.append(course_id)
    if status:
        if status not in ("scheduled", "active", "ended", "cancelled"):
            raise HTTPException(400, "status 非法")
        clauses.append("s.status = %s"); params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT s.attendancesessionid, s.courseid, c.course_code, c.course_name,
               s.start_time, s.end_time, s.status
        FROM attendance_session s
        JOIN course c ON c.courseid = s.courseid
        {where}
        ORDER BY s.start_time DESC
        LIMIT 500
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return {"success": True, "sessions": _dict_rows(cur)}


@router.get("/teacher/courses/{course_id}/students")
def teacher_course_roster(
    course_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    sql = """
        SELECT e.accountid, pi.full_name, pi.student_id, ua.email, e.status,
               COUNT(r.attendancerecordid)
                 FILTER (WHERE r.status IN ('present','late')) AS attended,
               COUNT(s.attendancesessionid)
                 FILTER (WHERE s.status='ended') AS sessions_completed
        FROM course_enrollment e
        JOIN user_account ua ON ua.accountid = e.accountid
        LEFT JOIN personal_info pi ON pi.accountid = e.accountid
        LEFT JOIN attendance_session s ON s.courseid = e.courseid
        LEFT JOIN attendance_record r
          ON r.attendancesessionid = s.attendancesessionid AND r.accountid = e.accountid
        WHERE e.courseid = %s AND e.status = 'active'
        GROUP BY e.accountid, pi.full_name, pi.student_id, ua.email, e.status
        ORDER BY pi.full_name NULLS LAST, e.accountid
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql, (course_id,))
        return {"success": True, "students": _dict_rows(cur)}


# ── Admin: Course management U26 + session scheduling + enrolments ──
class CourseBody(BaseModel):
    course_code: str
    course_name: str


class CourseStatusBody(BaseModel):
    status: str


class EnrollmentBody(BaseModel):
    account_id: int


class SessionBody(BaseModel):
    course_id: int
    start_time: str
    end_time: str | None = None
    status: str = "scheduled"


class SessionPatchBody(BaseModel):
    start_time: str | None = None
    end_time: str | None = None
    status: str | None = None


@router.get("/admin/courses")
def admin_list_courses(
    request: Request, user: CurrentUser = Depends(require_role("admin"))
):
    sql = """
        SELECT c.courseid, c.course_code, c.course_name,
               COALESCE(c.status,'active') AS status,
               (SELECT COUNT(*) FROM attendance_session s
                  WHERE s.courseid = c.courseid AND s.status='active') AS active_sessions
        FROM course c ORDER BY c.courseid
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "courses": _dict_rows(cur)}


@router.post("/admin/courses")
def admin_create_course(
    body: CourseBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM course WHERE course_code = %s", (body.course_code,))
        if cur.fetchone():
            raise HTTPException(409, f"课程代码 {body.course_code} 已存在")
        cur.execute(
            "INSERT INTO course (course_code, course_name) VALUES (%s, %s) RETURNING courseid",
            (body.course_code, body.course_name),
        )
        course_id = cur.fetchone()[0]
    return {"success": True, "course_id": course_id}


@router.patch("/admin/courses/{course_id}/status")
def admin_set_course_status(
    course_id: int, body: CourseStatusBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("active", "inactive"):
        raise HTTPException(400, "status 非法")
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(
            "UPDATE course SET status = %s WHERE courseid = %s",
            (body.status, course_id),
        )
    return {"success": True}


@router.delete("/admin/courses/{course_id}")
def admin_delete_course(
    course_id: int, request: Request, force: bool = False,
    user: CurrentUser = Depends(require_role("admin")),
):
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM course WHERE courseid = %s", (course_id,))
        if not cur.fetchone():
            raise HTTPException(404, "课程不存在")
        cur.execute(
            """SELECT 1 FROM attendance_record r
               JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
               WHERE s.courseid = %s LIMIT 1""",
            (course_id,),
        )
        if cur.fetchone():
            raise HTTPException(409, "该课程已有签到记录，无法删除（请改为停用）")
        cur.execute(
            "SELECT COUNT(*) FROM attendance_session WHERE courseid = %s",
            (course_id,),
        )
        session_count = cur.fetchone()[0]
        if session_count > 0 and not force:
            raise HTTPException(
                409, f"该课程下有 {session_count} 个课时安排，确认删除请使用 force=true",
            )
        if session_count > 0:
            cur.execute("DELETE FROM attendance_session WHERE courseid = %s", (course_id,))
        cur.execute("DELETE FROM course WHERE courseid = %s", (course_id,))
    return {"success": True}


@router.get("/admin/courses/{course_id}/enrollments")
def admin_list_enrollments(
    course_id: int, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    sql = """
        SELECT e.enrollmentid, e.accountid, e.status,
               pi.full_name, pi.student_id, ua.email
        FROM course_enrollment e
        JOIN user_account ua ON ua.accountid = e.accountid
        LEFT JOIN personal_info pi ON pi.accountid = e.accountid
        WHERE e.courseid = %s
        ORDER BY pi.full_name NULLS LAST, e.accountid
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql, (course_id,))
        return {"success": True, "enrollments": _dict_rows(cur)}


@router.post("/admin/courses/{course_id}/enrollments")
def admin_create_enrollment(
    course_id: int, body: EnrollmentBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM course WHERE courseid = %s", (course_id,))
        if not cur.fetchone():
            raise HTTPException(404, "课程不存在")
        cur.execute(
            """SELECT up.role FROM user_account ua
               JOIN user_profiles up ON up.profileid = ua.profileid
               WHERE ua.accountid = %s""",
            (body.account_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "用户不存在")
        if row[0] != "student":
            raise HTTPException(400, "只能将课程分配给学生")
        cur.execute(
            """INSERT INTO course_enrollment (courseid, accountid, status)
               VALUES (%s, %s, 'active')
               ON CONFLICT (courseid, accountid)
                 DO UPDATE SET status = 'active'
               RETURNING enrollmentid""",
            (course_id, body.account_id),
        )
        eid = cur.fetchone()[0]
    return {"success": True, "enrollment_id": eid}


@router.delete("/admin/courses/{course_id}/enrollments/{account_id}")
def admin_delete_enrollment(
    course_id: int, account_id: int, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(
            "DELETE FROM course_enrollment WHERE courseid = %s AND accountid = %s",
            (course_id, account_id),
        )
    return {"success": True}


@router.get("/admin/sessions")
def admin_list_sessions(
    request: Request, course_id: int | None = None,
    user: CurrentUser = Depends(require_role("admin")),
):
    sql = """
        SELECT s.attendancesessionid, s.courseid, c.course_code, c.course_name,
               s.start_time, s.end_time, s.status
        FROM attendance_session s
        JOIN course c ON c.courseid = s.courseid
        {where}
        ORDER BY s.start_time DESC LIMIT 500
    """
    where = "WHERE s.courseid = %s" if course_id else ""
    params = (course_id,) if course_id else ()
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        return {"success": True, "sessions": _dict_rows(cur)}


@router.post("/admin/sessions")
def admin_create_session(
    body: SessionBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("scheduled", "active", "ended", "cancelled"):
        raise HTTPException(400, "status 非法")
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
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
            """INSERT INTO attendance_session (courseid, start_time, end_time, status)
               VALUES (%s, %s, %s, %s) RETURNING attendancesessionid""",
            (body.course_id, body.start_time, body.end_time, body.status),
        )
        sid = cur.fetchone()[0]
    return {"success": True, "session_id": sid}


@router.patch("/admin/sessions/{session_id}")
def admin_update_session(
    session_id: int, body: SessionPatchBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    fields, params = [], []
    if body.start_time is not None:
        fields.append("start_time = %s"); params.append(body.start_time)
    if body.end_time is not None:
        fields.append("end_time = %s"); params.append(body.end_time)
    if body.status is not None:
        if body.status not in ("scheduled", "active", "ended", "cancelled"):
            raise HTTPException(400, "status 非法")
        fields.append("status = %s"); params.append(body.status)
    if not fields:
        raise HTTPException(400, "无可更新字段")
    params.append(session_id)
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(
            f"UPDATE attendance_session SET {', '.join(fields)} WHERE attendancesessionid = %s",
            params,
        )
    return {"success": True}


@router.delete("/admin/sessions/{session_id}")
def admin_delete_session(
    session_id: int, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
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
