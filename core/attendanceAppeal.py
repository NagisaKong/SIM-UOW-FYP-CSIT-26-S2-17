"""Class 6: attendanceAppeal

Covers UC:
  U08 Submit Attendance Appeal       (appealAbsence)
  U28 Submit Leave Application       (makeLeaveApplication)
  U31 Review Leave Application       (approveLeaveApplication)

Attributes: applicant (account_id of the requester).
"""

from __future__ import annotations

import contextlib
from typing import Any

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

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


class AttendanceAppeal:
    """Appeal + leave-application entity."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    # ── U08 appealAbsence (existing record disputed) ────────────────
    def appealAbsence(self, applicant: int, record_id: int, reason: str) -> dict[str, Any]:
        if not reason or not reason.strip():
            raise HTTPException(400, "申诉原因不能为空")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT accountid FROM attendance_record WHERE attendancerecordid = %s",
                (record_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Record not found")
            if row[0] != applicant:
                raise HTTPException(403, "不能对他人记录申诉")
            cur.execute(
                """INSERT INTO attendance_appeal (attendancerecordid, accountid, reason)
                   VALUES (%s, %s, %s) RETURNING appealid""",
                (record_id, applicant, reason),
            )
            appeal_id = cur.fetchone()[0]
        return {"success": True, "appeal_id": appeal_id}

    def listMyAppeals(self, applicant: int) -> dict[str, Any]:
        sql = """
            SELECT a.appealid, a.attendancerecordid, a.reason, a.status,
                   a.created_at, a.reviewed_at
            FROM attendance_appeal a
            WHERE a.accountid = %s
            ORDER BY a.created_at DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (applicant,))
            return {"success": True, "appeals": _dict_rows(cur)}

    # ── U28 makeLeaveApplication (future session) ───────────────────
    def makeLeaveApplication(
        self, applicant: int, session_id: int, reason: str,
        supporting_doc_url: str | None = None,
    ) -> dict[str, Any]:
        if not reason or not reason.strip():
            raise HTTPException(400, "请假原因不能为空")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT s.start_time, s.status, s.courseid
                   FROM attendance_session s WHERE s.attendancesessionid = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Session not found")
            start_time, sstatus, course_id = row
            if sstatus in ("active", "ended"):
                raise HTTPException(
                    400, "该课程已开始/结束，请改为提交考勤申诉（U08）",
                )
            cur.execute(
                """SELECT 1 FROM course_enrollment
                   WHERE courseid = %s AND accountid = %s AND status = 'active'""",
                (course_id, applicant),
            )
            if not cur.fetchone():
                raise HTTPException(403, "未选修该课程")
            cur.execute(
                """INSERT INTO leave_application
                       (accountid, attendancesessionid, reason, supporting_doc_url, status)
                   VALUES (%s, %s, %s, %s, 'pending')
                   RETURNING leaveapplicationid""",
                (applicant, session_id, reason, supporting_doc_url),
            )
            leave_id = cur.fetchone()[0]
        return {"success": True, "leave_application_id": leave_id}

    def listMyLeaveApplications(self, applicant: int) -> dict[str, Any]:
        sql = """
            SELECT la.leaveapplicationid, la.attendancesessionid,
                   c.course_code, c.course_name, s.start_time,
                   la.reason, la.status, la.created_at, la.reviewed_at,
                   la.reviewer_comment
            FROM leave_application la
            JOIN attendance_session s ON s.attendancesessionid = la.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            WHERE la.accountid = %s
            ORDER BY la.created_at DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (applicant,))
            return {"success": True, "applications": _dict_rows(cur)}

    # ── U31 approveLeaveApplication (teacher) ───────────────────────
    def approveLeaveApplication(
        self, reviewer: int, leave_id: int, decision: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        if decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision 必须为 approved/rejected")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT accountid, attendancesessionid FROM leave_application
                   WHERE leaveapplicationid = %s""",
                (leave_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Leave application not found")
            student_id, session_id = row
            cur.execute(
                """UPDATE leave_application
                   SET status = %s, reviewed_by = %s, reviewed_at = NOW(),
                       reviewer_comment = %s
                   WHERE leaveapplicationid = %s""",
                (decision, reviewer, comment, leave_id),
            )
            if decision == "approved":
                # Mark or update attendance_record so analytics exclude this session.
                cur.execute(
                    """INSERT INTO attendance_record
                          (attendancesessionid, accountid, status)
                       VALUES (%s, %s, 'leave')
                       ON CONFLICT (attendancesessionid, accountid)
                         DO UPDATE SET status = 'leave', marked_at = NOW()""",
                    (session_id, student_id),
                )
        return {"success": True}

    def listPendingLeaveApplications(self) -> dict[str, Any]:
        sql = """
            SELECT la.leaveapplicationid, la.accountid,
                   pi.full_name, pi.student_id,
                   c.course_code, c.course_name, s.start_time,
                   la.reason, la.supporting_doc_url, la.status, la.created_at
            FROM leave_application la
            JOIN attendance_session s ON s.attendancesessionid = la.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            LEFT JOIN personal_info pi ON pi.accountid = la.accountid
            WHERE la.status = 'pending'
            ORDER BY la.created_at DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql)
            return {"success": True, "applications": _dict_rows(cur)}


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["attendanceAppeal"])


def _svc(request: Request) -> AttendanceAppeal:
    return AttendanceAppeal(request.app.state.cfg.database_url)


class AppealBody(BaseModel):
    record_id: int
    reason: str


class LeaveBody(BaseModel):
    session_id: int
    reason: str
    supporting_doc_url: str | None = None


class LeaveReviewBody(BaseModel):
    decision: str   # approved | rejected
    comment: str | None = None


class AppealReviewBody(BaseModel):
    status: str   # approved | rejected


# Student: U08
@router.post("/student/appeals")
def student_create_appeal(
    body: AppealBody, request: Request,
    user: CurrentUser = Depends(require_role("student")),
):
    return _svc(request).appealAbsence(user.account_id, body.record_id, body.reason)


@router.get("/student/appeals")
def student_list_appeals(
    request: Request, user: CurrentUser = Depends(require_role("student"))
):
    return _svc(request).listMyAppeals(user.account_id)


# Student: U28
@router.post("/student/leave-applications")
def student_create_leave(
    body: LeaveBody, request: Request,
    user: CurrentUser = Depends(require_role("student")),
):
    return _svc(request).makeLeaveApplication(
        user.account_id, body.session_id, body.reason, body.supporting_doc_url
    )


@router.get("/student/leave-applications")
def student_list_leave(
    request: Request, user: CurrentUser = Depends(require_role("student"))
):
    return _svc(request).listMyLeaveApplications(user.account_id)


# Teacher: U31
@router.get("/teacher/leave-applications")
def teacher_list_leave(
    request: Request, user: CurrentUser = Depends(require_role("teacher"))
):
    return _svc(request).listPendingLeaveApplications()


@router.patch("/teacher/leave-applications/{leave_id}")
def teacher_review_leave(
    leave_id: int, body: LeaveReviewBody, request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).approveLeaveApplication(
        user.account_id, leave_id, body.decision, body.comment
    )


# Admin: oversee appeals (parity with old endpoints)
@router.get("/admin/appeals")
def admin_list_appeals(
    request: Request, user: CurrentUser = Depends(require_role("admin"))
):
    sql = """
        SELECT a.appealid, a.attendancerecordid, a.accountid,
               pi.full_name, pi.student_id,
               a.reason, a.status, a.created_at, a.reviewed_at
        FROM attendance_appeal a
        LEFT JOIN personal_info pi ON pi.accountid = a.accountid
        ORDER BY a.created_at DESC
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "appeals": _dict_rows(cur)}


@router.patch("/admin/appeals/{appeal_id}")
def admin_review_appeal(
    appeal_id: int, body: AppealReviewBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("approved", "rejected"):
        raise HTTPException(400, "status 非法")
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(
            """UPDATE attendance_appeal
               SET status = %s, reviewed_by = %s, reviewed_at = NOW()
               WHERE appealid = %s""",
            (body.status, user.account_id, appeal_id),
        )
    return {"success": True}
