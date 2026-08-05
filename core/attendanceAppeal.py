"""Class 6: attendanceAppeal

Covers UC:
  U08 Submit Attendance Appeal       (appealAbsence, reviewAppeal — teacher)
  U28 Submit Leave Application       (makeLeaveApplication)
  U31 Review Leave Application       (approveLeaveApplication)

Review authority (per FTD U08): appeals are reviewed by the TEACHER.
Admins may list appeals for oversight but have no review endpoint.

Attributes: applicant (account_id of the requester).
"""

from __future__ import annotations

import contextlib
from typing import Any

import psycopg2
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel

from core.notification import send_review_outcome_email
from core.userInformation import CurrentUser, get_current_user, require_role


@contextlib.contextmanager
def _db(database_url: str):
    conn = psycopg2.connect(database_url)
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


class AttendanceAppeal:
    """Appeal + leave-application entity."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    # ── U08 appealAbsence (existing record disputed) ────────────────
    def appealAbsence(self, applicant: int, record_id: int, reason: str) -> dict[str, Any]:
        if not reason or not reason.strip():
            raise HTTPException(400, "Appeal reason cannot be empty")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT accountid FROM attendance_record WHERE attendancerecordid = %s",
                (record_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Record not found")
            if row[0] != applicant:
                raise HTTPException(403, "Cannot appeal another user's record")
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

    # ── U08 reviewAppeal (teacher decision) ─────────────────────────
    def reviewAppeal(
        self, reviewer: int, appeal_id: int, status: str,
        background_tasks: BackgroundTasks | None = None,
    ) -> dict[str, Any]:
        if status not in ("approved", "rejected"):
            raise HTTPException(400, "status must be approved/rejected")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT a.attendancerecordid, a.status,
                          ua.email, pi.full_name,
                          c.course_code, c.course_name, s.start_time
                   FROM attendance_appeal a
                   JOIN attendance_record r ON r.attendancerecordid = a.attendancerecordid
                   JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
                   JOIN course c ON c.courseid = s.courseid
                   JOIN user_account ua ON ua.accountid = a.accountid
                   LEFT JOIN personal_info pi ON pi.accountid = a.accountid
                   WHERE a.appealid = %s""",
                (appeal_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Appeal not found")
            record_id, current, email, full_name, code, cname, start_time = row
            if current != "pending":
                raise HTTPException(409, f"This appeal has already been {current}")
            cur.execute(
                """UPDATE attendance_appeal
                   SET status = %s, reviewed_by = %s, reviewed_at = NOW()
                   WHERE appealid = %s""",
                (status, reviewer, appeal_id),
            )
            if status == "approved":
                # The disputed record is corrected to 'present' (mirrors how
                # an approved leave application writes 'leave' in U31).
                cur.execute(
                    """UPDATE attendance_record
                       SET status = 'present', marked_at = NOW()
                       WHERE attendancerecordid = %s""",
                    (record_id,),
                )
        # U08: the student is notified once the appeal has been reviewed.
        payload = {
            "email": email, "full_name": full_name, "kind": "attendance appeal",
            "decision": status, "course": f"{code} {cname}".strip(),
            "start_time": start_time,
            "corrected_status": "present" if status == "approved" else None,
        }
        if background_tasks is not None:
            background_tasks.add_task(send_review_outcome_email, payload)
        else:
            send_review_outcome_email(payload)
        return {"success": True, "appeal_id": appeal_id, "status": status}

    def listAppealsForReview(self) -> dict[str, Any]:
        """All appeals with course/session context, pending first."""
        sql = """
            SELECT a.appealid, a.attendancerecordid, a.accountid,
                   pi.full_name, pi.student_id,
                   c.course_code, c.course_name, s.start_time,
                   r.status AS record_status,
                   a.reason, a.status, a.created_at, a.reviewed_at,
                   -- Who decided the appeal (U08 audit trail): reviewers are
                   -- teachers or admins, so their name lives in personal_info
                   -- under staff_id, and the role comes from user_profiles.
                   a.reviewed_by,
                   rpi.full_name AS reviewer_name,
                   rpi.staff_id  AS reviewer_staff_id,
                   rua.email     AS reviewer_email,
                   rup.role      AS reviewer_role
            FROM attendance_appeal a
            JOIN attendance_record r ON r.attendancerecordid = a.attendancerecordid
            JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            LEFT JOIN personal_info pi ON pi.accountid = a.accountid
            LEFT JOIN user_account  rua ON rua.accountid = a.reviewed_by
            LEFT JOIN user_profiles rup ON rup.profileid = rua.profileid
            LEFT JOIN personal_info rpi ON rpi.accountid = a.reviewed_by
            ORDER BY (a.status = 'pending') DESC, a.created_at DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql)
            return {"success": True, "appeals": _dict_rows(cur)}

    # ── U28 makeLeaveApplication (future session) ───────────────────
    def makeLeaveApplication(
        self, applicant: int, session_id: int, reason: str,
        supporting_doc_url: str | None = None,
    ) -> dict[str, Any]:
        if not reason or not reason.strip():
            raise HTTPException(400, "Leave reason cannot be empty")
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
                    400, "This session has already started/ended; please submit an attendance appeal instead",
                )
            cur.execute(
                """SELECT 1 FROM course_enrollment
                   WHERE courseid = %s AND accountid = %s AND status = 'active'""",
                (course_id, applicant),
            )
            if not cur.fetchone():
                raise HTTPException(403, "Not enrolled in this course")
            cur.execute(
                """INSERT INTO leave_application
                       (accountid, attendancesessionid, reason, supporting_doc_url, status)
                   VALUES (%s, %s, %s, %s, 'pending')
                   RETURNING leaveapplicationid""",
                (applicant, session_id, reason, supporting_doc_url),
            )
            leave_id = cur.fetchone()[0]
        return {"success": True, "leave_application_id": leave_id}

    # ── U28 supporting evidence (upload / fetch) ────────────────────
    # Kept small and explicit rather than general-purpose: this is the only
    # user-uploaded file the system stores, and the limits are what make
    # storing it in the database acceptable.
    ALLOWED_DOC_TYPES = {
        "image/png", "image/jpeg", "image/webp", "application/pdf",
    }
    MAX_DOC_BYTES = 5 * 1024 * 1024

    def attachSupportingDocument(
        self, applicant: int, leave_id: int,
        filename: str | None, content_type: str | None, data: bytes,
    ) -> dict[str, Any]:
        if content_type not in self.ALLOWED_DOC_TYPES:
            raise HTTPException(
                400,
                "Supporting document must be a PNG, JPEG, WebP or PDF file "
                f"(received {content_type or 'an unknown type'})",
            )
        if not data:
            raise HTTPException(400, "The uploaded file is empty")
        if len(data) > self.MAX_DOC_BYTES:
            raise HTTPException(
                400,
                f"Supporting document must be {self.MAX_DOC_BYTES // (1024 * 1024)} MB "
                f"or smaller (received {len(data) / (1024 * 1024):.1f} MB)",
            )
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT accountid, status FROM leave_application WHERE leaveapplicationid = %s",
                (leave_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Leave application not found")
            owner, status = row
            if owner != applicant:
                raise HTTPException(403, "This is not your leave application")
            # Once a decision is made the evidence it was based on is fixed.
            if status != "pending":
                raise HTTPException(
                    409, "This application has already been reviewed",
                )
            cur.execute(
                """UPDATE leave_application
                      SET supporting_doc = %s, supporting_doc_name = %s,
                          supporting_doc_type = %s, updated_at = NOW()
                    WHERE leaveapplicationid = %s""",
                (psycopg2.Binary(data), filename, content_type, leave_id),
            )
        return {
            "success": True, "leave_application_id": leave_id,
            "filename": filename, "content_type": content_type,
            "bytes": len(data),
        }

    def getSupportingDocument(
        self, viewer: int, role: str, leave_id: int,
    ) -> tuple[bytes, str, str]:
        """Return (data, content_type, filename) if the viewer may see it.

        Visible to the student who submitted it, to any teacher (they are the
        review authority for leave, per U31) and to admins for oversight.
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT accountid, supporting_doc,
                          supporting_doc_type, supporting_doc_name
                     FROM leave_application WHERE leaveapplicationid = %s""",
                (leave_id,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Leave application not found")
        owner, data, content_type, filename = row
        if role == "student" and owner != viewer:
            raise HTTPException(403, "This is not your leave application")
        if data is None:
            raise HTTPException(404, "No supporting document was attached")
        return (
            bytes(data),
            content_type or "application/octet-stream",
            filename or f"leave_{leave_id}",
        )

    def listMyLeaveApplications(self, applicant: int) -> dict[str, Any]:
        sql = """
            SELECT la.leaveapplicationid, la.attendancesessionid,
                   c.course_code, c.course_name, s.start_time,
                   la.reason, la.status, la.created_at, la.reviewed_at,
                   la.reviewer_comment,
                   (la.supporting_doc IS NOT NULL) AS has_document,
                   la.supporting_doc_name
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
        background_tasks: BackgroundTasks | None = None,
    ) -> dict[str, Any]:
        if decision not in ("approved", "rejected"):
            raise HTTPException(400, "decision must be approved/rejected")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT la.accountid, la.attendancesessionid,
                          ua.email, pi.full_name,
                          c.course_code, c.course_name, s.start_time
                   FROM leave_application la
                   JOIN attendance_session s ON s.attendancesessionid = la.attendancesessionid
                   JOIN course c ON c.courseid = s.courseid
                   JOIN user_account ua ON ua.accountid = la.accountid
                   LEFT JOIN personal_info pi ON pi.accountid = la.accountid
                   WHERE la.leaveapplicationid = %s""",
                (leave_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Leave application not found")
            student_id, session_id, email, full_name, code, cname, start_time = row
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
        # U31: notify the student of the review outcome.
        payload = {
            "email": email, "full_name": full_name, "kind": "leave application",
            "decision": decision, "course": f"{code} {cname}".strip(),
            "start_time": start_time, "comment": comment,
            "corrected_status": "leave" if decision == "approved" else None,
        }
        if background_tasks is not None:
            background_tasks.add_task(send_review_outcome_email, payload)
        else:
            send_review_outcome_email(payload)
        return {"success": True}

    def listLeaveApplicationsForReview(self) -> dict[str, Any]:
        """Every leave application, newest first — not just the pending ones.

        Mirrors the appeals list: the teacher opens one to read it in full and
        decide, and can still go back to an already-reviewed application to
        see what was decided and why (reviewer_comment only exists on those).

        `attended` / `sessions_completed` give the student's track record in
        that same course, so the decision has some context. The ratio counts
        only ENDED sessions, matching teacher_course_roster — an in-progress
        session's provisional 'present' must not inflate the numerator.
        The supporting document is deliberately NOT selected here: it is a
        BYTEA and would bloat every row of the list. Only its presence and
        metadata travel with the list; the bytes have their own endpoint.
        """
        sql = """
            SELECT la.leaveapplicationid, la.accountid,
                   pi.full_name, pi.student_id,
                   s.courseid, c.course_code, c.course_name, s.start_time,
                   la.reason, la.status, la.created_at,
                   la.reviewed_at, la.reviewer_comment,
                   (la.supporting_doc IS NOT NULL) AS has_document,
                   la.supporting_doc_name, la.supporting_doc_type,
                   (SELECT COUNT(*)
                      FROM attendance_record r2
                      JOIN attendance_session s2
                        ON s2.attendancesessionid = r2.attendancesessionid
                     WHERE r2.accountid = la.accountid
                       AND s2.courseid = s.courseid
                       AND s2.status = 'ended'
                       AND r2.status IN ('present', 'late')) AS attended,
                   (SELECT COUNT(*)
                      FROM attendance_session s3
                     WHERE s3.courseid = s.courseid
                       AND s3.status = 'ended')              AS sessions_completed
            FROM leave_application la
            JOIN attendance_session s ON s.attendancesessionid = la.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            LEFT JOIN personal_info pi ON pi.accountid = la.accountid
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


# Teacher: U08 — review attendance appeals
@router.get("/teacher/appeals")
def teacher_list_appeals(
    request: Request, user: CurrentUser = Depends(require_role("teacher"))
):
    return _svc(request).listAppealsForReview()


@router.patch("/teacher/appeals/{appeal_id}")
def teacher_review_appeal(
    appeal_id: int, body: AppealReviewBody,
    background_tasks: BackgroundTasks, request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).reviewAppeal(
        user.account_id, appeal_id, body.status, background_tasks,
    )


# U28 supporting evidence. Deliberately a second step rather than part of the
# create call: the attachment is optional, and keeping POST
# /student/leave-applications a plain JSON endpoint means an application is
# never lost because its attachment failed to upload.
@router.post("/student/leave-applications/{leave_id}/document")
async def student_attach_leave_document(
    leave_id: int, request: Request,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_role("student")),
):
    data = await file.read()
    return _svc(request).attachSupportingDocument(
        user.account_id, leave_id, file.filename, file.content_type, data,
    )


@router.get("/leave-applications/{leave_id}/document")
def get_leave_document(
    leave_id: int, request: Request,
    user: CurrentUser = Depends(get_current_user),
):
    data, content_type, filename = _svc(request).getSupportingDocument(
        user.account_id, user.role, leave_id,
    )
    return Response(
        content=data,
        media_type=content_type,
        # inline: the teacher reviews it in the page, not in a download folder.
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# Teacher: U31
@router.get("/teacher/leave-applications")
def teacher_list_leave(
    request: Request, user: CurrentUser = Depends(require_role("teacher"))
):
    return _svc(request).listLeaveApplicationsForReview()


@router.patch("/teacher/leave-applications/{leave_id}")
def teacher_review_leave(
    leave_id: int, body: LeaveReviewBody,
    background_tasks: BackgroundTasks, request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).approveLeaveApplication(
        user.account_id, leave_id, body.decision, body.comment, background_tasks,
    )


# Admin: oversight only — admins may list appeals but not review them
# (FTD U08: review authority belongs to the teacher).
@router.get("/admin/appeals")
def admin_list_appeals(
    request: Request, user: CurrentUser = Depends(require_role("admin"))
):
    return _svc(request).listAppealsForReview()
