"""Class 7: report

Covers UC: U14 Export Attendance Report (exportAllMyClassReport).
"""

from __future__ import annotations

import contextlib
import csv
import io
from typing import Any

import psycopg2
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from core.userInformation import (
    CurrentUser,
    assert_teacher_course,
    assert_teacher_session,
    require_role,
    teacher_course_ids,
)


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


class Report:
    """Attendance report generator (CSV)."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    # ── U14 exportAllMyClassReport ──────────────────────────────────
    def exportAllMyClassReport(
        self,
        course_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        session_id: int | None = None,
        course_scope: list[int] | None = None,
    ) -> bytes:
        # ``course_scope`` is the requesting teacher's own courses. It is
        # applied on top of any explicit filter so an unfiltered export can
        # never spill another teacher's class into the CSV. An empty list is
        # a real answer (no assigned courses) and still gets applied.
        clauses, params = [], []
        if course_scope is not None:
            clauses.append("s.courseid = ANY(%s)")
            params.append(list(course_scope))
        if session_id is not None:
            clauses.append("s.attendancesessionid = %s")
            params.append(session_id)
        if course_id is not None:
            clauses.append("s.courseid = %s")
            params.append(course_id)
        if date_from:
            clauses.append("s.start_time >= %s")
            params.append(date_from)
        if date_to:
            clauses.append("s.start_time <= %s")
            params.append(date_to)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT c.course_code, c.course_name,
                   s.attendancesessionid, s.start_time, s.end_time,
                   s.status AS session_status,
                   pi.student_id, pi.full_name, ua.email,
                   r.status AS attendance_status, r.marked_at
            FROM attendance_session s
            JOIN course c ON c.courseid = s.courseid
            JOIN course_enrollment e ON e.courseid = s.courseid AND e.status = 'active'
            JOIN user_account ua ON ua.accountid = e.accountid
            LEFT JOIN personal_info pi ON pi.accountid = e.accountid
            LEFT JOIN attendance_record r
              ON r.attendancesessionid = s.attendancesessionid AND r.accountid = e.accountid
            {where}
            ORDER BY s.start_time DESC, pi.full_name NULLS LAST
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, params)
            rows = _dict_rows(cur)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "course_code", "course_name", "session_id", "start_time", "end_time",
            "session_status", "student_id", "full_name", "email",
            "attendance_status", "marked_at",
        ])
        for r in rows:
            writer.writerow([
                r.get("course_code"), r.get("course_name"),
                r.get("attendancesessionid"),
                r.get("start_time"), r.get("end_time"), r.get("session_status"),
                r.get("student_id"), r.get("full_name"), r.get("email"),
                r.get("attendance_status") or "absent", r.get("marked_at"),
            ])
        return buf.getvalue().encode("utf-8")


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["report"])


def _svc(request: Request) -> Report:
    return Report(request.app.state.cfg.database_url)


@router.get("/teacher/reports/export")
def teacher_export_report(
    request: Request,
    course_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    session_id: int | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    """U14: stream CSV attendance report for the teacher's own classes."""
    database_url = request.app.state.cfg.database_url
    if course_id is not None:
        assert_teacher_course(database_url, user.account_id, course_id)
    if session_id is not None:
        assert_teacher_session(database_url, user.account_id, session_id)
    scope = teacher_course_ids(database_url, user.account_id)
    csv_bytes = _svc(request).exportAllMyClassReport(
        course_id, date_from, date_to, session_id, course_scope=scope
    )
    fname = "attendance_report.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
