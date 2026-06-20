"""Class 2: attendanceRecord

Covers UC:
  U03 Automated Attendance Check-in   (viewActiveSessions — automatic, no manual check-in)
  U04 View Attendance Records         (viewAttendanceRecord)
  U12 View Real-Time Attendance       (viewRealTimeAttendanceStatus)
  U13 View Student Attendance History (viewAttendanceHistoryAcrossAllSession)
  U16 View Early Departure Summary    (viewEarlyLeftSummary)
  U27 Student Attendance Analytics    (viewStudentGraphicalReport)
  U30 Class Attendance Analytics      (viewTeacherGraphicalReport)
  U34 Configure Absence Threshold     (configureAbsenceThreshold)

Attributes:
  attendanceRate, minimumAttendanceRateRequirement, status,
  absenceThreshold.
"""

from __future__ import annotations

import contextlib
from typing import Any

import cv2
import numpy as np
import psycopg2
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from core.userInformation import CurrentUser, require_role


# Default fall-back values used when the attendance_threshold_config
# table is empty or unreachable (these mirror the row inserted by
# database/schema.sql).
_DEFAULT_MIN_RATE = 70.0
_DEFAULT_ABSENCE_THRESHOLD = 3
_DEFAULT_LATE_GRACE = 600
_DEFAULT_DETECTION_INTERVAL = 1200  # 20 min between detection windows (U03)


def load_threshold_config(database_url: str) -> dict[str, Any]:
    """Read the single-row attendance_threshold_config table.

    Falls back to module defaults if the table is missing or empty so
    the system keeps working even before the v0.5 migration runs.
    """
    try:
        with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT minimum_attendance_rate, absence_threshold,
                          late_grace_seconds, detection_interval_seconds
                   FROM attendance_threshold_config WHERE configid = 1"""
            )
            row = cur.fetchone()
            if row:
                return {
                    "minimum_attendance_rate": float(row[0]),
                    "absence_threshold": int(row[1]),
                    "late_grace_seconds": int(row[2]),
                    "detection_interval_seconds": int(row[3]),
                }
    except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn,
            psycopg2.OperationalError):
        pass
    except Exception:
        pass
    return {
        "minimum_attendance_rate": _DEFAULT_MIN_RATE,
        "absence_threshold": _DEFAULT_ABSENCE_THRESHOLD,
        "late_grace_seconds": _DEFAULT_LATE_GRACE,
        "detection_interval_seconds": _DEFAULT_DETECTION_INTERVAL,
    }


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


async def _bytes_to_cv2(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unable to parse image file")
    return img


class AttendanceRecord:
    """Attendance entity covering student/teacher/admin views.

    Business attributes (per class diagram) are loaded lazily from the
    attendance_threshold_config table — see load_threshold_config().
    """

    def __init__(self, pipeline, database_url: str):
        self.pipeline = pipeline
        self.database_url = database_url
        # Business attributes (per FYP class diagram) — backed by DB.
        cfg = load_threshold_config(database_url)
        self.minimumAttendanceRateRequirement: float = cfg["minimum_attendance_rate"]
        self.absenceThreshold: int = cfg["absence_threshold"]
        # attendanceRate / status are per-student-per-session values
        # computed on demand by the view* methods, not stored here.
        self.attendanceRate: float | None = None
        self.status: str | None = None

    # ── U03 Automated Attendance Check-in ────────────────────────────
    # Per U03, attendance is recorded automatically by the teacher-activated
    # classroom camera scan (SCRFD + ArcFace). The student performs NO manual
    # check-in. This method only lets the student see, for their enrolled
    # courses, any session the teacher is currently scanning and the live
    # status the system has recorded for them so far.
    def viewActiveSessions(self, account_id: int) -> dict[str, Any]:
        sql = """
            SELECT s.attendancesessionid AS session_id,
                   c.course_code, c.course_name,
                   s.start_time, s.status AS session_status,
                   r.status AS my_status, r.marked_at
            FROM attendance_session s
            JOIN course c ON c.courseid = s.courseid
            JOIN course_enrollment e
              ON e.courseid = s.courseid
             AND e.accountid = %s
             AND e.status = 'active'
            LEFT JOIN attendance_record r
              ON r.attendancesessionid = s.attendancesessionid
             AND r.accountid = %s
            WHERE s.status = 'active'
            ORDER BY s.start_time DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (account_id, account_id))
            return {"success": True, "sessions": _dict_rows(cur)}

    # ── U28 helper: upcoming (scheduled) sessions for leave requests ─
    def viewUpcomingSessions(self, account_id: int) -> dict[str, Any]:
        sql = """
            SELECT s.attendancesessionid AS session_id,
                   c.course_code, c.course_name, s.start_time
            FROM attendance_session s
            JOIN course c ON c.courseid = s.courseid
            JOIN course_enrollment e
              ON e.courseid = s.courseid
             AND e.accountid = %s
             AND e.status = 'active'
            WHERE s.status = 'scheduled'
            ORDER BY s.start_time
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (account_id,))
            return {"success": True, "sessions": _dict_rows(cur)}

    # ── U03 recordDetectionSnapshot (teacher-driven scan window) ─────
    # One detection window = one snapshot. The teacher's device uploads a
    # frame; we run SCRFD + ArcFace and write ONE temporary detection row
    # (presence_check) per enrolled student indicating whether they were
    # seen in this snapshot. Final statuses are aggregated later by
    # endSession (U15). The student performs no action (U03).
    def recordDetectionSnapshot(
        self, session_id: int, image: np.ndarray, camera_id: str | None = None,
    ) -> dict[str, Any]:
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT courseid, status FROM attendance_session WHERE attendancesessionid = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Session not found")
            course_id, status = row
            if status != "active":
                raise HTTPException(409, "The scan can only run while the session is active")

            # Recognise faces present in this frame.
            result = self.pipeline.process_frame(image)
            best_score: dict[int, float] = {}
            for p in result.predictions:
                if p.recognised and p.account_id is not None:
                    s = float(p.score)
                    if s > best_score.get(p.account_id, -1.0):
                        best_score[p.account_id] = s

            # Enrolled roster for this session's course.
            cur.execute(
                """SELECT e.accountid, pi.full_name, pi.student_id
                   FROM course_enrollment e
                   LEFT JOIN personal_info pi ON pi.accountid = e.accountid
                   WHERE e.courseid = %s AND e.status = 'active'""",
                (course_id,),
            )
            roster = _dict_rows(cur)

            # Write one detection row per enrolled student for this snapshot.
            detected = []
            for stu in roster:
                acc = stu["accountid"]
                seen = acc in best_score
                cur.execute(
                    """INSERT INTO presence_check
                         (attendancesessionid, accountid, detected, camera_id, confidence)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, acc, seen, camera_id,
                     round(best_score[acc], 3) if seen else None),
                )
                if seen:
                    detected.append({
                        "account_id": acc,
                        "full_name": stu.get("full_name"),
                        "student_id": stu.get("student_id"),
                        "confidence": round(best_score[acc], 3),
                    })

        faces = len(result.predictions)
        return {
            "success": True,
            "session_id": session_id,
            "enrolled": len(roster),
            "faces_in_frame": faces,
            "detected_count": len(detected),
            "detected": detected,
            "message": f"Snapshot recorded: {len(detected)}/{len(roster)} enrolled student(s) detected "
                       f"({faces} face(s) in frame)",
        }

    # ── U04 viewAttendanceRecord ─────────────────────────────────────
    # Returns every session of the courses the student is enrolled in,
    # including upcoming (not-yet-started) ones that have no attendance
    # record yet — those come back with status / record_id = NULL so the
    # UI can show them and disable the appeal action.
    def viewAttendanceRecord(self, account_id: int) -> dict[str, Any]:
        sql = """
            SELECT r.attendancerecordid AS record_id,
                   s.attendancesessionid AS session_id,
                   s.start_time, s.end_time, s.status AS session_status,
                   c.course_code, c.course_name,
                   r.status, r.marked_at
            FROM attendance_session s
            JOIN course c ON c.courseid = s.courseid
            JOIN course_enrollment e
              ON e.courseid = s.courseid
             AND e.accountid = %s
             AND e.status = 'active'
            LEFT JOIN attendance_record r
              ON r.attendancesessionid = s.attendancesessionid
             AND r.accountid = %s
            ORDER BY s.start_time DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, (account_id, account_id))
            return {"success": True, "records": _dict_rows(cur)}

    # ── U27 viewStudentGraphicalReport ───────────────────────────────
    def viewStudentGraphicalReport(
        self, account_id: int,
        date_from: str | None = None, date_to: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["r.accountid = %s"]
        params: list[Any] = [account_id]
        if date_from:
            clauses.append("s.start_time >= %s")
            params.append(date_from)
        if date_to:
            clauses.append("s.start_time <= %s")
            params.append(date_to)
        where = " AND ".join(clauses)
        trend_sql = f"""
            SELECT s.start_time::date AS bucket,
                   COUNT(*) FILTER (WHERE r.status='present') AS present,
                   COUNT(*) FILTER (WHERE r.status='late')    AS late,
                   COUNT(*) FILTER (WHERE r.status='absent')  AS absent,
                   COUNT(r.attendancerecordid) AS total
            FROM attendance_session s
            LEFT JOIN attendance_record r ON r.attendancesessionid = s.attendancesessionid
            WHERE {where}
            GROUP BY bucket ORDER BY bucket
        """
        breakdown_sql = f"""
            SELECT COUNT(*) FILTER (WHERE r.status='present') AS present,
                   COUNT(*) FILTER (WHERE r.status='late')    AS late,
                   COUNT(*) FILTER (WHERE r.status='absent')  AS absent,
                   COUNT(r.attendancerecordid) AS total
            FROM attendance_session s
            LEFT JOIN attendance_record r ON r.attendancesessionid = s.attendancesessionid
            WHERE {where}
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(trend_sql, params)
            trend = _dict_rows(cur)
            cur.execute(breakdown_sql, params)
            breakdown = _dict_rows(cur)[0] if cur.description else {}
        for row in trend:
            total = row.get("total") or 0
            attended = (row.get("present") or 0) + (row.get("late") or 0)
            row["rate"] = round(attended / total * 100, 1) if total else 0.0
        total = breakdown.get("total") or 0
        attended = (breakdown.get("present") or 0) + (breakdown.get("late") or 0)
        breakdown["rate"] = round(attended / total * 100, 1) if total else 0.0
        return {"success": True, "trend": trend, "breakdown": breakdown}

    # ── U30 viewTeacherGraphicalReport ───────────────────────────────
    def viewTeacherGraphicalReport(
        self, course_id: int,
        date_from: str | None = None, date_to: str | None = None,
        account_id: int | None = None,
    ) -> dict[str, Any]:
        clauses = ["s.courseid = %s"]
        params: list[Any] = [course_id]
        if date_from:
            clauses.append("s.start_time >= %s")
            params.append(date_from)
        if date_to:
            clauses.append("s.start_time <= %s")
            params.append(date_to)
        if account_id is not None:
            clauses.append("r.accountid = %s")
            params.append(account_id)
        where = " AND ".join(clauses)
        trend_sql = f"""
            SELECT s.start_time::date AS bucket,
                   COUNT(*) FILTER (WHERE r.status='present') AS present,
                   COUNT(*) FILTER (WHERE r.status='late')    AS late,
                   COUNT(*) FILTER (WHERE r.status='absent')  AS absent,
                   COUNT(r.attendancerecordid) AS total
            FROM attendance_session s
            LEFT JOIN attendance_record r ON r.attendancesessionid = s.attendancesessionid
            WHERE {where}
            GROUP BY bucket ORDER BY bucket
        """
        breakdown_sql = f"""
            SELECT COUNT(*) FILTER (WHERE r.status='present') AS present,
                   COUNT(*) FILTER (WHERE r.status='late')    AS late,
                   COUNT(*) FILTER (WHERE r.status='absent')  AS absent,
                   COUNT(r.attendancerecordid) AS total
            FROM attendance_session s
            LEFT JOIN attendance_record r ON r.attendancesessionid = s.attendancesessionid
            WHERE {where}
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(trend_sql, params)
            trend = _dict_rows(cur)
            cur.execute(breakdown_sql, params)
            breakdown = _dict_rows(cur)[0] if cur.description else {}
        for row in trend:
            total = row.get("total") or 0
            attended = (row.get("present") or 0) + (row.get("late") or 0)
            row["rate"] = round(attended / total * 100, 1) if total else 0.0
        total = breakdown.get("total") or 0
        attended = (breakdown.get("present") or 0) + (breakdown.get("late") or 0)
        breakdown["rate"] = round(attended / total * 100, 1) if total else 0.0
        return {"success": True, "trend": trend, "breakdown": breakdown}

    # ── U12 viewRealTimeAttendanceStatus ─────────────────────────────
    def viewRealTimeAttendanceStatus(self, session_id: int) -> dict[str, Any]:
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                """
                SELECT s.attendancesessionid, s.courseid, s.start_time, s.end_time,
                       s.status, c.course_code, c.course_name
                FROM attendance_session s
                JOIN course c ON c.courseid = s.courseid
                WHERE s.attendancesessionid = %s
                """, (session_id,),
            )
            srow = cur.fetchone()
            if not srow:
                raise HTTPException(404, "Session not found")
            cols = [d[0] for d in cur.description]
            session = dict(zip(cols, srow))
            cur.execute(
                """
                SELECT e.accountid, pi.full_name, pi.student_id,
                       r.status AS attendance_status, r.marked_at
                FROM course_enrollment e
                LEFT JOIN personal_info pi ON pi.accountid = e.accountid
                LEFT JOIN attendance_record r
                  ON r.attendancesessionid = %s AND r.accountid = e.accountid
                WHERE e.courseid = %s AND e.status = 'active'
                ORDER BY pi.full_name NULLS LAST, e.accountid
                """, (session_id, session["courseid"]),
            )
            roster = _dict_rows(cur)
        summary = {"present": 0, "late": 0, "early_left": 0,
                   "absent": 0, "leave": 0, "no_record": 0}
        for r in roster:
            st = r.get("attendance_status")
            summary[st if st in summary else "no_record"] += 1
        return {"success": True, "session": session, "roster": roster, "summary": summary}

    # ── U13 viewAttendanceHistoryAcrossAllSession ────────────────────
    def viewAttendanceHistoryAcrossAllSession(
        self, account_id: int, course_id: int | None = None,
    ) -> dict[str, Any]:
        clauses = ["r.accountid = %s"]
        params: list[Any] = [account_id]
        if course_id is not None:
            clauses.append("s.courseid = %s")
            params.append(course_id)
        where = " AND ".join(clauses)
        sql = f"""
            SELECT r.attendancerecordid AS record_id,
                   r.attendancesessionid AS session_id,
                   c.course_code, c.course_name,
                   s.start_time, s.end_time, s.status AS session_status,
                   r.status, r.marked_at
            FROM attendance_record r
            JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            WHERE {where}
            ORDER BY s.start_time DESC
        """
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(sql, params)
            rows = _dict_rows(cur)
        # Count every status so the summary total matches the listed rows
        # (U03 can produce early_left; approved leave produces 'leave').
        summary = {"present": 0, "late": 0, "early_left": 0, "absent": 0, "leave": 0}
        for r in rows:
            st = r["status"]
            if st in summary:
                summary[st] += 1
        total = len(rows)
        attended = summary["present"] + summary["late"] + summary["early_left"]
        # Approved leave is excused — exclude it from the rate denominator.
        rate_total = total - summary["leave"]
        rate = round(attended / rate_total * 100, 1) if rate_total else 0.0
        return {
            "success": True, "records": rows,
            "summary": summary, "total": total, "rate": rate,
        }

    # ── U16 viewEarlyLeftSummary ─────────────────────────────────────
    def viewEarlyLeftSummary(self, session_id: int) -> dict[str, Any]:
        """Students whose check-in exists but were missing from the
        periodic in-class scan (status='early_left' in attendance_record,
        OR a presence_check row marking them absent after check-in)."""
        sql = """
            SELECT r.accountid, pi.full_name, pi.student_id,
                   r.status AS attendance_status, r.marked_at,
                   MAX(pc.detected_at) AS last_seen
            FROM attendance_record r
            LEFT JOIN personal_info pi ON pi.accountid = r.accountid
            LEFT JOIN presence_check pc
              ON pc.attendancesessionid = r.attendancesessionid
             AND pc.accountid = r.accountid
             AND pc.detected = TRUE
            WHERE r.attendancesessionid = %s
              AND (r.status = 'early_left'
                   OR (r.status = 'present' AND EXISTS (
                       SELECT 1 FROM presence_check pc2
                       WHERE pc2.attendancesessionid = r.attendancesessionid
                         AND pc2.accountid = r.accountid
                         AND pc2.detected = FALSE
                   )))
            GROUP BY r.accountid, pi.full_name, pi.student_id,
                     r.status, r.marked_at
            ORDER BY pi.full_name NULLS LAST
        """
        # presence_check is optional; fall back to empty list on schema mismatch.
        try:
            with _db(self.database_url) as c, c.cursor() as cur:
                cur.execute(sql, (session_id,))
                rows = _dict_rows(cur)
            return {"success": True, "early_left": rows}
        except psycopg2.errors.UndefinedTable:
            return {"success": True, "early_left": [],
                    "note": "presence_check table not yet created"}
        except Exception:
            return {"success": True, "early_left": []}

    # ── U34 configureAbsenceThreshold (DB-backed) ────────────────────
    def configureAbsenceThreshold(
        self,
        consecutive_threshold: int | None = None,
        minimum_rate: float | None = None,
        late_grace_seconds: int | None = None,
        detection_interval_seconds: int | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        """Persist threshold updates to attendance_threshold_config (U34)."""
        if consecutive_threshold is not None:
            if consecutive_threshold < 1 or consecutive_threshold > 100:
                raise HTTPException(400, "consecutive_threshold must be between 1 and 100")
        if minimum_rate is not None:
            if minimum_rate < 0 or minimum_rate > 100:
                raise HTTPException(400, "minimum_rate must be between 0 and 100")
        if late_grace_seconds is not None:
            if late_grace_seconds < 0 or late_grace_seconds > 86400:
                raise HTTPException(400, "late_grace_seconds must be between 0 and 86400")
        if detection_interval_seconds is not None:
            if detection_interval_seconds < 3 or detection_interval_seconds > 86400:
                raise HTTPException(400, "detection_interval_seconds must be between 3 and 86400")

        # Build a parameterised UPDATE so we only touch the supplied fields.
        sets, params = [], []
        if consecutive_threshold is not None:
            sets.append("absence_threshold = %s")
            params.append(int(consecutive_threshold))
        if minimum_rate is not None:
            sets.append("minimum_attendance_rate = %s")
            params.append(float(minimum_rate))
        if late_grace_seconds is not None:
            sets.append("late_grace_seconds = %s")
            params.append(int(late_grace_seconds))
        if detection_interval_seconds is not None:
            sets.append("detection_interval_seconds = %s")
            params.append(int(detection_interval_seconds))
        if updated_by is not None:
            sets.append("updated_by = %s")
            params.append(updated_by)
        if sets:
            sets.append("updated_at = NOW()")
            try:
                with _db(self.database_url) as c, c.cursor() as cur:
                    # Make sure the singleton row exists (idempotent insert).
                    cur.execute(
                        """INSERT INTO attendance_threshold_config (configid)
                           VALUES (1) ON CONFLICT DO NOTHING"""
                    )
                    params.append(1)
                    cur.execute(
                        f"UPDATE attendance_threshold_config "
                        f"SET {', '.join(sets)} WHERE configid = %s",
                        params,
                    )
            except psycopg2.errors.UndefinedTable:
                raise HTTPException(
                    503, "attendance_threshold_config table does not exist; please run schema.sql first",
                )

        # Refresh instance attributes from DB so callers see fresh state.
        cfg = load_threshold_config(self.database_url)
        self.minimumAttendanceRateRequirement = cfg["minimum_attendance_rate"]
        self.absenceThreshold = cfg["absence_threshold"]
        return {
            "success": True,
            "absence_threshold": cfg["absence_threshold"],
            "minimum_attendance_rate": cfg["minimum_attendance_rate"],
            "late_grace_seconds": cfg["late_grace_seconds"],
            "detection_interval_seconds": cfg["detection_interval_seconds"],
        }


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["attendanceRecord"])


def _svc(request: Request) -> AttendanceRecord:
    return AttendanceRecord(
        request.app.state.pipeline,
        request.app.state.cfg.database_url,
    )


# Student: U03 — automated check-in is teacher-driven; the student only views
# which of their sessions are currently being scanned and their live status.
@router.get("/student/sessions/active")
def student_active_sessions(
    request: Request, user: CurrentUser = Depends(require_role("student"))
):
    return _svc(request).viewActiveSessions(user.account_id)


# Student: U28 helper — upcoming scheduled sessions to request leave for.
@router.get("/student/sessions/upcoming")
def student_upcoming_sessions(
    request: Request, user: CurrentUser = Depends(require_role("student"))
):
    return _svc(request).viewUpcomingSessions(user.account_id)


# Student: U04
@router.get("/student/attendance")
def student_attendance(
    request: Request, user: CurrentUser = Depends(require_role("student"))
):
    return _svc(request).viewAttendanceRecord(user.account_id)


# Student: U27
@router.get("/student/analytics")
def student_analytics(
    request: Request,
    date_from: str | None = None,
    date_to: str | None = None,
    user: CurrentUser = Depends(require_role("student")),
):
    return _svc(request).viewStudentGraphicalReport(user.account_id, date_from, date_to)


# Teacher: U03 — record one detection-window snapshot for an active session.
# The teacher's device (or demo webcam) uploads a frame; the system runs the
# recognition pipeline and stores one presence_check row per enrolled student.
@router.post("/teacher/sessions/{session_id}/scan")
async def teacher_scan_snapshot(
    session_id: int,
    request: Request,
    file: UploadFile = File(...),
    camera_id: str | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    img = await _bytes_to_cv2(file)
    return _svc(request).recordDetectionSnapshot(session_id, img, camera_id)


# Teacher: U12 — live roster
@router.get("/teacher/sessions/{session_id}/live")
def teacher_live_roster(
    session_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewRealTimeAttendanceStatus(session_id)


# Teacher: U13 — per-student history
@router.get("/teacher/students/{account_id}/attendance")
def teacher_student_history(
    account_id: int,
    request: Request,
    course_id: int | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewAttendanceHistoryAcrossAllSession(account_id, course_id)


# Teacher: U16 — early-left summary
@router.get("/teacher/sessions/{session_id}/early-left")
def teacher_early_left(
    session_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewEarlyLeftSummary(session_id)


# Teacher: U30 — class analytics
@router.get("/teacher/courses/{course_id}/analytics")
def teacher_class_analytics(
    course_id: int,
    request: Request,
    date_from: str | None = None,
    date_to: str | None = None,
    account_id: int | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewTeacherGraphicalReport(
        course_id, date_from, date_to, account_id
    )


# Shared: current attendance config (teachers read detection interval for the
# scan UI; admins read everything for the config form).
@router.get("/config/attendance")
def get_attendance_config(
    request: Request,
    user: CurrentUser = Depends(require_role("teacher", "admin")),
):
    cfg = load_threshold_config(request.app.state.cfg.database_url)
    return {"success": True, **cfg}


# Admin: U34
class ThresholdBody(BaseModel):
    consecutive_threshold: int | None = None
    minimum_rate: float | None = None
    late_grace_seconds: int | None = None
    detection_interval_seconds: int | None = None


@router.patch("/admin/config/absence-threshold")
def admin_configure_absence_threshold(
    body: ThresholdBody,
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).configureAbsenceThreshold(
        body.consecutive_threshold, body.minimum_rate,
        body.late_grace_seconds, body.detection_interval_seconds,
        updated_by=user.account_id,
    )


@router.get("/admin/attendance")
def admin_list_attendance(
    request: Request, user: CurrentUser = Depends(require_role("admin"))
):
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
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql)
        return {"success": True, "records": _dict_rows(cur)}
