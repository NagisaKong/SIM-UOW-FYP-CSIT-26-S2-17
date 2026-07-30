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
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel

from core.userInformation import (
    CurrentUser,
    assert_teacher_course,
    assert_teacher_session,
    require_role,
    teacher_course_ids,
)

# Default fall-back values used when the attendance_threshold_config
# table is empty or unreachable (these mirror the row inserted by
# database/schema.sql).
_DEFAULT_MIN_RATE = 70.0
_DEFAULT_ABSENCE_THRESHOLD = 3
_DEFAULT_LATE_GRACE = 600
_DEFAULT_DETECTION_INTERVAL = 1200  # 20 min between detection windows (U03)
_DEFAULT_MINIMUM_PRESENCE_RATIO = 50.0  # within-session presence floor
_DEFAULT_TAIL_RATIO = 20.0  # fraction of final snapshots checked for early_left


def load_threshold_config(database_url: str) -> dict[str, Any]:
    """Read the single-row attendance_threshold_config table.

    Falls back to module defaults if the table is missing or empty so
    the system keeps working even before the v0.5 migration runs.
    """
    try:
        with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT minimum_attendance_rate, absence_threshold,
                          late_grace_seconds, detection_interval_seconds,
                          minimum_presence_ratio, tail_ratio
                   FROM attendance_threshold_config WHERE configid = 1"""
            )
            row = cur.fetchone()
            if row:
                return {
                    "minimum_attendance_rate": float(row[0]),
                    "absence_threshold": int(row[1]),
                    "late_grace_seconds": int(row[2]),
                    "detection_interval_seconds": int(row[3]),
                    "minimum_presence_ratio": float(row[4]),
                    "tail_ratio": float(row[5]),
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
        "minimum_presence_ratio": _DEFAULT_MINIMUM_PRESENCE_RATIO,
        "tail_ratio": _DEFAULT_TAIL_RATIO,
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


# ── Match diagnostics ────────────────────────────────────────────────
# The primary recogniser's runner-up scores. Surfacing them turns "Unknown"
# from an opaque result into a debuggable one: a top score of 0.38 against a
# 0.43 threshold is a threshold problem, whereas 0.45 vs 0.44 is the margin
# rule correctly refusing to guess between two look-alikes.
_PRIMARY_MODEL = "arcface"


def _primary(prediction) -> dict[str, Any]:
    per_model = getattr(prediction, "per_model", None) or {}
    return per_model.get(_PRIMARY_MODEL) or next(iter(per_model.values()), {})


def _primary_model_name(prediction) -> str | None:
    """Which model the diagnostic candidates/threshold above actually came
    from. Usually "arcface" — but ArcFace only gets a free embedding when
    SCRFD supplied the anchor detection (see _group_anchors); if SCRFD missed
    a face that MTCNN caught, ArcFace has to re-detect on a cropped/padded
    region and can come back empty, silently leaving FaceNet — the weaker,
    secondary model — as the only one that voted. Without surfacing which
    model this was, "0.70 / 0.71 (need 0.55)" reads as ArcFace failing to
    separate two people it actually separates cleanly; it is FaceNet's own,
    less discriminative comparison standing in for ArcFace's absence.
    """
    per_model = getattr(prediction, "per_model", None) or {}
    if _PRIMARY_MODEL in per_model:
        return _PRIMARY_MODEL
    return next(iter(per_model), None)


def _match_candidates(prediction) -> list[dict[str, Any]]:
    return _primary(prediction).get("candidates", [])


def _match_threshold(prediction) -> float | None:
    return _primary(prediction).get("threshold")


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
              AND s.start_time > NOW()
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
            if self.pipeline is None:
                raise HTTPException(
                    503, "Face recognition is unavailable (AI pipeline failed to load)",
                )

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

        # Per-face boxes (for the teacher's live overlay during testing).
        # bbox is in pixel coords of the uploaded frame; the frontend scales
        # it to the displayed video using frame_width/frame_height.
        boxes = []
        for p in result.predictions:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            label = (p.full_name or p.student_id or f"acc#{p.account_id}") if p.recognised else "Unknown"
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "recognised": bool(p.recognised),
                "account_id": p.account_id if p.recognised else None,
                "label": label,
                "score": round(float(p.score), 3),
                "candidates": _match_candidates(p),
                "threshold": _match_threshold(p),
                "diagnostic_model": _primary_model_name(p),
            })
        frame_h, frame_w = (int(image.shape[0]), int(image.shape[1])) if image is not None else (0, 0)

        return {
            "success": True,
            "session_id": session_id,
            "enrolled": len(roster),
            "faces_in_frame": faces,
            "detected_count": len(detected),
            "detected": detected,
            "boxes": boxes,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "message": f"Snapshot recorded: {len(detected)}/{len(roster)} enrolled student(s) detected "
                       f"({faces} face(s) in frame)",
        }

    # ── Live preview (detection only — no DB write) ──────────────────
    # Runs the recognition pipeline and returns face boxes so the teacher's
    # UI can draw them in near-real-time, WITHOUT recording any presence_check
    # rows. Safe to poll frequently; attendance is still driven by the proper
    # scan snapshots (recordDetectionSnapshot).
    def detectFacesPreview(self, image: np.ndarray) -> dict[str, Any]:
        if self.pipeline is None:
            raise HTTPException(
                503, "Face recognition is unavailable (AI pipeline failed to load)",
            )
        result = self.pipeline.process_frame(image)
        boxes = []
        for p in result.predictions:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            label = (p.full_name or p.student_id or f"acc#{p.account_id}") if p.recognised else "Unknown"
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "recognised": bool(p.recognised),
                "account_id": p.account_id if p.recognised else None,
                "label": label,
                "score": round(float(p.score), 3),
                # Runners-up, so an operator can see WHY a face came back
                # Unknown (below threshold vs. too close to call) instead of
                # guessing. Diagnostic only — never used for attendance.
                "candidates": _match_candidates(p),
                "threshold": _match_threshold(p),
                "diagnostic_model": _primary_model_name(p),
            })
        frame_h, frame_w = (int(image.shape[0]), int(image.shape[1])) if image is not None else (0, 0)
        return {
            "success": True,
            "faces_in_frame": len(result.predictions),
            "boxes": boxes,
            "frame_width": frame_w,
            "frame_height": frame_h,
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
                   c.courseid, c.course_code, c.course_name,
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
        course_id: int | None = None,
    ) -> dict[str, Any]:
        clauses = ["r.accountid = %s"]
        params: list[Any] = [account_id]
        # U27: the student may narrow the view to a single module (course).
        if course_id is not None:
            clauses.append("s.courseid = %s")
            params.append(course_id)
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
                   -- early_left / leave are real statuses that land in the
                   -- same total; counting them keeps the parts summing to the
                   -- whole instead of silently vanishing from the breakdown.
                   COUNT(*) FILTER (WHERE r.status='early_left') AS early_left,
                   COUNT(*) FILTER (WHERE r.status='leave')      AS "leave",
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
                   -- early_left / leave are real statuses that land in the
                   -- same total; counting them keeps the parts summing to the
                   -- whole instead of silently vanishing from the breakdown.
                   COUNT(*) FILTER (WHERE r.status='early_left') AS early_left,
                   COUNT(*) FILTER (WHERE r.status='leave')      AS "leave",
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
                   -- early_left / leave are real statuses that land in the
                   -- same total; counting them keeps the parts summing to the
                   -- whole instead of silently vanishing from the breakdown.
                   COUNT(*) FILTER (WHERE r.status='early_left') AS early_left,
                   COUNT(*) FILTER (WHERE r.status='leave')      AS "leave",
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
                   -- early_left / leave are real statuses that land in the
                   -- same total; counting them keeps the parts summing to the
                   -- whole instead of silently vanishing from the breakdown.
                   COUNT(*) FILTER (WHERE r.status='early_left') AS early_left,
                   COUNT(*) FILTER (WHERE r.status='leave')      AS "leave",
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
        course_scope: list[int] | None = None,
    ) -> dict[str, Any]:
        """History for one student.

        ``course_scope`` restricts the result to a set of courses — the caller
        passes the requesting teacher's own courses so the summary and rate
        describe that teacher's classes rather than the student's institution-
        wide record. An empty list is a real answer (this teacher has no
        courses), not "no filter", so it must still be applied.
        """
        clauses = ["r.accountid = %s"]
        params: list[Any] = [account_id]
        if course_id is not None:
            clauses.append("s.courseid = %s")
            params.append(course_id)
        elif course_scope is not None:
            clauses.append("s.courseid = ANY(%s)")
            params.append(list(course_scope))
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
        """Students whose finalised status is 'early_left' for this session.

        This is a post-session view: the status is assigned by
        AttendanceSession._aggregate_presence when the teacher ends the scan,
        using the tail-window rule (detected earlier, then absent from every
        one of the last `tail_ratio`% of snapshots).

        An earlier version also listed any student still marked 'present' who
        had ever missed a single snapshot. That is a far looser test than the
        one that actually assigns early_left, so the panel routinely flagged
        students who were present the whole class and merely went undetected
        once — contradicting the finalised record shown right next to it.
        `last_seen` is still the timestamp of their final positive detection.
        """
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
              AND r.status = 'early_left'
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
        minimum_presence_ratio: float | None = None,
        tail_ratio: float | None = None,
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
        if minimum_presence_ratio is not None:
            if minimum_presence_ratio < 0 or minimum_presence_ratio > 100:
                raise HTTPException(400, "minimum_presence_ratio must be between 0 and 100")
        if tail_ratio is not None:
            if tail_ratio < 0 or tail_ratio > 100:
                raise HTTPException(400, "tail_ratio must be between 0 and 100")

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
        if minimum_presence_ratio is not None:
            sets.append("minimum_presence_ratio = %s")
            params.append(float(minimum_presence_ratio))
        if tail_ratio is not None:
            sets.append("tail_ratio = %s")
            params.append(float(tail_ratio))
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
            "minimum_presence_ratio": cfg["minimum_presence_ratio"],
            "tail_ratio": cfg["tail_ratio"],
        }


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["attendanceRecord"])


def _svc(request: Request) -> AttendanceRecord:
    return AttendanceRecord(
        getattr(request.app.state, "pipeline", None),
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
    course_id: int | None = None,
    user: CurrentUser = Depends(require_role("student")),
):
    return _svc(request).viewStudentGraphicalReport(
        user.account_id, date_from, date_to, course_id
    )


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
    assert_teacher_session(
        request.app.state.cfg.database_url, user.account_id, session_id
    )
    img = await _bytes_to_cv2(file)
    return _svc(request).recordDetectionSnapshot(session_id, img, camera_id)


# Teacher: live preview — detection-only, returns face boxes without recording
# any attendance. Used by the UI to draw real-time boxes; poll-friendly.
@router.post("/teacher/preview-detect")
async def teacher_preview_detect(
    request: Request,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_role("teacher")),
):
    img = await _bytes_to_cv2(file)
    return _svc(request).detectFacesPreview(img)


# Teacher: U12 — live roster
@router.get("/teacher/sessions/{session_id}/live")
def teacher_live_roster(
    session_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(require_role("teacher")),
):
    # Imported here rather than at module scope: attendanceSession already
    # imports this module for load_threshold_config, so a top-level import
    # would close the cycle.
    from core.attendanceSession import expire_overdue_sessions

    assert_teacher_session(
        request.app.state.cfg.database_url, user.account_id, session_id
    )
    # The dashboard the teacher is watching should settle by itself the
    # moment the class runs past its scheduled end.
    expire_overdue_sessions(request.app.state.cfg.database_url, background_tasks)
    return _svc(request).viewRealTimeAttendanceStatus(session_id)


# Teacher: U13 — per-student history
@router.get("/teacher/students/{account_id}/attendance")
def teacher_student_history(
    account_id: int,
    request: Request,
    course_id: int | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    database_url = request.app.state.cfg.database_url
    # A teacher sees a student's history *in their own classes*, not the
    # student's record across the whole institution.
    if course_id is not None:
        assert_teacher_course(database_url, user.account_id, course_id)
        scope = None
    else:
        scope = teacher_course_ids(database_url, user.account_id)
    return _svc(request).viewAttendanceHistoryAcrossAllSession(
        account_id, course_id, course_scope=scope
    )


# Teacher: U16 — early-left summary
@router.get("/teacher/sessions/{session_id}/early-left")
def teacher_early_left(
    session_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    assert_teacher_session(
        request.app.state.cfg.database_url, user.account_id, session_id
    )
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
    assert_teacher_course(
        request.app.state.cfg.database_url, user.account_id, course_id
    )
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
    minimum_presence_ratio: float | None = None
    tail_ratio: float | None = None


@router.patch("/admin/config/absence-threshold")
def admin_configure_absence_threshold(
    body: ThresholdBody,
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).configureAbsenceThreshold(
        body.consecutive_threshold, body.minimum_rate,
        body.late_grace_seconds, body.detection_interval_seconds,
        body.minimum_presence_ratio, body.tail_ratio,
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
