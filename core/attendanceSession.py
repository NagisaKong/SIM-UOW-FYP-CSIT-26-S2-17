"""Class 5: attendanceSession

Covers UC:
  U07 View Session Attendance Details (viewSessionDetail)
  U15 Start / End Attendance Session  (startSession / endSession)
  U26 Manage Courses                  (admin CRUD on courses + enrolments + session scheduling)

Attributes: sessionDetail* (start_time, end_time, status, courseid, …)
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import psycopg2
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from core.attendanceRecord import load_threshold_config
from core.notification import Notification, send_late_absent_emails
from core.userInformation import (
    CurrentUser,
    assert_teacher_course,
    assert_teacher_session,
    require_role,
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


def purge_expired_recordings(database_url: str) -> dict[str, int]:
    """U03 retention: delete class recordings (and detection rows) older
    than their 30-day expiry. Best-effort — safe to call at startup."""
    import os

    removed_files = 0
    deleted_rows = 0
    try:
        with _db(database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT recordingid, file_path FROM session_recording WHERE expires_at < NOW()"
            )
            expired = cur.fetchall()
            for _rid, path in expired:
                if path and os.path.isfile(path):
                    with contextlib.suppress(OSError):
                        os.remove(path)
                        removed_files += 1
            cur.execute("DELETE FROM session_recording WHERE expires_at < NOW()")
            deleted_rows = cur.rowcount
            # Detection snapshots are only needed while the recording exists.
            cur.execute(
                "DELETE FROM presence_check WHERE detected_at < NOW() - INTERVAL '30 days'"
            )
    except (psycopg2.errors.UndefinedTable, psycopg2.OperationalError):
        pass
    return {"recordings_deleted": deleted_rows, "files_removed": removed_files}


# Overdue-session sweep. Read endpoints call this, so it runs on the teacher's
# 5-second dashboard poll as well; the throttle keeps that from turning into a
# query storm without needing a scheduler process.
_EXPIRY_SWEEP_INTERVAL_SECONDS = 60.0
_last_expiry_sweep = 0.0


def expire_overdue_sessions(
    database_url: str,
    background_tasks: BackgroundTasks | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Close out sessions whose scheduled end_time has passed.

    A session used to stay `active` forever if the teacher never pressed
    "End Scan & Finalize" — it kept showing up as selectable long after the
    class was over, and no student ever got a final status for it.

    Two cases, deliberately handled differently:

    * `active`   — the scan did run, so finalise it exactly as the teacher
      would have: aggregate the presence_check snapshots into per-student
      statuses and send the U05 notifications.
    * `scheduled` — the scan never started, so there is no attendance
      evidence at all. Marking the whole class absent for a lesson nobody
      recorded would both misrepresent the students and trip the U29
      long-term-absence reminders, so the session is cancelled instead.

    Best-effort: one failing session must not stop the others, and the sweep
    must never take down the request that triggered it.
    """
    import time

    global _last_expiry_sweep

    now = time.time()
    if not force and now - _last_expiry_sweep < _EXPIRY_SWEEP_INTERVAL_SECONDS:
        return {"finalized": 0, "cancelled": 0, "skipped": 0}
    _last_expiry_sweep = now

    finalized = cancelled = skipped = 0
    try:
        with _db(database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT attendancesessionid, status
                   FROM attendance_session
                   WHERE status IN ('scheduled', 'active')
                     AND end_time IS NOT NULL
                     AND end_time < NOW()"""
            )
            overdue = cur.fetchall()
    except (psycopg2.errors.UndefinedTable, psycopg2.OperationalError):
        return {"finalized": 0, "cancelled": 0, "skipped": 0}

    svc = AttendanceSession(database_url)
    for session_id, status in overdue:
        try:
            if status == "active":
                svc.endSession(session_id, background_tasks)
                finalized += 1
            else:
                with _db(database_url) as c, c.cursor() as cur:
                    cur.execute(
                        """UPDATE attendance_session SET status = 'cancelled'
                           WHERE attendancesessionid = %s AND status = 'scheduled'""",
                        (session_id,),
                    )
                cancelled += 1
        except Exception as exc:  # noqa: BLE001 — never break the caller
            skipped += 1
            print(f"[expiry] session {session_id} ({status}) not closed: {exc}")
    if finalized or cancelled:
        print(f"[expiry] finalized {finalized}, cancelled {cancelled} overdue session(s)")
    return {"finalized": finalized, "cancelled": cancelled, "skipped": skipped}


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
                """SELECT courseid, status, end_time IS NOT NULL AND end_time < NOW()
                   FROM attendance_session WHERE attendancesessionid = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Session not found")
            course_id, current_status, overdue = row
            if current_status == "active":
                raise HTTPException(409, "This session is already in progress")
            if current_status == "ended":
                raise HTTPException(409, "This session has ended and cannot be started again")
            if current_status == "cancelled":
                raise HTTPException(409, "This session has been cancelled")
            # Closes the gap between the expiry sweep and this click: without
            # it, a session that went overdue seconds ago could still be
            # started and would then be cancelled out from under the teacher.
            if overdue:
                raise HTTPException(
                    409, "This session's scheduled end time has already passed",
                )
            cur.execute(
                """SELECT 1 FROM attendance_session
                   WHERE courseid = %s AND status = 'active'
                     AND attendancesessionid <> %s LIMIT 1""",
                (course_id, session_id),
            )
            if cur.fetchone():
                raise HTTPException(409, "This course already has a session in progress")
            cur.execute(
                """UPDATE attendance_session
                   SET status = 'active',
                       start_time = CASE WHEN start_time > NOW() THEN NOW() ELSE start_time END
                   WHERE attendancesessionid = %s""",
                (session_id,),
            )
            # U03: the class recording begins when the scan is activated.
            # Tracked here with a 30-day expiry (file written by the capture
            # device). Best-effort: skip silently if the table is absent.
            with contextlib.suppress(psycopg2.errors.UndefinedTable):
                cur.execute(
                    """INSERT INTO session_recording (attendancesessionid, file_path)
                       VALUES (%s, %s)
                       ON CONFLICT (attendancesessionid) DO NOTHING""",
                    (session_id, f"recordings/session_{session_id}.mp4"),
                )
        return {"success": True, "session_id": session_id, "status": "active"}

    # ── U15 endSession (U03 aggregation) ─────────────────────────────
    # When the teacher ends the scan, aggregate every detection snapshot
    # (presence_check rows) into ONE final status per enrolled student:
    #   Absent     — never detected in any snapshot (or no scan ran).
    #   Left Early — detected earlier but missing from the final snapshot
    #                (signed-in-then-disappeared / tail gap).
    #   Late       — missing from the first snapshot but detected later.
    #   Present    — detected in both the first and the last snapshot.
    # An approved leave ('leave') is never overwritten.
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
                raise HTTPException(404, "Session not found")
            course_id, current_status = row
            if current_status == "ended":
                raise HTTPException(409, "This session has already ended")
            if current_status == "cancelled":
                raise HTTPException(409, "This session has been cancelled")

            # Active roster for the course.
            cur.execute(
                """SELECT e.accountid FROM course_enrollment e
                   WHERE e.courseid = %s AND e.status = 'active'""",
                (course_id,),
            )
            roster = [r[0] for r in cur.fetchall()]

            thresholds = load_threshold_config(self.database_url)
            final_status = self._aggregate_presence(
                cur, session_id, roster,
                late_grace_seconds=thresholds["late_grace_seconds"],
                minimum_presence_ratio=thresholds["minimum_presence_ratio"],
                tail_ratio=thresholds["tail_ratio"],
            )

            counts = {"present": 0, "late": 0, "early_left": 0, "absent": 0}
            for acc in roster:
                st = final_status.get(acc, "absent")
                counts[st] = counts.get(st, 0) + 1
                # Upsert; never clobber an approved leave.
                cur.execute(
                    """INSERT INTO attendance_record (attendancesessionid, accountid, status)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (attendancesessionid, accountid)
                       DO UPDATE SET status = EXCLUDED.status, marked_at = NOW()
                       WHERE attendance_record.status <> 'leave'""",
                    (session_id, acc, st),
                )

            cur.execute(
                """UPDATE attendance_session
                   SET status = 'ended',
                       end_time = COALESCE(end_time, NOW())
                   WHERE attendancesessionid = %s""",
                (session_id,),
            )
        recipients = self._fetch_late_absent_recipients(session_id)
        # U29 (FTD): the long-term-absence check runs automatically after
        # every completed session, using the U34-configured thresholds.
        reminder_scan = Notification(self.database_url).generateLongTermAbsenceNotification
        if background_tasks is not None:
            background_tasks.add_task(send_late_absent_emails, recipients)
            background_tasks.add_task(reminder_scan)
            queued = len(recipients)
        else:
            send_late_absent_emails(recipients)
            with contextlib.suppress(Exception):
                reminder_scan()
            queued = len(recipients)
        return {
            "success": True, "session_id": session_id,
            "present": counts["present"], "late": counts["late"],
            "early_left": counts["early_left"], "marked_absent": counts["absent"],
            "notifications_queued": queued,
        }

    # ── internal: snapshot aggregation (U03) ─────────────────────────
    def _aggregate_presence(
        self, cur, session_id: int, roster: list[int],
        late_grace_seconds: int = 0,
        minimum_presence_ratio: float = 50.0,
        tail_ratio: float = 20.0,
    ) -> dict[int, str]:
        """Return {account_id: final_status} from presence_check snapshots.

        A "snapshot" is a distinct detected_at timestamp (one detection
        window writes all its rows in a single transaction).

        Rules (per enrolled student), checked in this order:
          1. never detected in any snapshot
                 → absent (also when no scan ran)
          2. detected, but detected in fewer than ``minimum_presence_ratio``%
             (floor-rounded) of the snapshots taken from their OWN first
             detection onward — i.e. even restricted to the snapshots that
             could plausibly have caught them, they were still mostly missed
                 → absent
          3. detected, meets the presence-ratio floor, but ABSENT FROM EVERY
             ONE of the last ``tail_ratio``% of the session's snapshots
             (floor-rounded, at least 1)
                 → early_left (signed in, then left before the end)
          4. first detected later than (scan start + late_grace_seconds)
                 → late
          5. otherwise
                 → present

        Replaces an earlier version of this rule that only ever looked at
        the single very-last snapshot to decide early_left: one missed
        detection on the final snapshot was enough to brand someone who
        was present almost the entire class as early_left, while someone
        who left mid-class but happened to be caught on the last snapshot
        was not flagged at all. Using a floor-rounded percentage of the
        tail, plus an overall presence-ratio floor, absorbs a single missed
        detection without losing the ability to catch a genuine early exit.

        The presence-ratio denominator (step 2) is counted only from the
        student's own first detection onward, not the whole session, so a
        student who simply arrived late is not double-penalised for
        snapshots taken before they showed up — that case is instead
        handled by the separate ``late`` check in step 4.

        ``late_grace_seconds``, ``minimum_presence_ratio`` and
        ``tail_ratio`` are all admin-configurable (U34).
        """
        try:
            cur.execute(
                """SELECT accountid, detected_at, BOOL_OR(detected) AS seen
                   FROM presence_check
                   WHERE attendancesessionid = %s
                   GROUP BY accountid, detected_at""",
                (session_id,),
            )
            rows = cur.fetchall()
        except psycopg2.errors.UndefinedTable:
            rows = []

        # Ordered list of distinct snapshot timestamps.
        snapshots = sorted({r[1] for r in rows})
        if not snapshots:
            return {acc: "absent" for acc in roster}

        scan_start = snapshots[0]
        tail_count = max(1, math.floor(len(snapshots) * (tail_ratio / 100.0)))
        tail_snapshots = set(snapshots[-tail_count:])

        first_seen: dict[int, Any] = {}
        detected_snapshots: dict[int, set] = {}
        for acc, ts, seen in rows:
            if not seen:
                continue
            detected_snapshots.setdefault(acc, set()).add(ts)
            if acc not in first_seen or ts < first_seen[acc]:
                first_seen[acc] = ts

        result: dict[int, str] = {}
        for acc in roster:
            if acc not in first_seen:
                result[acc] = "absent"
                continue

            applicable = [s for s in snapshots if s >= first_seen[acc]]
            required = math.floor(len(applicable) * (minimum_presence_ratio / 100.0))
            detected_count = len(detected_snapshots[acc])

            if detected_count < required:
                result[acc] = "absent"
            elif not (detected_snapshots[acc] & tail_snapshots):
                result[acc] = "early_left"
            elif (first_seen[acc] - scan_start).total_seconds() > late_grace_seconds:
                result[acc] = "late"
            else:
                result[acc] = "present"
        return result

    # ── internal ─────────────────────────────────────────────────────
    def _fetch_late_absent_recipients(self, session_id: int) -> list[dict[str, Any]]:
        sql = """
            SELECT ua.email, pi.full_name, r.status,
                   c.course_code, c.course_name, s.start_time
            FROM attendance_record r
            JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            JOIN user_account ua ON ua.accountid = r.accountid
            -- Only currently-enrolled, active accounts get emailed:
            -- skip students who have since withdrawn or been deactivated
            -- (a stale late/absent record must not trigger a notification).
            JOIN course_enrollment e
              ON e.accountid = r.accountid AND e.courseid = c.courseid
             AND e.status = 'active'
            LEFT JOIN personal_info pi ON pi.accountid = r.accountid
            WHERE r.attendancesessionid = %s
              AND r.status IN ('late', 'absent')
              AND ua.status = 'active'
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
    assert_teacher_session(
        request.app.state.cfg.database_url, user.account_id, session_id
    )
    return _svc(request).startSession(session_id)


@router.post("/teacher/sessions/{session_id}/end")
def teacher_end_session(
    session_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    assert_teacher_session(
        request.app.state.cfg.database_url, user.account_id, session_id
    )
    return _svc(request).endSession(session_id, background_tasks)


# Teacher: list sessions / courses (read-only views)
@router.get("/teacher/courses")
def teacher_list_courses(
    request: Request, user: CurrentUser = Depends(require_role("teacher"))
):
    # Scoped to the courses this teacher is assigned to — a teacher's dashboard
    # must not offer another teacher's class as something to open or scan.
    sql = """
        SELECT c.courseid, c.course_code, c.course_name,
               COALESCE(c.status,'active') AS status,
               (SELECT COUNT(*) FROM course_enrollment e
                  WHERE e.courseid = c.courseid AND e.status='active') AS enrolled
        FROM course c
        WHERE COALESCE(c.status,'active') = 'active'
          AND c.teacher_id = %s
        ORDER BY c.course_code
    """
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(sql, (user.account_id,))
        return {"success": True, "courses": _dict_rows(cur)}


@router.get("/teacher/sessions")
def teacher_list_sessions(
    request: Request,
    background_tasks: BackgroundTasks,
    course_id: int | None = None,
    status: str | None = None,
    user: CurrentUser = Depends(require_role("teacher")),
):
    # This feeds the teacher's "Active session" picker, so it is the place
    # where an overdue session must not still be on offer.
    expire_overdue_sessions(request.app.state.cfg.database_url, background_tasks)
    # Own courses only — this feeds the session picker on every teacher tab.
    clauses, params = ["c.teacher_id = %s"], [user.account_id]
    if course_id is not None:
        clauses.append("s.courseid = %s")
        params.append(course_id)
    if status:
        if status not in ("scheduled", "active", "ended", "cancelled"):
            raise HTTPException(400, "Invalid status")
        clauses.append("s.status = %s")
        params.append(status)
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
    assert_teacher_course(
        request.app.state.cfg.database_url, user.account_id, course_id
    )
    sql = """
        SELECT e.accountid, pi.full_name, pi.student_id, ua.email, e.status,
               -- Count attendance only within ENDED sessions so the ratio
               -- stays consistent with the denominator. An in-progress
               -- (active) session's provisional 'present' must not inflate
               -- the numerator above the completed-session count.
               COUNT(r.attendancerecordid)
                 FILTER (WHERE r.status IN ('present','late') AND s.status='ended') AS attended,
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
    teacher_id: int | None = None


class CoursePatchBody(BaseModel):
    """Partial update — only the fields actually sent are written.

    `teacher_id` needs three states, not two: absent = leave the assignment
    alone, null = unassign, an id = reassign. A plain `int | None` default of
    None could not tell "not sent" from "clear it", so the sentinel below is
    what distinguishes them.
    """

    course_code: str | None = None
    course_name: str | None = None
    teacher_id: int | None = None
    status: str | None = None


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
               c.teacher_id,
               tpi.full_name AS teacher_name,
               (SELECT COUNT(*) FROM attendance_session s
                  WHERE s.courseid = c.courseid AND s.status='active') AS active_sessions
        FROM course c
        LEFT JOIN personal_info tpi ON tpi.accountid = c.teacher_id
        ORDER BY c.courseid
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
            raise HTTPException(409, f"Course code {body.course_code} already exists")
        if body.teacher_id is not None:
            cur.execute(
                """SELECT up.role FROM user_account ua
                   JOIN user_profiles up ON up.profileid = ua.profileid
                   WHERE ua.accountid = %s""",
                (body.teacher_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Assigned teacher account not found")
            if row[0] != "teacher":
                raise HTTPException(400, "Assigned account must be a teacher")
        cur.execute(
            "INSERT INTO course (course_code, course_name, teacher_id) VALUES (%s, %s, %s) RETURNING courseid",
            (body.course_code, body.course_name, body.teacher_id),
        )
        course_id = cur.fetchone()[0]
    return {"success": True, "course_id": course_id}


@router.patch("/admin/courses/{course_id}")
def admin_update_course(
    course_id: int, body: CoursePatchBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    """U26: edit a course's details (code, name, assigned teacher, status)."""
    supplied = body.model_dump(exclude_unset=True)
    if not supplied:
        raise HTTPException(400, "No fields to update")
    if "status" in supplied and supplied["status"] not in ("active", "inactive"):
        raise HTTPException(400, "Invalid status")
    for field in ("course_code", "course_name"):
        if field in supplied and not (supplied[field] or "").strip():
            raise HTTPException(400, f"{field} cannot be empty")

    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM course WHERE courseid = %s", (course_id,))
        if not cur.fetchone():
            raise HTTPException(404, "Course not found")

        # course_code carries a UNIQUE constraint; catching the clash here
        # gives the admin the offending code instead of a 500 from the driver.
        if "course_code" in supplied:
            cur.execute(
                "SELECT 1 FROM course WHERE course_code = %s AND courseid <> %s",
                (supplied["course_code"], course_id),
            )
            if cur.fetchone():
                raise HTTPException(
                    409, f"Course code {supplied['course_code']} already exists",
                )

        # Same rule as course creation: an assignment must point at a teacher.
        if supplied.get("teacher_id") is not None:
            cur.execute(
                """SELECT up.role FROM user_account ua
                   JOIN user_profiles up ON up.profileid = ua.profileid
                   WHERE ua.accountid = %s""",
                (supplied["teacher_id"],),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Assigned teacher account not found")
            if row[0] != "teacher":
                raise HTTPException(400, "Assigned account must be a teacher")

        columns = [f for f in
                   ("course_code", "course_name", "teacher_id", "status")
                   if f in supplied]
        cur.execute(
            f"UPDATE course SET {', '.join(f'{c_} = %s' for c_ in columns)} "
            "WHERE courseid = %s",
            [supplied[c_] for c_ in columns] + [course_id],
        )
    return {"success": True, "course_id": course_id, "updated": columns}


@router.patch("/admin/courses/{course_id}/status")
def admin_set_course_status(
    course_id: int, body: CourseStatusBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    if body.status not in ("active", "inactive"):
        raise HTTPException(400, "Invalid status")
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
            raise HTTPException(404, "Course not found")
        cur.execute(
            """SELECT 1 FROM attendance_record r
               JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
               WHERE s.courseid = %s LIMIT 1""",
            (course_id,),
        )
        if cur.fetchone():
            raise HTTPException(409, "This course already has attendance records and cannot be deleted (deactivate it instead)")
        cur.execute(
            "SELECT COUNT(*) FROM attendance_session WHERE courseid = %s",
            (course_id,),
        )
        session_count = cur.fetchone()[0]
        if session_count > 0 and not force:
            raise HTTPException(
                409, f"This course has {session_count} scheduled session(s); use force=true to confirm deletion",
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
            raise HTTPException(404, "Course not found")
        cur.execute(
            """SELECT up.role FROM user_account ua
               JOIN user_profiles up ON up.profileid = ua.profileid
               WHERE ua.accountid = %s""",
            (body.account_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "User not found")
        if row[0] != "student":
            raise HTTPException(400, "Courses can only be assigned to students")
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
    request: Request, background_tasks: BackgroundTasks,
    course_id: int | None = None,
    user: CurrentUser = Depends(require_role("admin")),
):
    expire_overdue_sessions(request.app.state.cfg.database_url, background_tasks)
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
        raise HTTPException(400, "Invalid status")
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(status,'active') FROM course WHERE courseid = %s",
            (body.course_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Course not found")
        if row[0] == "inactive":
            raise HTTPException(400, "Course is inactive; cannot schedule sessions")
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
        fields.append("start_time = %s")
        params.append(body.start_time)
    if body.end_time is not None:
        fields.append("end_time = %s")
        params.append(body.end_time)
    if body.status is not None:
        if body.status not in ("scheduled", "active", "ended", "cancelled"):
            raise HTTPException(400, "Invalid status")
        fields.append("status = %s")
        params.append(body.status)
    if not fields:
        raise HTTPException(400, "No fields to update")
    with _db(request.app.state.cfg.database_url) as c, c.cursor() as cur:
        # 'ended' and 'cancelled' are terminal. Reopening one would put a
        # sitting whose attendance is already finalised back in progress, and
        # the expiry sweep would immediately finalise it a second time.
        if body.status is not None:
            cur.execute(
                "SELECT status FROM attendance_session WHERE attendancesessionid = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Session not found")
            current = row[0]
            if current in ("ended", "cancelled") and body.status != current:
                raise HTTPException(
                    409,
                    f"This session is {current} and cannot be reopened; "
                    "schedule a new session instead",
                )
        params.append(session_id)
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
            raise HTTPException(409, "This session already has attendance records and cannot be deleted")
        cur.execute(
            "DELETE FROM attendance_session WHERE attendancesessionid = %s",
            (session_id,),
        )
    return {"success": True}
