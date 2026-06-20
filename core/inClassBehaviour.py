"""Class 8: inClassBehaviour

Covers UC:
  U32 View Classroom Behaviour Analysis Report (viewBehaviourReport)
  U33 View Classroom Activity Heatmap          (viewHeatmap)
  U35 Configure Classroom Behaviour Analysis   (enableBehaviourAnalysis /
                                                disableBehaviourAnalysis)

Attributes (per FYP class diagram):
  - behaviour                 (per-event payloads — sourced live from DB)
  - isAnalysisEnabled: bool   (per-course toggle — persisted in behaviour_config)

The U35 toggle now persists to the `behaviour_config` table introduced
in schema v0.5; the U32 / U33 queries read from `behaviour_event` and
`heatmap_snapshot`. All three queries fall back gracefully when their
tables are missing so the system still boots before the migration runs.
"""

from __future__ import annotations

from typing import Any

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.userInformation import CurrentUser, require_role


_DEFAULT_FLAGS = {
    "enabled": False,
    "drowsiness": False,
    "phone_usage": False,
    "heatmap": False,
}


def load_behaviour_config(database_url: str, course_id: int) -> dict[str, bool]:
    """Read the per-course behaviour-analysis switch from behaviour_config.

    Returns _DEFAULT_FLAGS (all-off) when no row exists or the table is
    missing so callers get a consistent dict shape.
    """
    try:
        with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT enabled, drowsiness, phone_usage, heatmap
                   FROM behaviour_config WHERE courseid = %s""",
                (course_id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "enabled": bool(row[0]),
                    "drowsiness": bool(row[1]),
                    "phone_usage": bool(row[2]),
                    "heatmap": bool(row[3]),
                }
    except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn,
            psycopg2.OperationalError):
        pass
    except Exception:
        pass
    return dict(_DEFAULT_FLAGS)


class InClassBehaviour:
    """Classroom behaviour analytics entity.

    Per-course toggles are persisted in behaviour_config (U35).
    Event/heatmap queries read live from behaviour_event /
    heatmap_snapshot (U32/U33).
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        # Class diagram attribute: + behaviour (loose dict of recent events)
        self.behaviour: dict[str, Any] = {}
        # Class diagram attribute: + isAnalysisEnabled: bool
        # (course-scoped — set per request via _refresh_enabled_for)
        self.isAnalysisEnabled: bool = False

    def _refresh_enabled_for(self, course_id: int) -> dict[str, bool]:
        flags = load_behaviour_config(self.database_url, course_id)
        self.isAnalysisEnabled = flags["enabled"]
        return flags

    # ── U32 viewBehaviourReport ─────────────────────────────────────
    def viewBehaviourReport(self, session_id: int) -> dict[str, Any]:
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT be.accountid, pi.full_name,
                              COUNT(*) FILTER (WHERE be.event_type='drowsiness') AS drowsy,
                              COUNT(*) FILTER (WHERE be.event_type='phone')      AS phone
                       FROM behaviour_event be
                       LEFT JOIN personal_info pi ON pi.accountid = be.accountid
                       WHERE be.attendancesessionid = %s
                       GROUP BY be.accountid, pi.full_name
                       ORDER BY pi.full_name NULLS LAST""",
                    (session_id,),
                )
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            self.behaviour = {"events_by_student": rows}
            return {"success": True, "session_id": session_id, "students": rows}
        except psycopg2.errors.UndefinedTable:
            return {"success": True, "session_id": session_id, "students": [],
                    "note": "behaviour_event table not yet created (behaviour analysis module pending integration)"}
        except Exception:
            return {"success": True, "session_id": session_id, "students": []}

    # ── U33 viewHeatmap ─────────────────────────────────────────────
    def viewHeatmap(self, session_id: int) -> dict[str, Any]:
        try:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT zone_x, zone_y, intensity, captured_at
                       FROM heatmap_snapshot
                       WHERE attendancesessionid = %s
                       ORDER BY captured_at""",
                    (session_id,),
                )
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return {"success": True, "session_id": session_id, "zones": rows}
        except psycopg2.errors.UndefinedTable:
            return {"success": True, "session_id": session_id, "zones": [],
                    "note": "heatmap_snapshot table not yet created (pending integration)"}
        except Exception:
            return {"success": True, "session_id": session_id, "zones": []}

    # ── U35 enable / disable (DB-backed) ────────────────────────────
    def enableBehaviourAnalysis(
        self, course_id: int,
        drowsiness: bool = True, phone_usage: bool = True, heatmap: bool = True,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        return self._upsert_config(
            course_id,
            enabled=True,
            drowsiness=drowsiness,
            phone_usage=phone_usage,
            heatmap=heatmap,
            updated_by=updated_by,
        )

    def disableBehaviourAnalysis(
        self, course_id: int, updated_by: int | None = None,
    ) -> dict[str, Any]:
        return self._upsert_config(
            course_id, enabled=False, updated_by=updated_by,
        )

    # ── internal ────────────────────────────────────────────────────
    def _upsert_config(
        self, course_id: int, enabled: bool,
        drowsiness: bool | None = None, phone_usage: bool | None = None,
        heatmap: bool | None = None, updated_by: int | None = None,
    ) -> dict[str, Any]:
        try:
            with psycopg2.connect(self.database_url) as conn:
                try:
                    with conn.cursor() as cur:
                        # 1. Verify the course exists (FK protection).
                        cur.execute("SELECT 1 FROM course WHERE courseid = %s", (course_id,))
                        if not cur.fetchone():
                            raise HTTPException(404, "Course not found")

                        # 2. UPSERT. When disabling we only flip `enabled`;
                        #    when enabling we may also flip sub-feature flags.
                        if enabled:
                            cur.execute(
                                """INSERT INTO behaviour_config
                                      (courseid, enabled, drowsiness, phone_usage,
                                       heatmap, updated_by, updated_at)
                                   VALUES (%s, TRUE, %s, %s, %s, %s, NOW())
                                   ON CONFLICT (courseid) DO UPDATE
                                     SET enabled    = EXCLUDED.enabled,
                                         drowsiness = EXCLUDED.drowsiness,
                                         phone_usage = EXCLUDED.phone_usage,
                                         heatmap    = EXCLUDED.heatmap,
                                         updated_by = EXCLUDED.updated_by,
                                         updated_at = NOW()""",
                                (course_id, drowsiness, phone_usage, heatmap, updated_by),
                            )
                        else:
                            cur.execute(
                                """INSERT INTO behaviour_config
                                      (courseid, enabled, updated_by, updated_at)
                                   VALUES (%s, FALSE, %s, NOW())
                                   ON CONFLICT (courseid) DO UPDATE
                                     SET enabled    = FALSE,
                                         updated_by = EXCLUDED.updated_by,
                                         updated_at = NOW()""",
                                (course_id, updated_by),
                            )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        except psycopg2.errors.UndefinedTable:
            raise HTTPException(
                503, "behaviour_config table does not exist; please run schema.sql first",
            )

        flags = self._refresh_enabled_for(course_id)
        return {"success": True, "course_id": course_id, "config": flags}


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["inClassBehaviour"])


def _svc(request: Request) -> InClassBehaviour:
    return InClassBehaviour(request.app.state.cfg.database_url)


# Teacher: U32 / U33
@router.get("/teacher/sessions/{session_id}/behaviour")
def teacher_view_behaviour(
    session_id: int, request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewBehaviourReport(session_id)


@router.get("/teacher/sessions/{session_id}/heatmap")
def teacher_view_heatmap(
    session_id: int, request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    return _svc(request).viewHeatmap(session_id)


# Admin: U35
class BehaviourToggleBody(BaseModel):
    enable: bool
    drowsiness: bool = True
    phone_usage: bool = True
    heatmap: bool = True


@router.patch("/admin/courses/{course_id}/behaviour-analysis")
def admin_toggle_behaviour(
    course_id: int, body: BehaviourToggleBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    svc = _svc(request)
    if body.enable:
        return svc.enableBehaviourAnalysis(
            course_id, body.drowsiness, body.phone_usage, body.heatmap,
            updated_by=user.account_id,
        )
    return svc.disableBehaviourAnalysis(course_id, updated_by=user.account_id)


@router.get("/admin/courses/{course_id}/behaviour-analysis")
def admin_get_behaviour(
    course_id: int, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    flags = load_behaviour_config(request.app.state.cfg.database_url, course_id)
    return {"success": True, "course_id": course_id, "config": flags}
