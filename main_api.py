"""FastAPI entrypoint for the FYP-26-S2-17 attendance system.

Loads the AttendancePipeline once at startup, mounts every business-class
router from core/, and starts uvicorn.

Run:
    python main_api.py
    # or:
    uvicorn main_api:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow running this file directly (`python main_api.py`).
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import all_routers
from core.attendancePipeline import AIConfig, AttendancePipeline
from core.attendanceSession import purge_expired_recordings


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AIConfig()
    print(cfg.log_summary())
    app.state.cfg = cfg
    app.state.pipeline = AttendancePipeline.from_env(cfg)
    # U03 retention: drop class recordings / detection rows past their 30-day expiry.
    try:
        stats = purge_expired_recordings(cfg.database_url)
        if stats["recordings_deleted"]:
            print(f"[retention] purged {stats['recordings_deleted']} expired recording(s)")
    except Exception as exc:  # noqa: BLE001 — startup must not crash on cleanup
        print(f"[retention] skipped: {exc}")
    yield


app = FastAPI(
    title="SIM-UOW Face Attendance System API",
    description=(
        "Face enrolment + identification + role-scoped endpoints, "
        "organised by the 9 business classes in core/."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Mount every business-class router (userInformation, attendanceRecord,
# notification, facialImage, attendanceSession, attendanceAppeal, report,
# inClassBehaviour, trainConfiguration).
for r in all_routers:
    app.include_router(r)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
