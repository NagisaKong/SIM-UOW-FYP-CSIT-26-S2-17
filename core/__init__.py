"""core/ — business class modules.

Each *.py under core/ defines:
  • A business class (per the FYP class diagram).
  • A FastAPI APIRouter exposing that class's user-facing endpoints.

main_api.py imports `all_routers` and mounts them on the FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter

from core import (
    attendanceAppeal,
    attendanceRecord,
    attendanceSession,
    facialImage,
    inClassBehaviour,
    notification,
    report,
    trainConfiguration,
    userInformation,
)


all_routers: list[APIRouter] = [
    userInformation.router,
    attendanceRecord.router,
    notification.router,
    facialImage.router,
    attendanceSession.router,
    attendanceAppeal.router,
    report.router,
    inClassBehaviour.router,
    trainConfiguration.router,
]
