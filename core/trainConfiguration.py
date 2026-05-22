"""Class 9: trainConfiguration

Covers UC:
  U22 Assign Training and Testing Data    (selectTrainingDataSet)
  U23 Configure and Train ML Model        (trainSelectedModelWithSelectedTrainDataSet,
                                           configureModelTraining)
  U24 Configure Ensemble Model Voting     (combineDifferentModel)
  U25 Retrain and Redeploy ML Model       (deployUpdatedModel)

Attributes:
  trainingDataSet, singleOrMultimodel:bool

Admin-only AI model governance. Per requirements:
  - Function signatures and routes are wired up.
  - Bodies are stubs / minimal — full training pipeline is out of scope
    for this preliminary build.
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


# In-memory training-run config (would be persisted in production).
_active_config: dict[str, Any] = {
    "trainingDataSet": None,        # e.g. {"model_name": "arcface", "train_pct": 80}
    "singleOrMultimodel": False,    # True = ensemble of multiple models
    "training_params": {},          # epochs, batch_size, …
    "ensemble": {"models": [], "weighting": "equal"},
}


class TrainConfiguration:
    """AI model training / ensemble / deployment configuration.

    All methods are STUBS for the preliminary submission.  They store
    configuration in process memory and return success payloads so the
    admin UI can be exercised end-to-end; actual training is deferred.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url

    # ── U22 selectTrainingDataSet ───────────────────────────────────
    def selectTrainingDataSet(
        self, model_name: str, train_pct: int,
    ) -> dict[str, Any]:
        if not (10 <= train_pct <= 95):
            raise HTTPException(400, "train_pct 必须在 10-95 之间")
        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM face_embedding WHERE is_active = TRUE AND model_name = %s",
                (model_name,),
            )
            total = cur.fetchone()[0]
        if total == 0:
            raise HTTPException(400, "训练集为空，无法分配数据")
        train_count = int(total * train_pct / 100)
        test_count = total - train_count
        _active_config["trainingDataSet"] = {
            "model_name": model_name,
            "train_pct": train_pct,
            "train_count": train_count,
            "test_count": test_count,
        }
        return {"success": True, **_active_config["trainingDataSet"]}

    # ── U23 configureModelTraining ──────────────────────────────────
    def configureModelTraining(
        self, params: dict[str, Any],
    ) -> dict[str, Any]:
        # TODO: validate model_architecture / epochs / learning_rate …
        _active_config["training_params"] = dict(params or {})
        return {"success": True, "training_params": _active_config["training_params"]}

    # ── U23 trainSelectedModelWithSelectedTrainDataSet ──────────────
    def trainSelectedModelWithSelectedTrainDataSet(self) -> dict[str, Any]:
        """STUB: real implementation will hand off to core/training/.
        For the preliminary build we just acknowledge the request."""
        if not _active_config.get("trainingDataSet"):
            raise HTTPException(400, "请先调用 selectTrainingDataSet 选择数据集")
        # TODO: spawn background training job, stream progress over WebSocket.
        return {
            "success": True,
            "status": "training_started_stub",
            "dataset": _active_config["trainingDataSet"],
            "params": _active_config["training_params"],
        }

    # ── U24 combineDifferentModel ───────────────────────────────────
    def combineDifferentModel(
        self, models: list[str], weighting: str = "equal",
    ) -> dict[str, Any]:
        if len(models) < 2:
            raise HTTPException(400, "Ensemble 至少需要两个模型 (U24)")
        if weighting not in ("equal", "confidence"):
            raise HTTPException(400, "weighting 非法")
        _active_config["singleOrMultimodel"] = True
        _active_config["ensemble"] = {"models": models, "weighting": weighting}
        return {"success": True, **_active_config["ensemble"]}

    # ── U25 deployUpdatedModel ──────────────────────────────────────
    def deployUpdatedModel(
        self, force: bool = False,
    ) -> dict[str, Any]:
        """STUB: real implementation will swap weights atomically and reload
        AttendancePipeline.  Currently returns success placeholder."""
        # TODO: persist deployment metadata, run health-check, swap pipeline.
        return {
            "success": True,
            "status": "deployed_stub",
            "force": force,
            "active_config": _active_config,
        }


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["trainConfiguration"])


def _svc(request: Request) -> TrainConfiguration:
    return TrainConfiguration(request.app.state.cfg.database_url)


class TrainingDataBody(BaseModel):
    model_name: str
    train_pct: int


class TrainingParamsBody(BaseModel):
    model_architecture: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None


class EnsembleBody(BaseModel):
    models: list[str]
    weighting: str = "equal"


class DeployBody(BaseModel):
    force: bool = False


@router.post("/admin/training-data")
def admin_select_training_data(
    body: TrainingDataBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).selectTrainingDataSet(body.model_name, body.train_pct)


@router.post("/admin/training-config")
def admin_configure_training(
    body: TrainingParamsBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).configureModelTraining(body.model_dump(exclude_none=True))


@router.post("/admin/train")
def admin_train_model(
    request: Request, user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).trainSelectedModelWithSelectedTrainDataSet()


@router.post("/admin/ensemble")
def admin_configure_ensemble(
    body: EnsembleBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).combineDifferentModel(body.models, body.weighting)


@router.post("/admin/deploy")
def admin_deploy_model(
    body: DeployBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).deployUpdatedModel(body.force)
