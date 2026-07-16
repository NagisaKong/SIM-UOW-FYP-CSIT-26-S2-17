"""Class 9: trainConfiguration

Covers UC:
  U22 Assign Training and Testing Data    (selectTrainingDataSet)
  U23 Configure and Train ML Model        (trainSelectedModelWithSelectedTrainDataSet,
                                           configureModelTraining)
  U24 Configure Ensemble Model Voting     (combineDifferentModel)
  U25 Retrain and Redeploy ML Model       (deployUpdatedModel)

Admin-only AI model governance.

What "training" means here: the recognition networks (ArcFace / FaceNet)
are pre-trained; per CR-04 the GAN runs strictly offline. The tunable,
deployable artefact of this system is the SIMILARITY THRESHOLD each
embedding store uses to accept a match. A training run therefore:

  1. loads every stored embedding for the selected model (active rows are
     the current gallery; superseded/inactive rows of the same account
     provide genuine same-person pairs),
  2. builds genuine (same account) and imposter (cross account) pairs and
     splits them into train/test by the U22 percentage,
  3. sweeps candidate thresholds on the train split and picks the one with
     the best balanced accuracy,
  4. reports test-split statistics (accuracy / FPR / FNR) for the chosen
     threshold AND for the currently deployed threshold (U25 comparison).

Deployment (U25) persists the threshold to MODEL_CONFIGS and hot-applies
it to the live pipeline's embedding store; AttendancePipeline.from_env
re-applies the active MODEL_CONFIGS row on startup.
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from typing import Any, Callable

import numpy as np
import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Request
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel

from core.userInformation import CurrentUser, require_role

VALID_MODELS = ("arcface", "facenet")

# Pair-count caps keep a run fast even with a large gallery.
_MAX_GENUINE_PAIRS = 5000
_MAX_IMPOSTER_PAIRS = 20000
_THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 3)
_FALLBACK_THRESHOLD = 0.35


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


def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _normalise_model_name(name: str) -> str:
    """Accept legacy frontend values like 'arcface_ensemble'."""
    name = (name or "").lower().removesuffix("_ensemble")
    if name not in VALID_MODELS:
        raise HTTPException(400, f"model_name must be one of {list(VALID_MODELS)}")
    return name


# ── shared in-process state ──────────────────────────────────────────
# Training-run config (U22/U23/U24 selections) + the async job monitor.
_active_config: dict[str, Any] = {
    "trainingDataSet": None,        # {"model_name", "train_pct", counts…}
    "singleOrMultimodel": False,    # True = ensemble of multiple models
    "training_params": {},          # epochs, batch_size, … (metadata)
    "ensemble": {"models": ["arcface"], "weighting": "equal"},
}

_job_lock = threading.Lock()
_job: dict[str, Any] = {
    "status": "idle",       # idle | running | done | failed
    "progress": 0,          # 0..100
    "message": "",
    "model_name": None,
    "started_at": None,
    "result": None,         # metrics dict on success
    "error": None,
}


def _job_update(**kwargs) -> None:
    with _job_lock:
        _job.update(kwargs)


def _job_snapshot() -> dict[str, Any]:
    with _job_lock:
        return dict(_job)


def load_deployed_threshold(database_url: str, model_name: str) -> float | None:
    """Active MODEL_CONFIGS threshold for a model, or None."""
    try:
        with _db(database_url) as c, c.cursor() as cur:
            cur.execute(
                """SELECT similarity_threshold FROM model_configs
                   WHERE model_name = %s AND is_active = TRUE
                   ORDER BY updated_at DESC LIMIT 1""",
                (model_name,),
            )
            row = cur.fetchone()
            return float(row[0]) if row else None
    except Exception:  # noqa: BLE001 — table absent must not break callers
        return None


class TrainConfiguration:
    """AI model training / ensemble / deployment configuration (U22–U25)."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    # ── U22 selectTrainingDataSet ───────────────────────────────────
    def selectTrainingDataSet(self, model_name: str, train_pct: int) -> dict[str, Any]:
        model_name = _normalise_model_name(model_name)
        if not (10 <= train_pct <= 95):
            raise HTTPException(400, "train_pct must be between 10 and 95")
        with _db(self.database_url) as c, c.cursor() as cur:
            # The evaluation corpus is EVERY stored embedding for the model:
            # inactive (superseded) rows supply genuine same-person pairs.
            cur.execute(
                """SELECT COUNT(*), COUNT(DISTINCT accountid)
                   FROM face_embedding WHERE model_name = %s""",
                (model_name,),
            )
            total, accounts = cur.fetchone()
        if total == 0:
            raise HTTPException(400, "Training set is empty; cannot allocate data")
        train_count = int(total * train_pct / 100)
        _active_config["trainingDataSet"] = {
            "model_name": model_name,
            "train_pct": train_pct,
            "train_count": train_count,
            "test_count": total - train_count,
            "embeddings": total,
            "accounts": accounts,
        }
        return {"success": True, **_active_config["trainingDataSet"]}

    # ── U23 configureModelTraining ──────────────────────────────────
    def configureModelTraining(self, params: dict[str, Any]) -> dict[str, Any]:
        epochs = params.get("epochs")
        if epochs is not None and epochs <= 0:
            raise HTTPException(400, "epochs must be positive")
        lr = params.get("learning_rate")
        if lr is not None and not (0 < lr < 1):
            raise HTTPException(400, "learning_rate must be in (0, 1)")
        _active_config["training_params"] = dict(params or {})
        return {"success": True, "training_params": _active_config["training_params"]}

    # ── evaluation core (U23/U25) ───────────────────────────────────
    def _load_embeddings(self, model_name: str) -> dict[int, list[np.ndarray]]:
        """All embeddings for the model grouped by account (active first)."""
        by_account: dict[int, list[np.ndarray]] = {}
        with _db(self.database_url) as c:
            register_vector(c)
            with c.cursor() as cur:
                cur.execute(
                    """SELECT accountid, embedding_vector
                       FROM face_embedding
                       WHERE model_name = %s
                       ORDER BY is_active DESC, created_at DESC""",
                    (model_name,),
                )
                for account_id, vec in cur.fetchall():
                    arr = _normalise(np.asarray(vec, dtype=np.float32))
                    by_account.setdefault(account_id, []).append(arr)
        return by_account

    def runEvaluation(
        self,
        model_name: str,
        train_pct: int,
        current_threshold: float | None,
        progress: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Build genuine/imposter pairs, fit the threshold on the train
        split, report metrics on the test split. Pure numpy — CPU-fast."""
        def report(pct: int, msg: str) -> None:
            if progress:
                progress(pct, msg)

        t0 = time.time()
        report(5, "Loading embeddings from the database…")
        by_account = self._load_embeddings(model_name)
        if not by_account:
            raise HTTPException(400, f"No embeddings stored for model '{model_name}'")
        if len(by_account) < 2:
            raise HTTPException(
                400,
                "At least two enrolled accounts are required to evaluate "
                "recognition (imposter pairs cannot be formed from one account)",
            )

        rng = random.Random(42)
        accounts = sorted(by_account)

        report(20, "Building genuine (same-person) pairs…")
        genuine: list[float] = []
        for acc in accounts:
            vecs = by_account[acc]
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    genuine.append(float(np.dot(vecs[i], vecs[j])))
        if len(genuine) > _MAX_GENUINE_PAIRS:
            genuine = rng.sample(genuine, _MAX_GENUINE_PAIRS)

        report(40, "Building imposter (cross-person) pairs…")
        imposter: list[float] = []
        flat = [(acc, v) for acc in accounts for v in by_account[acc]]
        n = len(flat)
        all_cross = n * (n - 1) // 2
        if all_cross <= _MAX_IMPOSTER_PAIRS:
            for i in range(n):
                for j in range(i + 1, n):
                    if flat[i][0] != flat[j][0]:
                        imposter.append(float(np.dot(flat[i][1], flat[j][1])))
        else:
            seen = set()
            while len(imposter) < _MAX_IMPOSTER_PAIRS and len(seen) < all_cross:
                i, j = rng.randrange(n), rng.randrange(n)
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                if flat[i][0] != flat[j][0]:
                    imposter.append(float(np.dot(flat[i][1], flat[j][1])))
        if not imposter:
            raise HTTPException(400, "Could not form any imposter pairs")

        # Train/test split per the U22 percentage.
        rng.shuffle(genuine)
        rng.shuffle(imposter)
        g_cut = max(1, int(len(genuine) * train_pct / 100)) if genuine else 0
        i_cut = max(1, int(len(imposter) * train_pct / 100))
        g_train, g_test = genuine[:g_cut], genuine[g_cut:] or genuine[:g_cut]
        i_train, i_test = imposter[:i_cut], imposter[i_cut:] or imposter[:i_cut]

        def metrics_at(th: float, gen: list[float], imp: list[float]) -> dict[str, Any]:
            fn = sum(1 for s in gen if s < th)
            fp = sum(1 for s in imp if s >= th)
            fnr = fn / len(gen) if gen else None
            fpr = fp / len(imp) if imp else None
            if gen:
                acc = (((1 - fnr) + (1 - fpr)) / 2)   # balanced accuracy
            else:
                acc = 1 - fpr                          # imposter-only rejection
            return {
                "accuracy": round(acc, 4),
                "fpr": round(fpr, 4) if fpr is not None else None,
                "fnr": round(fnr, 4) if fnr is not None else None,
            }

        report(60, "Sweeping candidate thresholds on the train split…")
        limited = not g_train
        if limited:
            # No same-person pairs (one embedding per account): calibrate
            # just above the imposter distribution with a safety margin.
            base = float(np.percentile(np.asarray(i_train), 99.5))
            best_th = float(np.clip(base + 0.05, 0.2, 0.9))
        else:
            best_acc, tied = -1.0, [_FALLBACK_THRESHOLD]
            for th in _THRESHOLD_GRID:
                acc = metrics_at(float(th), g_train, i_train)["accuracy"]
                if acc > best_acc:
                    best_acc, tied = acc, [float(th)]
                elif acc == best_acc:
                    tied.append(float(th))
            # Among equally-scoring thresholds take the middle one: it sits
            # in the widest margin between the two score distributions.
            best_th = tied[len(tied) // 2]

        report(85, "Scoring the held-out test split…")
        test = metrics_at(best_th, g_test, i_test)
        current = (
            metrics_at(current_threshold, g_test, i_test)
            if current_threshold is not None else None
        )

        result = {
            "model_name": model_name,
            "train_pct": train_pct,
            "new_threshold": round(best_th, 3),
            **test,
            "genuine_pairs": len(genuine),
            "imposter_pairs": len(imposter),
            "accounts": len(accounts),
            "current_threshold": current_threshold,
            "current_accuracy": current["accuracy"] if current else None,
            "improved": (current is None) or (test["accuracy"] >= current["accuracy"]),
            "limited_calibration": limited,
            "duration_s": round(time.time() - t0, 2),
        }
        report(100, "Training complete.")
        return result

    # ── U23 trainSelectedModelWithSelectedTrainDataSet (async job) ──
    def trainSelectedModelWithSelectedTrainDataSet(
        self, current_threshold: float | None,
    ) -> dict[str, Any]:
        dataset = _active_config.get("trainingDataSet")
        if not dataset:
            raise HTTPException(400, "Please call selectTrainingDataSet to choose a dataset first")
        with _job_lock:
            if _job["status"] == "running":
                raise HTTPException(409, "A training run is already in progress")
            _job.update(
                status="running", progress=0, message="Starting…",
                model_name=dataset["model_name"], started_at=time.time(),
                result=None, error=None,
            )

        def worker() -> None:
            try:
                result = self.runEvaluation(
                    dataset["model_name"], dataset["train_pct"], current_threshold,
                    progress=lambda p, m: _job_update(progress=p, message=m),
                )
                _job_update(status="done", progress=100,
                            message="Training complete.", result=result)
            except HTTPException as exc:
                _job_update(status="failed", message=str(exc.detail), error=str(exc.detail))
            except Exception as exc:  # noqa: BLE001 — surface any failure to the poller
                _job_update(status="failed", message=str(exc), error=str(exc))

        threading.Thread(target=worker, name="training-run", daemon=True).start()
        return {"success": True, "status": "started", "model_name": dataset["model_name"]}

    # ── U24 combineDifferentModel ───────────────────────────────────
    def combineDifferentModel(
        self, models: list[str], weighting: str, pipeline,
    ) -> dict[str, Any]:
        if not models:
            raise HTTPException(400, "Select at least one model")
        if weighting not in ("equal", "confidence"):
            raise HTTPException(400, "Invalid weighting")
        selected = [_normalise_model_name(m) for m in models]

        warnings: list[str] = []
        loaded = set(pipeline.store_manager.stores) if pipeline else set()
        active = [m for m in selected if m in loaded]
        for m in selected:
            if m not in loaded:
                warnings.append(
                    f"'{m}' is not loaded on this server "
                    "(enable it via AI_USE_FACENET=true and restart)"
                )
        if not active:
            raise HTTPException(400, "; ".join(warnings) or "No selected model is available")

        # Apply live voting weights: equal → 1.0 each; confidence → keep the
        # AIConfig per-model weights. Unselected models drop to 0.
        if pipeline is not None:
            cfg = pipeline.cfg
            base = {"arcface": cfg.arcface_weight, "facenet": cfg.facenet_weight}
            for name in pipeline._weights:
                if name not in active:
                    pipeline._weights[name] = 0.0
                else:
                    pipeline._weights[name] = 1.0 if weighting == "equal" else base.get(name, 1.0)

        is_ensemble = len(active) >= 2
        _active_config["singleOrMultimodel"] = is_ensemble
        _active_config["ensemble"] = {"models": active, "weighting": weighting}
        return {
            "success": True, "is_ensemble": is_ensemble,
            "models": active, "weighting": weighting,
            "applied_weights": dict(pipeline._weights) if pipeline else None,
            "warnings": warnings,
        }

    # ── U25 deployUpdatedModel ──────────────────────────────────────
    def deployUpdatedModel(
        self, result: dict[str, Any], force: bool, updated_by: int, pipeline,
    ) -> dict[str, Any]:
        """Persist a calibrated threshold to MODEL_CONFIGS and hot-apply it
        to the live embedding store. If the new configuration scores worse
        than the currently deployed threshold on the same test split, the
        deploy is withheld and a warning returned unless force=True."""
        model_name = result["model_name"]
        if not result.get("improved") and not force:
            return {
                "success": True,
                "deployed": False,
                "warning": (
                    f"New threshold {result['new_threshold']} scores "
                    f"{result['accuracy']:.2%} vs the current model's "
                    f"{result['current_accuracy']:.2%} on the same test set."
                ),
                **result,
            }

        with _db(self.database_url) as c, c.cursor() as cur:
            cur.execute(
                "UPDATE model_configs SET is_active = FALSE WHERE model_name = %s",
                (model_name,),
            )
            cur.execute(
                """INSERT INTO model_configs
                       (model_name, similarity_threshold, is_active, updated_by)
                   VALUES (%s, %s, TRUE, %s)""",
                (model_name, result["new_threshold"], updated_by),
            )

        applied_live = False
        if pipeline is not None:
            store = pipeline.store_manager.stores.get(model_name)
            if store is not None:
                store.threshold = float(result["new_threshold"])
                applied_live = True

        return {
            "success": True,
            "deployed": True,
            "applied_live": applied_live,
            **result,
        }


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["trainConfiguration"])


def _svc(request: Request) -> TrainConfiguration:
    return TrainConfiguration(request.app.state.cfg.database_url)


def _live_threshold(request: Request, model_name: str) -> float | None:
    pipeline = getattr(request.app.state, "pipeline", None)
    store = pipeline.store_manager.stores.get(model_name) if pipeline else None
    if store is not None:
        return float(store.threshold)
    return load_deployed_threshold(request.app.state.cfg.database_url, model_name)


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
    """U23: start an asynchronous training run; poll /admin/training-status."""
    dataset = _active_config.get("trainingDataSet") or {}
    current = _live_threshold(request, dataset.get("model_name", "arcface"))
    return _svc(request).trainSelectedModelWithSelectedTrainDataSet(current)


@router.get("/admin/training-status")
def admin_training_status(
    user: CurrentUser = Depends(require_role("admin")),
):
    """U23: real-time progress + final statistics of the training run."""
    return {"success": True, **_job_snapshot()}


@router.post("/admin/ensemble")
def admin_configure_ensemble(
    body: EnsembleBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    pipeline = getattr(request.app.state, "pipeline", None)
    return _svc(request).combineDifferentModel(body.models, body.weighting, pipeline)


@router.post("/admin/deploy")
def admin_deploy_model(
    body: DeployBody, request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    """U25: deploy the most recent completed training run."""
    snap = _job_snapshot()
    if snap["status"] != "done" or not snap["result"]:
        raise HTTPException(400, "No completed training run to deploy — run training first")
    pipeline = getattr(request.app.state, "pipeline", None)
    return _svc(request).deployUpdatedModel(
        snap["result"], body.force, user.account_id, pipeline,
    )


@router.post("/admin/retrain")
def admin_retrain_model(
    request: Request,
    force: bool = False,
    user: CurrentUser = Depends(require_role("admin")),
):
    """U25 one-click retrain & redeploy: synchronously re-evaluate on the
    latest data and deploy unless the result is worse (then a warning is
    returned; call again with ?force=true to deploy anyway)."""
    dataset = _active_config.get("trainingDataSet") or {}
    model_name = dataset.get("model_name", "arcface")
    train_pct = dataset.get("train_pct", 80)
    svc = _svc(request)
    current = _live_threshold(request, model_name)
    result = svc.runEvaluation(model_name, train_pct, current)
    pipeline = getattr(request.app.state, "pipeline", None)
    return svc.deployUpdatedModel(result, force, user.account_id, pipeline)
