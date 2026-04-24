"""Embedding stores.

Two layers:

* `EmbeddingStore` — in-memory matrix for fast cosine lookup. One instance
  per recognizer model. Used at inference time.
* `SupabaseEmbeddingStore` — persistence wrapper around `EmbeddingRepo`
  that hydrates one `EmbeddingStore` per model from the database at
  startup and writes new enrolments back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .db import EmbeddingRepo, EmbeddingRow


@dataclass
class StudentInfo:
    account_id: int
    student_id: str | None
    full_name: str | None


# ────────────────────────────────────────────────────────────────────────────
# In-memory cosine-similarity store
# ────────────────────────────────────────────────────────────────────────────

class EmbeddingStore:
    """In-memory matrix of (N, dim) for a single recognition model."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        threshold: float,
    ):
        self.model_name = model_name
        self.dim = dim
        self.threshold = threshold
        self._ids: list[int] = []              # account_id per row
        self._info: dict[int, StudentInfo] = {}
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    # ── Hydration ───────────────────────────────────────────────────────
    def load_rows(self, rows: list[EmbeddingRow]) -> None:
        if not rows:
            self._ids = []
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            return

        self._ids = [r.account_id for r in rows]
        self._matrix = np.stack(
            [_normalise(r.vector) for r in rows], axis=0
        ).astype(np.float32)
        for r in rows:
            self._info[r.account_id] = StudentInfo(
                account_id=r.account_id,
                student_id=r.student_id,
                full_name=r.full_name,
            )

    def upsert(
        self,
        account_id: int,
        vector: np.ndarray,
        info: StudentInfo | None = None,
    ) -> None:
        vec = _normalise(vector).astype(np.float32)
        if account_id in self._ids:
            idx = self._ids.index(account_id)
            self._matrix[idx] = vec
        else:
            self._ids.append(account_id)
            self._matrix = np.vstack([self._matrix, vec[None, :]])
        if info is not None:
            self._info[account_id] = info

    # ── Query ───────────────────────────────────────────────────────────
    def best_match(self, query: np.ndarray) -> tuple[int | None, float]:
        """Return `(account_id, score)` for the closest embedding, or
        `(None, best_score)` if no one clears the threshold."""
        if self._matrix.shape[0] == 0:
            return None, 0.0
        q = _normalise(query).astype(np.float32)
        scores = self._matrix @ q
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score >= self.threshold:
            return self._ids[idx], score
        return None, score

    def info_for(self, account_id: int) -> StudentInfo | None:
        return self._info.get(account_id)

    def __len__(self) -> int:
        return len(self._ids)


# ────────────────────────────────────────────────────────────────────────────
# Supabase-backed manager
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SupabaseEmbeddingStore:
    """Per-model stores hydrated from the Supabase FACE_EMBEDDING table."""

    repo: EmbeddingRepo
    stores: dict[str, EmbeddingStore] = field(default_factory=dict)

    def register_store(self, store: EmbeddingStore) -> None:
        self.stores[store.model_name] = store

    def hydrate_all(self) -> dict[str, int]:
        """Load every active embedding into the matching in-memory store.

        Returns `{model_name: rows_loaded}` for logging.
        """
        summary: dict[str, int] = {}
        for name, store in self.stores.items():
            rows = self.repo.load_active_embeddings(name)
            store.load_rows(rows)
            summary[name] = len(rows)
        return summary

    def get(self, model_name: str) -> EmbeddingStore:
        return self.stores[model_name]

    # ── Enrolment: average across photos, write one row per model ───────
    def enrol_account(
        self,
        account_id: int,
        vectors_by_model: dict[str, list[np.ndarray]],
        retention_days: int,
        model_versions: dict[str, str],
        deactivate_previous: bool = True,
    ) -> dict[str, int]:
        """Insert (or refresh) a student's embedding for each model.

        `vectors_by_model` maps e.g. {"arcface": [v1, v2, v3], "facenet": [...]}.
        Returns `{model_name: face_id}` for the newly inserted DB rows.
        """
        written: dict[str, int] = {}
        for model_name, vectors in vectors_by_model.items():
            if not vectors:
                continue
            if deactivate_previous:
                self.repo.deactivate_embeddings(account_id, model_name)
            avg = _normalise(np.mean(vectors, axis=0))
            face_id = self.repo.insert_embedding(
                account_id=account_id,
                vec=avg,
                model_name=model_name,
                model_version=model_versions.get(model_name, "unknown"),
                retention_days=retention_days,
            )
            written[model_name] = face_id

            # Keep the in-memory store in sync immediately so the
            # just-enrolled student is recognisable without a restart.
            store = self.stores.get(model_name)
            if store is not None:
                store.upsert(account_id, avg)
        return written


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v
