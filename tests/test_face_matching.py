"""Unit tests for gallery matching (core/attendancePipeline.EmbeddingStore).

These cover the rules that stop the system confidently naming the wrong
student — the failure reported from classroom testing, where three enrolled
people were nearly always reported as the same person:

  * a match must beat the runner-up ACCOUNT by a margin, not merely clear the
    similarity threshold;
  * one student may hold several gallery entries (angles), and scores are
    collapsed per account so a well-enrolled person cannot occupy every slot.

No database, no model weights — vectors are constructed directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.attendancePipeline import ARCFACE_DIM, EmbeddingRow, EmbeddingStore, StudentInfo


def _unit(*components: float) -> np.ndarray:
    """A unit vector whose first components are given, rest zero."""
    vec = np.zeros(ARCFACE_DIM, dtype=np.float32)
    vec[: len(components)] = components
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def _row(account_id: int, vec: np.ndarray, name: str) -> EmbeddingRow:
    return EmbeddingRow(
        face_id=account_id * 10, account_id=account_id,
        student_id=f"S{account_id:05d}", full_name=name,
        model_name="arcface", model_version="test", dimension=ARCFACE_DIM,
        vector=vec,
    )


def _store(rows, threshold=0.40, margin=0.05) -> EmbeddingStore:
    store = EmbeddingStore("arcface", ARCFACE_DIM, threshold, margin)
    store.load_rows(rows)
    return store


def test_clear_winner_is_identified():
    store = _store([_row(1, _unit(1, 0), "Alice"), _row(2, _unit(0, 1), "Bob")])
    account_id, score = store.best_match(_unit(1, 0.05))
    assert account_id == 1
    assert score > 0.9


def test_ambiguous_face_is_refused_not_guessed():
    """Two students equally close: naming either one would be a coin flip."""
    store = _store([_row(1, _unit(1, 0), "Alice"), _row(2, _unit(0, 1), "Bob")])
    account_id, score = store.best_match(_unit(1, 1))  # exactly between them
    assert account_id is None
    assert score == pytest.approx(0.707, abs=1e-3)  # both cleared the threshold


def test_margin_zero_restores_argmax_behaviour():
    """The old behaviour is still reachable, so the rule can be A/B tested."""
    rows = [_row(1, _unit(1, 0), "Alice"), _row(2, _unit(0, 1), "Bob")]
    assert _store(rows, margin=0.0).best_match(_unit(1, 1))[0] is not None


def test_below_threshold_is_unknown_even_with_a_clear_lead():
    store = _store([_row(1, _unit(1, 0), "Alice")])
    account_id, score = store.best_match(_unit(0.2, 1))
    assert account_id is None
    assert score < 0.4


def test_multiple_angles_per_student_do_not_block_the_runner_up():
    """A student with several entries must not fill the top-2 and mask the
    real second-place account, which the margin test needs to see."""
    rows = [
        _row(1, _unit(1, 0, 0), "Alice"),        # Alice, frontal
        _row(1, _unit(0.98, 0.2, 0), "Alice"),   # Alice, slight angle
        _row(2, _unit(0.95, 0, 0.3), "Bob"),     # Bob, similar-looking
    ]
    store = _store(rows)
    ranked = store.rank(_unit(1, 0, 0), top_k=3)
    assert [account for account, _ in ranked] == [1, 2]  # one row per account


def test_extra_angle_improves_a_previously_missed_match():
    """The point of multi-angle enrolment: a pose that missed the threshold
    against a single frontal entry is recognised once that angle is added."""
    frontal = _unit(1, 0, 0)
    side_pose = _unit(0.3, 0.954, 0)         # cos 0.30 to frontal — below 0.40

    single = _store([_row(1, frontal, "Alice")])
    assert single.best_match(side_pose)[0] is None, "single entry cannot cover this pose"

    # The student adds that angle; the same live frame now matches.
    multi = _store([_row(1, frontal, "Alice"), _row(1, _unit(0.35, 0.937, 0), "Alice")])
    assert multi.best_match(side_pose)[0] == 1


def test_upsert_replace_versus_append():
    store = _store([_row(1, _unit(1, 0), "Alice")])
    info = StudentInfo(account_id=1, student_id="S00001", full_name="Alice")

    store.upsert(1, _unit(0, 1), info=info)              # replace (default)
    assert len(store) == 1

    store.upsert(1, _unit(1, 0), info=info, replace=False)  # append an angle
    assert len(store) == 2
    # Both angles belong to the same account, so rank() still yields one row.
    assert len(store.rank(_unit(1, 0), top_k=3)) == 1


def test_empty_gallery_is_safe():
    store = _store([])
    assert store.best_match(_unit(1, 0)) == (None, 0.0)
    assert store.rank(_unit(1, 0)) == []
