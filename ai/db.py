"""Supabase / PostgreSQL connection + embedding persistence.

Embeddings live in a pgvector `vector(N)` column. We register the
pgvector adapter on every connection so numpy arrays round-trip natively
in both directions — reads return `np.ndarray`, writes accept one as `%s`.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingRow:
    face_id: int
    account_id: int
    student_id: str | None
    full_name: str | None
    model_name: str
    model_version: str
    dimension: int
    vector: np.ndarray


# ────────────────────────────────────────────────────────────────────────────
# Repository
# ────────────────────────────────────────────────────────────────────────────

class EmbeddingRepo:
    """Thin repository over the FACE_EMBEDDING + PERSONAL_INFO tables."""

    def __init__(self, database_url: str):
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set; add it to .env before starting the AI pipeline."
            )
        self._database_url = database_url

    # ── Connection management ───────────────────────────────────────────
    @contextlib.contextmanager
    def _conn(self) -> Iterator[psycopg2.extensions.connection]:
        conn = psycopg2.connect(self._database_url)
        try:
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ping(self) -> bool:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1

    # ── Read: bulk load for pipeline startup ────────────────────────────
    def load_active_embeddings(self, model_name: str) -> list[EmbeddingRow]:
        """Return every active embedding for a given recognizer model.

        Joined with PERSONAL_INFO so the caller can label bboxes with the
        student's actual ID rather than the database PK.

        With `register_vector` installed on the connection, pgvector
        columns come back as `np.ndarray` directly — no text parsing.
        """
        sql = """
            SELECT f.faceid, f.accountid, p.student_id, p.full_name,
                   f.model_name, f.model_version, f.dimension,
                   f.embedding_vector
            FROM face_embedding f
            LEFT JOIN personal_info p ON p.accountid = f.accountid
            WHERE f.is_active = TRUE AND f.model_name = %s
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (model_name,))
            rows = cur.fetchall()

        print(f"  [db] model={model_name}  rows_fetched={len(rows)}")

        out: list[EmbeddingRow] = []
        for face_id, account_id, student_id, full_name, m_name, m_ver, dim, vec in rows:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.size != dim:
                print(f"  [db] WARNING: skipping faceid={face_id} "
                      f"— vector has {arr.size} floats, expected {dim}")
                continue
            out.append(
                EmbeddingRow(
                    face_id=face_id,
                    account_id=account_id,
                    student_id=student_id,
                    full_name=full_name,
                    model_name=m_name,
                    model_version=m_ver,
                    dimension=dim,
                    vector=arr,
                )
            )
        return out

    # ── Read: lookup helpers ────────────────────────────────────────────
    def find_account_by_student_id(self, student_id: str) -> int | None:
        sql = "SELECT accountid FROM personal_info WHERE student_id = %s LIMIT 1"
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (student_id,))
            row = cur.fetchone()
            return row[0] if row else None

    # ── Write: persist a newly enrolled embedding ───────────────────────
    def deactivate_embeddings(
        self, account_id: int, model_name: str
    ) -> int:
        sql = """
            UPDATE face_embedding
            SET is_active = FALSE, updated_at = NOW()
            WHERE accountid = %s AND model_name = %s AND is_active = TRUE
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (account_id, model_name))
            return cur.rowcount

    def insert_embedding(
        self,
        account_id: int,
        vec: np.ndarray,
        model_name: str,
        model_version: str,
        retention_days: int,
    ) -> int:
        sql = """
            INSERT INTO face_embedding
              (accountid, embedding_vector, model_name, model_version, dimension,
               is_active, consent_given_at, retention_until)
            VALUES (%s, %s, %s, %s, %s, TRUE, NOW(), %s)
            RETURNING faceid
        """
        vec = np.asarray(vec, dtype=np.float32)
        retention = dt.date.today() + dt.timedelta(days=retention_days)
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                sql,
                (
                    account_id,
                    vec,
                    model_name,
                    model_version,
                    int(vec.size),
                    retention,
                ),
            )
            return cur.fetchone()[0]

    # ── Write: attendance record upsert (prototype helper) ──────────────
    def mark_attendance(
        self,
        session_id: int,
        account_id: int,
        status: str,
    ) -> None:
        """Best-effort insert of an attendance record, ignoring duplicates."""
        sql = """
            INSERT INTO attendance_record (attendancesessionid, accountid, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (attendancesessionid, accountid) DO NOTHING
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, (session_id, account_id, status))

    def get_or_open_session(self, course_id: int) -> int:
        """Prototype helper — return the currently-active session for a
        course or open a fresh one. Real scheduling lives in the backend."""
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                """
                SELECT attendancesessionid FROM attendance_session
                WHERE courseid = %s AND status = 'active'
                ORDER BY start_time DESC LIMIT 1
                """,
                (course_id,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute(
                """
                INSERT INTO attendance_session (courseid, start_time, status)
                VALUES (%s, NOW(), 'active') RETURNING attendancesessionid
                """,
                (course_id,),
            )
            return cur.fetchone()[0]
