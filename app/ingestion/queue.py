"""SQLite-backed ingestion queue (per-container, separate ``ingestion.db``).

Single worker per container, so we don't need Postgres' ``SELECT … FOR UPDATE
SKIP LOCKED``; a ``threading.Lock`` + a status transition (queued → processing)
gives the same claim-once guarantee. Retries use exponential backoff via
``next_attempt_at``; exhausted jobs land in a ``dead`` (dead-letter) state.

All methods are synchronous; callers wrap with ``asyncio.to_thread`` as needed
(same pattern as ``conversation/store.py``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.ingestion.models import (
    STATUS_DEAD,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_QUEUED,
    IngestionJob,
)

logger = logging.getLogger(__name__)

# Exponential backoff: delay = min(BASE * 2**attempts, CAP) seconds.
_BACKOFF_BASE_SECONDS = 30
_BACKOFF_CAP_SECONDS = 3600


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class IngestionQueue:
    """Durable job queue for documents + MCP captures."""

    def __init__(self, db_path: str, max_attempts: int = 5) -> None:
        self._db_path = db_path
        self._max_attempts = max(1, int(max_attempts))
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()
        logger.info("ingestion_queue_initialized db_path=%s", db_path)

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id          TEXT PRIMARY KEY,
                    kind            TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    content_hash    TEXT NOT NULL,
                    payload         TEXT NOT NULL DEFAULT '{}',
                    attempts        INTEGER NOT NULL DEFAULT 0,
                    max_attempts    INTEGER NOT NULL DEFAULT 5,
                    last_error      TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    next_attempt_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status, next_attempt_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_hash ON ingestion_jobs(content_hash)"
            )
            self._conn.commit()

    def _row_to_job(self, r: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            job_id=r["job_id"],
            kind=r["kind"],
            status=r["status"],
            content_hash=r["content_hash"],
            payload=json.loads(r["payload"] or "{}"),
            attempts=r["attempts"],
            max_attempts=r["max_attempts"],
            last_error=r["last_error"] or "",
            created_at=_parse(r["created_at"]),
            updated_at=_parse(r["updated_at"]),
            next_attempt_at=_parse(r["next_attempt_at"]),
        )

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        content_hash: str,
        *,
        max_attempts: int | None = None,
    ) -> tuple[str, bool]:
        """Add a job. Returns (job_id, created).

        Content-hash dedup: if a job with the same hash already exists and is not
        in a terminal-failure state (``dead``), reuse it instead of duplicating.
        A previously ``done`` doc is skipped (created=False) — already in the KB.
        """
        now = _utcnow()
        with self._lock:
            existing = self._conn.execute(
                "SELECT job_id FROM ingestion_jobs WHERE content_hash=? AND status!=? LIMIT 1",
                (content_hash, STATUS_DEAD),
            ).fetchone()
            if existing:
                return existing["job_id"], False

            job_id = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO ingestion_jobs
                    (job_id, kind, status, content_hash, payload, attempts,
                     max_attempts, last_error, created_at, updated_at, next_attempt_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, '', ?, ?, ?)
                """,
                (
                    job_id, kind, STATUS_QUEUED, content_hash,
                    json.dumps(payload, ensure_ascii=False),
                    int(max_attempts or self._max_attempts),
                    _iso(now), _iso(now), _iso(now),
                ),
            )
            self._conn.commit()
            return job_id, True

    def claim_next(self) -> IngestionJob | None:
        """Atomically claim the oldest due queued/failed job → processing."""
        now = _utcnow()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE status IN (?, ?) AND next_attempt_at <= ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (STATUS_QUEUED, STATUS_FAILED, _iso(now)),
            ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE ingestion_jobs SET status=?, updated_at=? WHERE job_id=?",
                (STATUS_PROCESSING, _iso(now), row["job_id"]),
            )
            self._conn.commit()
            job = self._row_to_job(row)
            job.status = STATUS_PROCESSING
            return job

    def complete(self, job_id: str) -> None:
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                "UPDATE ingestion_jobs SET status=?, updated_at=?, last_error='' WHERE job_id=?",
                (STATUS_DONE, _iso(now), job_id),
            )
            self._conn.commit()

    def fail(self, job_id: str, error: str) -> str:
        """Record a failure. Retries with backoff until max_attempts → dead.

        Returns the resulting status (``failed`` if it will retry, ``dead`` if
        exhausted).
        """
        now = _utcnow()
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts, max_attempts FROM ingestion_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if not row:
                return STATUS_DEAD
            attempts = row["attempts"] + 1
            if attempts >= row["max_attempts"]:
                self._conn.execute(
                    "UPDATE ingestion_jobs SET status=?, attempts=?, last_error=?, updated_at=? WHERE job_id=?",
                    (STATUS_DEAD, attempts, error[:1000], _iso(now), job_id),
                )
                self._conn.commit()
                logger.warning("ingestion_job_dead job=%s attempts=%d err=%s", job_id, attempts, error[:200])
                return STATUS_DEAD
            delay = min(_BACKOFF_BASE_SECONDS * (2 ** (attempts - 1)), _BACKOFF_CAP_SECONDS)
            self._conn.execute(
                "UPDATE ingestion_jobs SET status=?, attempts=?, last_error=?, updated_at=?, next_attempt_at=? WHERE job_id=?",
                (STATUS_FAILED, attempts, error[:1000], _iso(now), _iso(now + timedelta(seconds=delay)), job_id),
            )
            self._conn.commit()
            return STATUS_FAILED

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, status: str | None = None, limit: int = 100) -> list[IngestionJob]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM ingestion_jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) c FROM ingestion_jobs GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["c"] for r in rows}
        return {
            "queued": counts.get(STATUS_QUEUED, 0),
            "processing": counts.get(STATUS_PROCESSING, 0),
            "done": counts.get(STATUS_DONE, 0),
            "failed": counts.get(STATUS_FAILED, 0),
            "dead": counts.get(STATUS_DEAD, 0),
            "total": sum(counts.values()),
        }

    def retry_dead(self, job_id: str) -> bool:
        """Requeue a dead-lettered job (admin action)."""
        now = _utcnow()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE ingestion_jobs SET status=?, attempts=0, last_error='', updated_at=?, next_attempt_at=? WHERE job_id=? AND status=?",
                (STATUS_QUEUED, _iso(now), _iso(now), job_id, STATUS_DEAD),
            )
            self._conn.commit()
            return cur.rowcount > 0
