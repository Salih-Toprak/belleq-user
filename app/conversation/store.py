"""Konuşma arşivi: stdlib sqlite3 ile kalıcı depolama.

rag-wiki ``belleq.db`` şemasını sahiplendiği için bu depo AYRI bir dosyaya
(``conversations.db``) yazar. Tüm metotlar eşzamanlıdır; çağıranlar gerektiğinde
``asyncio.to_thread`` ile sarmalar (bkz. docs_routes.py deseni).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.conversation.models import (
    ConversationSession,
    ConversationTurn,
    STATUS_CLOSED,
    STATUS_OPEN,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return _utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class ConversationStore:
    """SQLite-backed raw conversation archive (sessions + turns)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # check_same_thread=False so the single connection can be reused from
        # the threads asyncio.to_thread dispatches to; a lock serialises writes.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()
        logger.info("conversation_store_initialized db_path=%s", db_path)

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id       TEXT PRIMARY KEY,
                    source           TEXT NOT NULL DEFAULT '',
                    status           TEXT NOT NULL DEFAULT 'open',
                    exchange_count   INTEGER NOT NULL DEFAULT 0,
                    started_at       TEXT NOT NULL,
                    last_activity_at TEXT NOT NULL,
                    metadata_json    TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id       TEXT PRIMARY KEY,
                    session_id    TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    content       TEXT NOT NULL DEFAULT '',
                    source        TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON conversation_turns(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                    ON conversation_sessions(status, last_activity_at);
                """
            )
            self._conn.commit()

    # --- sessions -----------------------------------------------------

    def ensure_session(
        self,
        session_id: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create the session if it does not exist (idempotent)."""
        now = _iso(_utcnow())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversation_sessions
                    (session_id, source, status, exchange_count,
                     started_at, last_activity_at, metadata_json)
                VALUES (?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (
                    session_id,
                    source,
                    STATUS_OPEN,
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

    def touch_session(self, session_id: str, *, exchange_delta: int = 0) -> None:
        """Bump last_activity_at (and optionally exchange_count)."""
        now = _iso(_utcnow())
        with self._lock:
            self._conn.execute(
                """
                UPDATE conversation_sessions
                   SET last_activity_at = ?,
                       exchange_count = exchange_count + ?
                 WHERE session_id = ?
                """,
                (now, int(exchange_delta), session_id),
            )
            self._conn.commit()

    def mark_session(self, session_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversation_sessions SET status = ? WHERE session_id = ?",
                (status, session_id),
            )
            self._conn.commit()

    def close_session(self, session_id: str) -> None:
        self.mark_session(session_id, STATUS_CLOSED)

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[ConversationSession]:
        sql = "SELECT * FROM conversation_sessions"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY last_activity_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_idle_open_sessions(self, idle_before: datetime) -> list[ConversationSession]:
        """Open sessions whose last activity is older than ``idle_before``."""
        cutoff = _iso(idle_before)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM conversation_sessions
                 WHERE status = ? AND last_activity_at < ?
                 ORDER BY last_activity_at ASC
                """,
                (STATUS_OPEN, cutoff),
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def count_exchanges(self, session_id: str) -> int:
        s = self.get_session(session_id)
        return s.exchange_count if s else 0

    # --- turns --------------------------------------------------------

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        turn_id = uuid.uuid4().hex
        now = _iso(_utcnow())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversation_turns
                    (turn_id, session_id, role, content, source, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    role,
                    content or "",
                    source,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()
        return turn_id

    def get_session_turns(self, session_id: str, limit: int = 1000) -> list[ConversationTurn]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM conversation_turns
                 WHERE session_id = ?
                 ORDER BY created_at ASC, rowid ASC
                 LIMIT ?
                """,
                (session_id, int(limit)),
            ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    # --- stats --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            session_total = self._conn.execute(
                "SELECT COUNT(*) FROM conversation_sessions"
            ).fetchone()[0]
            turn_total = self._conn.execute(
                "SELECT COUNT(*) FROM conversation_turns"
            ).fetchone()[0]
            by_status_rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM conversation_sessions GROUP BY status"
            ).fetchall()
        return {
            "sessions": int(session_total),
            "turns": int(turn_total),
            "sessions_by_status": {r[0]: int(r[1]) for r in by_status_rows},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- row mapping --------------------------------------------------

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ConversationSession:
        return ConversationSession(
            session_id=row["session_id"],
            source=row["source"] or "",
            status=row["status"] or STATUS_OPEN,
            exchange_count=int(row["exchange_count"] or 0),
            started_at=_parse_iso(row["started_at"]),
            last_activity_at=_parse_iso(row["last_activity_at"]),
            metadata=_loads(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> ConversationTurn:
        return ConversationTurn(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"] or "",
            source=row["source"] or "",
            created_at=_parse_iso(row["created_at"]),
            metadata=_loads(row["metadata_json"]),
        )
