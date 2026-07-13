"""Activity-gated retention: archive stale KB documents, optionally purge.

Keeps the knowledge base current without punishing absence. The staleness
clock counts **active days** — days on which this context was actually used
(a query, a saved exchange, an upload) — never calendar days. A user who
disappears for six months accrues zero active days, so nothing is archived
while they're gone; the clock only ticks while they are using Belleq and a
document keeps NOT being useful.

Two stages, both per-document:

  live ──(archive_after active days without a fetch)──▶ ARCHIVED (soft)
  ARCHIVED ──(purge_after further active days, opt-in)──▶ deleted from Qdrant

Archived docs get ``archived: true`` on every chunk payload; retrieval and
recall exclude them, but the vectors + text remain and one call restores
them. Purge is off by default (``retention_purge_enabled``) and physically
removes the points.

All vector-db coroutines are marshalled to the retriever's persistent loop
(the async client is bound to it — see retriever.py / recall.py). Tests pass
a direct-await ``runner`` instead.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)

Runner = Callable[[Awaitable[Any]], Awaitable[Any]]

_SCROLL_PAGE = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


async def _persistent_runner(coro: Awaitable[Any]) -> Any:
    from app.lifecycle.retriever import _get_persistent_loop

    loop = _get_persistent_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return await asyncio.wrap_future(fut)


# ── Activity tracking ────────────────────────────────────────────────

class ActivityTracker:
    """Which days was this context used, and when was each doc last fetched.

    Plain sqlite3 (own file), synchronous and cheap — call sites either run
    in a thread or tolerate ~1ms. Connection per call keeps it thread-safe.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS active_days ("
                " day TEXT PRIMARY KEY, events INTEGER NOT NULL DEFAULT 0)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS doc_activity ("
                " doc_id TEXT PRIMARY KEY,"
                " last_fetched_at TEXT,"
                " fetch_count INTEGER NOT NULL DEFAULT 0)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS retention_meta ("
                " key TEXT PRIMARY KEY, value TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def mark_active(self, now: datetime | None = None) -> None:
        day = (now or _utcnow()).date().isoformat()
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO active_days (day, events) VALUES (?, 1)"
                    " ON CONFLICT(day) DO UPDATE SET events = events + 1",
                    (day,),
                )
        except sqlite3.Error:
            logger.debug("activity_mark_failed", exc_info=True)

    def record_fetch(self, doc_ids: list[str], now: datetime | None = None) -> None:
        ts = _iso(now or _utcnow())
        try:
            with self._conn() as c:
                for doc_id in doc_ids:
                    if not doc_id:
                        continue
                    c.execute(
                        "INSERT INTO doc_activity (doc_id, last_fetched_at, fetch_count)"
                        " VALUES (?, ?, 1)"
                        " ON CONFLICT(doc_id) DO UPDATE SET"
                        "  last_fetched_at = excluded.last_fetched_at,"
                        "  fetch_count = fetch_count + 1",
                        (str(doc_id), ts),
                    )
        except sqlite3.Error:
            logger.debug("activity_record_fetch_failed", exc_info=True)

    def last_fetch(self, doc_id: str) -> datetime | None:
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT last_fetched_at FROM doc_activity WHERE doc_id = ?",
                    (str(doc_id),),
                ).fetchone()
            return _parse_ts(row[0]) if row else None
        except sqlite3.Error:
            return None

    def active_days_since(self, ref: datetime, now: datetime | None = None) -> int:
        """Distinct used-days strictly after ``ref``'s date, up to today.

        This is the retention clock: 0 while the user is away, +1 per day of
        real use. ``ref``'s own day doesn't count — a doc saved and never
        touched again on a busy day isn't 'one active day stale' yet.
        """
        today = (now or _utcnow()).date().isoformat()
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT COUNT(*) FROM active_days WHERE day > ? AND day <= ?",
                    (ref.date().isoformat(), today),
                ).fetchone()
            return int(row[0] or 0)
        except sqlite3.Error:
            return 0

    def total_active_days(self) -> int:
        try:
            with self._conn() as c:
                row = c.execute("SELECT COUNT(*) FROM active_days").fetchone()
            return int(row[0] or 0)
        except sqlite3.Error:
            return 0

    # last-run summary (shown on the dashboard)
    def set_meta(self, key: str, value: str) -> None:
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO retention_meta (key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
        except sqlite3.Error:
            logger.debug("retention_meta_set_failed", exc_info=True)

    def get_meta(self, key: str) -> str | None:
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT value FROM retention_meta WHERE key = ?", (key,)
                ).fetchone()
            return row[0] if row else None
        except sqlite3.Error:
            return None

    # Pending storage to release upstream — accumulated when purge deletes
    # points here, drained by the backend (which owns the usage meter) and then
    # zeroed. The container can't reach the backend directly, so it parks the
    # freed-byte total and the backend claims it out-of-band.
    def add_pending_release(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO retention_meta (key, value) VALUES ('pending_release_bytes', ?)"
                    " ON CONFLICT(key) DO UPDATE SET"
                    "  value = CAST(retention_meta.value AS INTEGER) + ?",
                    (str(int(nbytes)), int(nbytes)),
                )
        except sqlite3.Error:
            logger.debug("pending_release_add_failed", exc_info=True)

    def pending_release(self) -> int:
        raw = self.get_meta("pending_release_bytes")
        try:
            return int(raw) if raw else 0
        except (TypeError, ValueError):
            return 0

    def claim_pending_release(self) -> int:
        """Atomically read the pending byte total and subtract exactly that much
        (so a concurrent purge adding bytes between the read and write is never
        lost). Returns the claimed amount for the backend to release."""
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT value FROM retention_meta WHERE key = 'pending_release_bytes'"
                ).fetchone()
                val = int(row[0]) if row and row[0] else 0
                if val:
                    c.execute(
                        "UPDATE retention_meta SET value = CAST(value AS INTEGER) - ?"
                        " WHERE key = 'pending_release_bytes'",
                        (val,),
                    )
                return val
        except sqlite3.Error:
            logger.debug("pending_release_claim_failed", exc_info=True)
            return 0


# ── Sweep ────────────────────────────────────────────────────────────

# rag_wiki lifecycle states a user has explicitly committed to — never
# auto-archive these. Matched case-insensitively against the state table's
# ``user_state`` column so we don't depend on importing the rag_wiki enum.
_PROTECTED_STATES = ("pinned", "claimed")
# Candidate column names for the document id in the rag_wiki state table
# (reflected, since the library owns the schema). First present one wins.
_DOC_ID_COLUMNS = ("doc_id", "document_id", "id", "key")


class RetentionSweeper:
    def __init__(
        self,
        vectordb: Any,
        collection_name: str,
        tracker: ActivityTracker,
        runner: Runner | None = None,
        state_store: Any = None,
        user_id: str = "",
    ) -> None:
        self._vectordb = vectordb
        self._collection = collection_name
        self._tracker = tracker
        self._run: Runner = runner or _persistent_runner
        self._state_store = state_store
        self._user_id = user_id

    def _protected_doc_ids(self) -> set[str]:
        """Doc ids the user has pinned/claimed — read straight from the rag_wiki
        SQLite state table by reflection (same access pattern as
        lifecycle_stats). Fails safe: any problem returns an empty set, so a
        read failure never *causes* a wrong archive on its own, but pairs with
        the archived-doc being reversible.
        """
        store = self._state_store
        if store is None:
            return set()
        try:
            from sqlalchemy import func, select

            t = store._table
            cols = set(t.c.keys())
            doc_col = next((c for c in _DOC_ID_COLUMNS if c in cols), None)
            if doc_col is None:
                logger.warning(
                    "retention_protected_no_doc_col cols=%s — pinned/claimed "
                    "protection disabled for this table", sorted(cols),
                )
                return set()
            state_col = "user_state" if "user_state" in cols else None
            if state_col is None:
                return set()
            col = t.c[doc_col]
            with store._engine.connect() as conn:
                rows = conn.execute(
                    select(col).where(
                        t.c.user_id == self._user_id,
                        func.lower(t.c[state_col]).in_(_PROTECTED_STATES),
                    )
                ).fetchall()
            return {str(r[0]) for r in rows if r[0] is not None}
        except Exception:  # noqa: BLE001 — protection is best-effort, never fatal
            logger.debug("retention_protected_read_failed", exc_info=True)
            return set()

    async def _scroll_all(self) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            page = await self._run(
                self._vectordb.scroll(
                    self._collection, filters=None, limit=_SCROLL_PAGE, offset=offset
                )
            )
            rows.extend(page)
            if len(page) < _SCROLL_PAGE:
                return rows
            offset += _SCROLL_PAGE

    @staticmethod
    def _group_docs(points: list[dict]) -> dict[str, dict]:
        """Chunk points → one entry per doc_id with the fields retention needs."""
        docs: dict[str, dict] = {}
        for p in points:
            payload = p.get("payload") or {}
            doc_id = str(payload.get("doc_id") or "")
            if not doc_id:
                continue
            d = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "doc_title": payload.get("doc_title") or "",
                    "source": payload.get("source") or "",
                    "indexed_at": None,
                    "archived": False,
                    "archived_at": None,
                    "pinned": False,
                    "chunks": 0,
                    "bytes": 0,
                },
            )
            d["chunks"] += 1
            d["bytes"] += len(str(payload.get("text") or "").encode())
            ts = _parse_ts(payload.get("indexed_at"))
            if ts and (d["indexed_at"] is None or ts < d["indexed_at"]):
                d["indexed_at"] = ts
            if payload.get("archived") is True:
                d["archived"] = True
                aat = _parse_ts(payload.get("archived_at"))
                if aat and (d["archived_at"] is None or aat < d["archived_at"]):
                    d["archived_at"] = aat
            if payload.get("pinned") is True:
                d["pinned"] = True
        return docs

    async def sweep(self, dry_run: bool = False, now: datetime | None = None) -> dict:
        """One retention pass. Returns a summary dict (also persisted)."""
        from app import config

        eff = config.settings
        now = now or _utcnow()

        if self._vectordb is None:
            return {"status": "skipped", "reason": "no vector db"}
        if not eff.retention_enabled and not dry_run:
            return {"status": "skipped", "reason": "retention disabled"}

        archive_after = max(1, int(eff.retention_archive_after_days))
        purge_after = max(1, int(eff.retention_purge_after_days))
        purge_enabled = bool(eff.retention_purge_enabled)

        points = await self._scroll_all()
        docs = self._group_docs(points)
        # Docs the user pinned/claimed are never auto-archived — either via the
        # rag_wiki state machine (CLAIMED/PINNED) or the belleq-level `pinned`
        # payload flag set from the dashboard.
        protected = self._protected_doc_ids()
        protected |= {d["doc_id"] for d in docs.values() if d.get("pinned")}

        to_archive: list[dict] = []
        to_purge: list[dict] = []
        for d in docs.values():
            if d["doc_id"] in protected:
                continue
            if d["archived"]:
                ref = d["archived_at"] or d["indexed_at"]
                if (
                    purge_enabled
                    and ref is not None
                    and self._tracker.active_days_since(ref, now) >= purge_after
                ):
                    to_purge.append(d)
                continue
            # Unknown age and never fetched → not eligible (safe default).
            last_ref = self._tracker.last_fetch(d["doc_id"]) or d["indexed_at"]
            if last_ref is None:
                continue
            if self._tracker.active_days_since(last_ref, now) >= archive_after:
                to_archive.append(d)

        archived_ids: list[str] = []
        purged_ids: list[str] = []
        bytes_freed = 0
        if not dry_run:
            stamp = _iso(now)
            for d in to_archive:
                try:
                    await self._run(
                        self._vectordb.set_payload_by_filter(
                            self._collection,
                            {"must": [{"field": "doc_id", "value": d["doc_id"]}]},
                            {"archived": True, "archived_at": stamp},
                        )
                    )
                    archived_ids.append(d["doc_id"])
                except Exception:  # noqa: BLE001 — one bad doc must not stop the pass
                    logger.warning("retention_archive_failed doc=%s", d["doc_id"], exc_info=True)
            for d in to_purge:
                try:
                    await self._run(
                        self._vectordb.delete_by_doc_id(self._collection, d["doc_id"])
                    )
                    purged_ids.append(d["doc_id"])
                    bytes_freed += int(d["bytes"])
                except Exception:  # noqa: BLE001
                    logger.warning("retention_purge_failed doc=%s", d["doc_id"], exc_info=True)
            # Park freed bytes for the backend to release from the usage meter
            # (the container can't reach the backend itself).
            if bytes_freed > 0:
                self._tracker.add_pending_release(bytes_freed)

        summary = {
            "status": "dry_run" if dry_run else "ok",
            "ran_at": _iso(now),
            "docs_examined": len(docs),
            "protected_docs": len(protected),
            "archive_candidates": [
                {"doc_id": d["doc_id"], "doc_title": d["doc_title"], "source": d["source"]}
                for d in to_archive
            ],
            "purge_candidates": [
                {"doc_id": d["doc_id"], "doc_title": d["doc_title"], "source": d["source"]}
                for d in to_purge
            ],
            "archived": archived_ids,
            "purged": purged_ids,
            "bytes_freed": bytes_freed,
            "active_days_total": self._tracker.total_active_days(),
            "config": {
                "retention_enabled": eff.retention_enabled,
                "archive_after_days": archive_after,
                "purge_enabled": purge_enabled,
                "purge_after_days": purge_after,
            },
        }
        if not dry_run:
            import json

            self._tracker.set_meta("last_sweep", json.dumps(summary))
            logger.info(
                "retention_sweep docs=%d archived=%d purged=%d bytes_freed=%d",
                len(docs), len(archived_ids), len(purged_ids), bytes_freed,
            )
        return summary

    async def list_archived(self) -> list[dict]:
        """Archived docs, newest archive first."""
        points = await self._run(
            self._vectordb.scroll(
                self._collection,
                filters={"must": [{"field": "archived", "value": True}]},
                limit=2000,
            )
        )
        docs = self._group_docs(points)
        out = [d for d in docs.values() if d["archived"]]
        out.sort(key=lambda d: d["archived_at"] or _EPOCH_MIN, reverse=True)
        return [
            {
                "doc_id": d["doc_id"],
                "doc_title": d["doc_title"],
                "source": d["source"],
                "chunks": d["chunks"],
                "archived_at": _iso(d["archived_at"]) if d["archived_at"] else None,
            }
            for d in out
        ]

    async def restore(self, doc_id: str) -> int:
        """Un-archive a doc; it surfaces in retrieval again immediately."""
        n = await self._run(
            self._vectordb.set_payload_by_filter(
                self._collection,
                {"must": [{"field": "doc_id", "value": str(doc_id)}]},
                {"archived": False, "archived_at": None},
            )
        )
        # Restoring is a fetch-equivalent signal: the user wants it back, so
        # its staleness clock restarts from now.
        self._tracker.record_fetch([str(doc_id)])
        return int(n)

    def last_sweep(self) -> dict | None:
        raw = self._tracker.get_meta("last_sweep")
        if not raw:
            return None
        try:
            import json

            return json.loads(raw)
        except ValueError:
            return None


_EPOCH_MIN = datetime.min.replace(tzinfo=timezone.utc)


# ── Scheduler ────────────────────────────────────────────────────────

class RetentionScheduler:
    def __init__(self, sweeper: RetentionSweeper, interval_hours: int = 24) -> None:
        self._sweeper = sweeper
        self._scheduler = AsyncIOScheduler()
        self._interval = max(1, int(interval_hours))

    def start(self) -> None:
        self._scheduler.add_job(
            self._run,
            trigger="interval",
            hours=self._interval,
            id="retention_sweep",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info("retention_scheduler_started interval_hours=%d", self._interval)

    async def _run(self) -> None:
        try:
            await self._sweeper.sweep()
        except Exception:  # noqa: BLE001
            logger.error("retention_sweep_failed", exc_info=True)

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
