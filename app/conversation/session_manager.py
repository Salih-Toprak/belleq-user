"""Oturum süpürücü: boşta kalan oturumları kapatır ve çıkarıma yönlendirir.

``lifecycle/decay_scheduler.py`` ile aynı AsyncIOScheduler desenini izler.
Her süpürmede:
- ``conversation_session_idle_minutes`` boyunca etkinlik olmayan açık oturumları kapatır,
- kapanan oturumu ``>= min_exchanges`` ise ``pending_extraction``, değilse ``skipped`` yapar,
- ardından bekleyen oturumlar için ``ExtractionWorker.run_pending()`` çağırır.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.conversation.models import STATUS_PENDING_EXTRACTION, STATUS_SKIPPED

if TYPE_CHECKING:
    from app.config import Settings
    from app.conversation.extraction import ExtractionWorker
    from app.conversation.store import ConversationStore

logger = logging.getLogger(__name__)


class SessionManager:
    """Periodically closes idle sessions and runs pending extraction."""

    def __init__(
        self,
        user_id: str,
        store: "ConversationStore",
        worker: "ExtractionWorker",
        settings: "Settings",
    ) -> None:
        self._user_id = user_id
        self._store = store
        self._worker = worker
        self._settings = settings
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        interval = max(1, int(self._settings.conversation_sweep_interval_minutes))
        self._scheduler.add_job(
            self._sweep,
            trigger="interval",
            minutes=interval,
            id="conversation_sweep",
            name=f"Conversation sweep for {self._user_id}",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "session_manager_started user=%s interval_min=%d idle_min=%d min_exchanges=%d",
            self._user_id,
            interval,
            self._settings.conversation_session_idle_minutes,
            self._settings.conversation_min_exchanges,
        )

    async def _sweep(self) -> None:
        # Run in a thread: the extraction step may call an LLM, the embedder,
        # and Qdrant synchronously, which must not block the event loop.
        try:
            await asyncio.to_thread(self.sweep_once)
        except Exception:  # noqa: BLE001
            logger.error("conversation_sweep_failed user=%s", self._user_id, exc_info=True)

    def sweep_once(self) -> dict[str, int]:
        """Run one sweep synchronously. Returns counts (also used by tests)."""
        idle_minutes = max(0, int(self._settings.conversation_session_idle_minutes))
        min_exchanges = max(1, int(self._settings.conversation_min_exchanges))
        idle_before = datetime.now(timezone.utc) - timedelta(minutes=idle_minutes)

        stale = self._store.get_idle_open_sessions(idle_before)
        pending = 0
        skipped = 0
        for session in stale:
            count = self._store.count_exchanges(session.session_id)
            if count >= min_exchanges:
                self._store.mark_session(session.session_id, STATUS_PENDING_EXTRACTION)
                pending += 1
            else:
                self._store.mark_session(session.session_id, STATUS_SKIPPED)
                skipped += 1

        extracted = self._worker.run_pending()
        if stale or extracted:
            logger.info(
                "conversation_sweep user=%s closed=%d pending=%d skipped=%d extracted=%d",
                self._user_id,
                len(stale),
                pending,
                skipped,
                extracted,
            )
        return {
            "closed": len(stale),
            "pending": pending,
            "skipped": skipped,
            "extracted": extracted,
        }

    def flush_now(self, *, respect_min_exchanges: bool = False) -> dict[str, int]:
        """Force-close every open session immediately and run extraction now.

        Powers the manual ingestion endpoint/tool: lets a user push buffered
        conversation facts into the KB without waiting for the idle-gap sweep
        (and makes end-to-end testing fast). Unlike the periodic sweep, a manual
        flush does NOT enforce ``min_exchanges`` by default — it is an explicit
        request to ingest whatever is buffered; pass ``respect_min_exchanges``
        to apply the same threshold the sweep uses.
        """
        threshold = (
            max(1, int(self._settings.conversation_min_exchanges))
            if respect_min_exchanges
            else 1
        )
        # last_activity < cutoff selects every open session; a small future
        # cutoff guards against a just-recorded turn being excluded by clock skew.
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=1)
        open_sessions = self._store.get_idle_open_sessions(cutoff)
        pending = 0
        skipped = 0
        for session in open_sessions:
            if self._store.count_exchanges(session.session_id) >= threshold:
                self._store.mark_session(session.session_id, STATUS_PENDING_EXTRACTION)
                pending += 1
            else:
                self._store.mark_session(session.session_id, STATUS_SKIPPED)
                skipped += 1

        extracted = self._worker.run_pending()
        logger.info(
            "conversation_flush user=%s closed=%d pending=%d skipped=%d extracted=%d",
            self._user_id,
            len(open_sessions),
            pending,
            skipped,
            extracted,
        )
        return {
            "closed": len(open_sessions),
            "pending": pending,
            "skipped": skipped,
            "extracted": extracted,
        }

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
