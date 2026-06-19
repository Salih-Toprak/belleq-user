"""APScheduler-driven drain of the ingestion queue (mirrors SessionManager)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

if TYPE_CHECKING:
    from app.ingestion.worker import IngestionWorker

logger = logging.getLogger(__name__)


class IngestionScheduler:
    """Periodically runs the ingestion worker; embeds/Qdrant are blocking so the
    drain runs in a thread to keep the event loop responsive."""

    def __init__(self, worker: "IngestionWorker", interval_seconds: int = 20) -> None:
        self._worker = worker
        self._interval = max(5, int(interval_seconds))
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._scheduler.add_job(
            self._drain,
            trigger="interval",
            seconds=self._interval,
            id="ingestion_drain",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info("ingestion_scheduler_started interval_s=%d", self._interval)

    async def _drain(self) -> None:
        try:
            await asyncio.to_thread(self._worker.run_pending)
        except Exception:  # noqa: BLE001
            logger.error("ingestion_drain_failed", exc_info=True)

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
