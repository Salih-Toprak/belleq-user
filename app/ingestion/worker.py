"""Ingestion worker: claims queued jobs, chunks + embeds, writes to the KB.

Text extraction happens at enqueue time (so a bad upload fails fast with user
feedback); the worker handles the heavy, decoupled part — chunk → embed →
Qdrant upsert → GlobalDocStore register — via the shared ``KBWriter``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ingestion.chunker import chunk_text

if TYPE_CHECKING:
    from app.conversation.kb_writer import KBWriter
    from app.ingestion.queue import IngestionQueue
    from app.ingestion.models import IngestionJob

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Drains the ingestion queue, one job at a time."""

    def __init__(self, queue: "IngestionQueue", kb_writer: "KBWriter") -> None:
        self._queue = queue
        self._kb_writer = kb_writer

    def run_pending(self, max_jobs: int = 25) -> dict[str, int]:
        """Process up to ``max_jobs`` due jobs. Returns counts."""
        processed = failed = 0
        for _ in range(max(1, max_jobs)):
            job = self._queue.claim_next()
            if job is None:
                break
            try:
                points = self._process(job)
                self._queue.complete(job.job_id)
                processed += 1
                logger.info("ingestion_job_done job=%s kind=%s points=%d", job.job_id, job.kind, points)
            except Exception as exc:  # noqa: BLE001
                status = self._queue.fail(job.job_id, str(exc))
                failed += 1
                logger.warning("ingestion_job_failed job=%s status=%s err=%s", job.job_id, status, exc)
        return {"processed": processed, "failed": failed}

    def _process(self, job: "IngestionJob") -> int:
        p = job.payload or {}
        text = (p.get("text") or "").strip()
        if not text:
            raise ValueError("job has no text to ingest")

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("text produced no chunks")

        return self._kb_writer.write_chunks(
            doc_id=p["doc_id"],
            doc_title=p.get("doc_title") or p["doc_id"],
            doc_path=p.get("doc_path") or p["doc_id"],
            source=p.get("source") or job.kind,
            chunks=chunks,
            department=p.get("department") or "general",
            extra_payload=p.get("extra") or {},
            replace=bool(p.get("replace")),
        )
