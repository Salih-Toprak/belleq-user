"""Sorgu hattı: yaşam döngülü getirme — yalnızca retrieval."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from app.lifecycle.retriever import LifecycleRetriever

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class QueryPipeline:
    def __init__(
        self,
        user_id: str,
        lifecycle_retriever: LifecycleRetriever,
        global_store: Any,
        settings: "Settings",
        capture: Any = None,
        activity: Any = None,
    ) -> None:
        self._user_id = user_id
        self._lifecycle = lifecycle_retriever
        self._global_store = global_store
        self._settings = settings
        self._capture = capture
        self._activity = activity  # retention ActivityTracker (optional)

    def bind_settings(self, settings: "Settings") -> None:
        """runtime_config güncellemelerinde ayar referansını yeniler."""
        self._settings = settings

    def _mark_activity(self, fetched_doc_ids: list[str]) -> None:
        """Sync helper (runs in a thread): bump the retention activity clock."""
        self._activity.mark_active()
        if fetched_doc_ids:
            self._activity.record_fetch(fetched_doc_ids)

    async def query(
        self,
        user_message: str,
        top_k: int | None = None,
    ) -> dict:
        """
        Retrieve relevant chunks for user_message via rag-wiki lifecycle.
        Returns raw chunks with metadata. No LLM. No answer generation.
        """
        from app import config

        eff = config.settings
        start = time.monotonic()

        docs, provenance = await self._lifecycle.retrieve(
            user_message,
            top_k=top_k if top_k is not None else eff.rag_wiki_top_k,
        )

        # Passive query-stream capture — best-effort, never affects the result.
        if self._capture is not None and getattr(eff, "conversation_capture_enabled", False):
            try:
                import asyncio

                await asyncio.to_thread(self._capture.record_query, user_message)
            except Exception:
                logger.debug("query_stream_capture_failed", exc_info=True)

        try:
            for doc in docs:
                doc_id = (doc.metadata or {}).get("doc_id")
                if doc_id:
                    self._global_store.increment_fetch(str(doc_id), self._user_id)
        except Exception:
            logger.warning("increment_fetch_failed", exc_info=True)

        # Retention clock: this query marks today an active day, and every
        # fetched doc's staleness clock restarts.
        if self._activity is not None:
            try:
                import asyncio as _asyncio

                fetched = [
                    str((d.metadata or {}).get("doc_id"))
                    for d in docs
                    if (d.metadata or {}).get("doc_id")
                ]
                await _asyncio.to_thread(self._mark_activity, fetched)
            except Exception:
                logger.debug("activity_track_failed", exc_info=True)

        chunks = []
        for doc in docs:
            meta = doc.metadata or {}
            chunks.append({
                "text": doc.page_content,
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "source": meta.get("source", ""),
                "channel": meta.get("channel", ""),
                "department": meta.get("department", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "state": meta.get("state", "GLOBAL"),
                "metadata": meta,
            })

        prov = {}
        if provenance and hasattr(provenance, "sources"):
            prov = {
                "cache_hits": sum(1 for s in provenance.sources
                                  if getattr(s, "from_cache", False)),
                "global_hits": sum(1 for s in provenance.sources
                                  if not getattr(s, "from_cache", False)),
                "total_retrieved": len(provenance.sources),
            }

        latency_ms = int((time.monotonic() - start) * 1000)

        return {
            "chunks": chunks,
            "user_id": self._user_id,
            "query_id": str(uuid.uuid4()),
            "latency_ms": latency_ms,
            "provenance": prov,
        }

    async def recent_context(self, limit: int = 10) -> dict:
        """Return the most recently saved knowledge (conversation facts).

        Powers the zero-instruction `recall_context` MCP tool: a cheap, no-LLM
        primer the connected AI calls at the start of a chat to load what belleq
        already knows. Delegates to `app.conversation.recall.recent_facts`,
        which marshals vector-db access onto the retriever's persistent loop
        (the async qdrant client is bound to it — see retriever.py).
        """
        from app.conversation.recall import recent_facts

        return await recent_facts(
            self._global_store,
            self._lifecycle.vectordb,
            self._lifecycle.collection_name,
            limit,
        )

    async def close(self) -> None:
        pass
