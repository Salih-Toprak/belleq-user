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
    ) -> None:
        self._user_id = user_id
        self._lifecycle = lifecycle_retriever
        self._global_store = global_store
        self._settings = settings

    def bind_settings(self, settings: "Settings") -> None:
        """runtime_config güncellemelerinde ayar referansını yeniler."""
        self._settings = settings

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

        try:
            for doc in docs:
                doc_id = (doc.metadata or {}).get("doc_id")
                if doc_id:
                    self._global_store.increment_fetch(str(doc_id), self._user_id)
        except Exception:
            logger.warning("increment_fetch_failed", exc_info=True)

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

    async def close(self) -> None:
        pass
