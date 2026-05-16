"""RagWikiRetriever sarmalayıcısı ve vektör arama adaptörü."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# ── Persistent background event loop ────────────────────────────────────────
# All async clients (httpx for embeddings, AsyncQdrantClient) must live on a
# single event loop that never closes.  asyncio.run() creates and DESTROYS
# loops, corrupting connection pools.  This module-level loop runs forever in
# a daemon thread; sync code schedules coroutines on it via
# run_coroutine_threadsafe and awaits the future.

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_lock = threading.Lock()


def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    """Return (and lazily create) a long-lived background event loop."""
    global _bg_loop
    if _bg_loop is not None and not _bg_loop.is_closed():
        return _bg_loop
    with _bg_lock:
        if _bg_loop is not None and not _bg_loop.is_closed():
            return _bg_loop
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True, name="persistent-loop")
        t.start()
        _bg_loop = loop
        logger.debug("persistent_background_loop_started")
        return _bg_loop


class VectorDBRetrieverAdapter(BaseRetriever):
    """
    Vektör veritabanı adaptörünü LangChain BaseRetriever arayüzüne bağlar.
    """

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        vectordb: Any,
        embedder: Any,
        collection_name: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_vectordb", vectordb)
        object.__setattr__(self, "_embedder", embedder)
        object.__setattr__(self, "_collection_name", collection_name)
        object.__setattr__(self, "_top_k", top_k)

    @property
    def top_k(self) -> int:
        return object.__getattribute__(self, "_top_k")

    @top_k.setter
    def top_k(self, value: int) -> None:
        object.__setattr__(self, "_top_k", max(1, int(value)))

    async def _fetch_async(self, query: str) -> list[Document]:
        vd = object.__getattribute__(self, "_vectordb")
        emb = object.__getattribute__(self, "_embedder")
        collection = object.__getattribute__(self, "_collection_name")
        top_k = object.__getattribute__(self, "_top_k")
        if vd is None or emb is None:
            logger.warning("vector_retriever_skip vd_or_embedder_none")
            return []
        try:
            # Use the sync embed_query wrapper (httpx.Client, no event loop issues)
            # to compute the query vector. This is safe even when called from
            # the persistent background loop because it doesn't touch the
            # async client at all.
            loop = asyncio.get_running_loop()
            qvec = await loop.run_in_executor(None, emb.embed_query, query)
        except Exception:
            logger.exception("vector_retriever_embed_failed")
            return []
        try:
            hits = await vd.search(
                collection_name=collection,
                query_vector=qvec,
                top_k=top_k,
                filters=None,
            )
        except Exception:
            logger.exception("vector_retriever_search_failed")
            return []
        docs: list[Document] = []
        for h in hits:
            payload = dict(h.get("payload") or {})
            text = (
                payload.get("text")
                or payload.get("page_content")
                or payload.get("content")
                or ""
            )
            meta = {**payload, "score": float(h.get("score") or 0.0)}
            docs.append(Document(page_content=str(text), metadata=meta))
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        loop = _get_persistent_loop()
        future = asyncio.run_coroutine_threadsafe(self._fetch_async(query), loop)
        return future.result(timeout=60)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return await self._fetch_async(query)


class LifecycleRetriever:
    """
    RagWikiRetriever'ı user_id, yapılandırma ve vektör adaptörü ile sarar.
    """

    def __init__(
        self,
        user_id: str,
        vectordb: Any,
        embedder: Any,
        state_store: Any,
        config: RagWikiRetrieverConfig,
        collection_name: str,
        top_k: int,
    ) -> None:
        self._user_id = user_id
        self._vectordb = vectordb
        self._embedder = embedder
        self._state_store = state_store
        self._config = config
        self._collection_name = collection_name
        self._top_k = top_k
        self._adapter: VectorDBRetrieverAdapter | None = None
        self._retriever: RagWikiRetriever | None = None

    def build(self) -> None:
        self._adapter = VectorDBRetrieverAdapter(
            vectordb=self._vectordb,
            embedder=self._embedder,
            collection_name=self._collection_name,
            top_k=self._top_k,
        )
        self._retriever = RagWikiRetriever(
            user_id=self._user_id,
            global_retriever=self._adapter,
            state_store=self._state_store,
            config=self._config,
            embedding_model=self._embedder,
        )
        logger.info(
            "lifecycle_retriever_built user_id=%s collection=%s top_k=%s",
            self._user_id,
            self._collection_name,
            self._top_k,
        )

    def reconfigure(self, settings: "Settings") -> None:
        """runtime_config ile değişen eşikleri ve top_k'yi uygular."""
        self._config.fetch_threshold = settings.rag_wiki_fetch_threshold
        self._top_k = settings.rag_wiki_top_k
        if self._adapter is not None:
            self._adapter.top_k = settings.rag_wiki_top_k
        if self._retriever is not None:
            self._retriever.config.fetch_threshold = settings.rag_wiki_fetch_threshold
            ctr = getattr(self._retriever, "_counter", None)
            if ctr is not None:
                ctr._threshold = settings.rag_wiki_fetch_threshold

    async def retrieve(self, query: str, top_k: int | None = None) -> tuple[list, Any]:
        if self._retriever is None:
            raise RuntimeError("LifecycleRetriever not built — call build() first")
        adapter = self._adapter
        old_top = None
        if adapter is not None and top_k is not None:
            old_top = adapter.top_k
            adapter.top_k = top_k
        try:
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, self._retriever.invoke, query)
        finally:
            if adapter is not None and old_top is not None:
                adapter.top_k = old_top
        provenance = self._retriever.last_provenance
        return docs, provenance

    def get_pending_suggestions(self) -> list:
        if self._retriever is None:
            return []
        return list(getattr(self._retriever, "pending_suggestions", []) or [])

    def accept_suggestion(self, doc_id: str) -> None:
        if self._retriever and hasattr(self._retriever, "accept_suggestion"):
            self._retriever.accept_suggestion(doc_id)

    def decline_suggestion(self, doc_id: str) -> None:
        if self._retriever and hasattr(self._retriever, "decline_suggestion"):
            self._retriever.decline_suggestion(doc_id)
