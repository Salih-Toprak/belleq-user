"""Çıkarılan konuşma gerçeklerini KB'ye (Qdrant + GlobalDocStore) yazar.

Her gerçek tek bir chunk'tır (gerçekler kısadır). Senkron çalışır:
- embedder.embed_documents (kısa ömürlü httpx.Client — herhangi bir thread'den güvenli)
- vektör upsert, retriever'ın kalıcı arka plan döngüsü üzerinden (paylaşılan async
  qdrant istemcisini bozmadan)
- rag-wiki GlobalDocStore'a kayıt (yaşam döngüsü için GLOBAL doc olarak görünür)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.lifecycle.retriever import _get_persistent_loop

if TYPE_CHECKING:
    from app.config import Settings
    from app.conversation.models import ConversationSession

logger = logging.getLogger(__name__)


def _make_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID point id (Qdrant requires UUID or uint)."""
    h = hashlib.sha256(f"{doc_id}::{chunk_index}".encode()).digest()
    return str(uuid.UUID(bytes=h[:16]))


class KBWriter:
    """Embeds conversation facts and upserts them into the KB collection."""

    def __init__(
        self,
        embedder: Any,
        vectordb: Any,
        global_store: Any,
        collection_name: str,
        settings: "Settings",
    ) -> None:
        self._embedder = embedder
        self._vectordb = vectordb
        self._global_store = global_store
        self._collection = collection_name
        self._settings = settings

    def available(self) -> bool:
        return self._embedder is not None and self._vectordb is not None

    def write_facts(self, session: "ConversationSession", facts: list[str]) -> int:
        """Embed + upsert one conversation doc (one chunk per fact)."""
        return self.write_chunks(
            doc_id=f"conv-{session.session_id}",
            doc_title=f"Conversation {session.session_id}",
            doc_path=f"conversation:{session.session_id}",
            source="conversation",
            chunks=facts,
            extra_payload={"session_id": session.session_id},
        )

    def write_chunks(
        self,
        *,
        doc_id: str,
        doc_title: str,
        doc_path: str,
        source: str,
        chunks: list[str],
        department: str = "general",
        extra_payload: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> int:
        """Embed + upsert a document (one or more chunks) into the KB.

        Shared by the conversation pipeline (one chunk per fact) and the
        ingestion pipeline (uploaded docs / MCP captures, many chunks). Returns
        the number of vector points written.
        """
        clean = [c.strip() for c in chunks if c and c.strip()]
        if not clean:
            return 0
        if not self.available():
            logger.warning("kb_writer_unavailable doc_id=%s", doc_id)
            return 0

        now = datetime.now(timezone.utc).isoformat()
        total = len(clean)
        extra = extra_payload or {}

        # 1) Embed (sync wrapper, safe from any thread).
        vectors = self._embedder.embed_documents(clean)

        # 2) Build points with standard payload (mirrors the master chunker).
        points = []
        point_ids = []
        for i, (text, vec) in enumerate(zip(clean, vectors)):
            pid = _make_point_id(doc_id, i)
            point_ids.append(pid)
            points.append(
                {
                    "id": pid,
                    "vector": vec,
                    "payload": {
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "doc_path": doc_path,
                        "source": source,
                        "chunk_index": i,
                        "total_chunks": total,
                        "department": department,
                        "indexed_at": now,
                        "ac_source_id": source,
                        "ac_channels": [],
                        "ac_page_ids": [],
                        "ac_departments": [department],
                        "text": text,
                        **extra,
                    },
                }
            )

        # 3) Ensure the collection exists, then upsert — both on the retriever's
        #    persistent loop (don't spin a new loop — that corrupts the shared
        #    async qdrant client; see retriever.py). A freshly provisioned context
        #    has a collection *name* but no Qdrant collection until its first
        #    write, so the upsert would 404 without this.
        loop = _get_persistent_loop()
        asyncio.run_coroutine_threadsafe(self._ensure_collection(), loop).result(timeout=30)
        # Replace-in-place: drop every existing chunk of this doc before writing
        # the new version, so an edited re-upload doesn't leave stale chunks
        # behind (deterministic point ids only overwrite matching indices — a
        # shorter new version would otherwise keep the old tail).
        if replace:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._vectordb.delete_by_doc_id(self._collection, doc_id), loop
                ).result(timeout=30)
            except Exception:  # noqa: BLE001 — a failed delete shouldn't block the write
                logger.warning("kb_replace_delete_failed doc_id=%s", doc_id, exc_info=True)
        fut = asyncio.run_coroutine_threadsafe(
            self._vectordb.upsert(self._collection, points), loop
        )
        written = int(fut.result(timeout=60))

        # 4) Register the doc so it shows in the dashboard + gets lifecycle state.
        try:
            self._register_global_doc(doc_id, doc_title, doc_path, clean, point_ids, now, source, department)
        except Exception:  # noqa: BLE001
            logger.warning("global_store_register_failed doc_id=%s", doc_id, exc_info=True)

        logger.info(
            "kb_chunks_written doc_id=%s source=%s chunks=%d points=%d",
            doc_id, source, total, written,
        )
        return written

    async def _ensure_collection(self) -> None:
        """Create the KB collection on demand if Qdrant doesn't have it yet.

        Contexts are provisioned with a collection *name* only; Qdrant holds
        nothing until the first write, so an empty context's first fact-write
        would 404. Create it with the embedder's vector size (must match what
        the master ingestion path and the query pipeline use for this context).
        """
        try:
            existing = await self._vectordb.list_collections()
            if self._collection in existing:
                return
            size = int(getattr(self._embedder, "vector_size", 0) or 0) or int(
                getattr(self._settings, "embedding_vector_size", 768) or 768
            )
            await self._vectordb.create_collection(self._collection, size, "Cosine")
            logger.info(
                "kb_collection_created collection=%s vector_size=%d",
                self._collection,
                size,
            )
        except Exception:  # noqa: BLE001
            # A race (another writer created it first) surfaces here as
            # "already exists" — harmless, the upsert below still succeeds.
            logger.warning(
                "kb_ensure_collection_failed collection=%s", self._collection, exc_info=True
            )

    def _register_global_doc(
        self,
        doc_id: str,
        doc_title: str,
        doc_path: str,
        chunks: list[str],
        point_ids: list[str],
        now_iso: str,
        source: str = "conversation",
        department: str = "general",
    ) -> None:
        from rag_wiki.storage.global_store import GlobalDocRecord

        now = datetime.fromisoformat(now_iso)
        self._global_store.upsert(
            GlobalDocRecord(
                doc_id=doc_id,
                source=source,
                department=department,
                doc_title=doc_title,
                doc_path=doc_path,
                ingested_at=now,
                last_updated_at=now,
                chunk_count=len(chunks),
                doc_size_chars=sum(len(c) for c in chunks),
                qdrant_ids=",".join(point_ids),
            )
        )
