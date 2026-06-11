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
        """Embed + upsert one doc (one chunk per fact). Returns points written."""
        clean = [f.strip() for f in facts if f and f.strip()]
        if not clean:
            return 0
        if not self.available():
            logger.warning("kb_writer_unavailable session=%s", session.session_id)
            return 0

        doc_id = f"conv-{session.session_id}"
        doc_title = f"Conversation {session.session_id}"
        doc_path = f"conversation:{session.session_id}"
        now = datetime.now(timezone.utc).isoformat()
        total = len(clean)

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
                        "source": "conversation",
                        "chunk_index": i,
                        "total_chunks": total,
                        "department": "general",
                        "indexed_at": now,
                        "ac_source_id": "conversation",
                        "ac_channels": [],
                        "ac_page_ids": [],
                        "ac_departments": ["general"],
                        "text": text,
                        "session_id": session.session_id,
                    },
                }
            )

        # 3) Upsert via the retriever's persistent loop (don't spin a new loop —
        #    that corrupts the shared async qdrant client; see retriever.py).
        loop = _get_persistent_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self._vectordb.upsert(self._collection, points), loop
        )
        written = int(fut.result(timeout=60))

        # 4) Register the doc so it shows in the dashboard + gets lifecycle state.
        try:
            self._register_global_doc(doc_id, doc_title, doc_path, clean, point_ids, now)
        except Exception:  # noqa: BLE001
            logger.warning("global_store_register_failed doc_id=%s", doc_id, exc_info=True)

        logger.info(
            "kb_facts_written session=%s doc_id=%s facts=%d points=%d",
            session.session_id,
            doc_id,
            total,
            written,
        )
        return written

    def _register_global_doc(
        self,
        doc_id: str,
        doc_title: str,
        doc_path: str,
        facts: list[str],
        point_ids: list[str],
        now_iso: str,
    ) -> None:
        from rag_wiki.storage.global_store import GlobalDocRecord

        now = datetime.fromisoformat(now_iso)
        self._global_store.upsert(
            GlobalDocRecord(
                doc_id=doc_id,
                source="conversation",
                department="general",
                doc_title=doc_title,
                doc_path=doc_path,
                ingested_at=now,
                last_updated_at=now,
                chunk_count=len(facts),
                doc_size_chars=sum(len(f) for f in facts),
                qdrant_ids=",".join(point_ids),
            )
        )
