# Shared adapter — keep in sync with belleq-master/app/vectordb/
"""Vector database adapter contract and errors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VectorDBError(Exception):
    """Normalized error crossing the adapter boundary."""

    def __init__(self, message: str, backend: str, detail: str = "") -> None:
        self.backend = backend
        self.detail = detail
        super().__init__(message)
        logger.debug(
            "vectordb_error backend=%s message=%s detail=%s",
            backend,
            message,
            detail,
        )


class VectorDBAdapter(ABC):
    """Abstract vector database backend (Qdrant, Pinecone, ...)."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier string e.g. 'qdrant', 'pinecone'."""

    @abstractmethod
    async def health(self) -> dict:
        """
        Returns:
          {"status": "ok"|"error", "backend": str, "detail": str}
        Never raises — catches all exceptions internally.
        """

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """Returns list of collection/index names."""

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> dict:
        """
        Returns:
          {
            "name": str,
            "vector_count": int,
            "indexed_vector_count": int,
            "status": str,
            "vector_size": int,
            "distance": str
          }
        Raises VectorDBError if collection not found.
        """

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Creates a new collection/index. Raises VectorDBError on failure."""

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Deletes entire collection. Raises VectorDBError on failure."""

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: list[dict],
    ) -> int:
        """
        points: [{"id": str, "vector": list[float], "payload": dict}]
        Returns count of upserted points.
        Raises VectorDBError on failure.
        """

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Returns:
          [{"id": str, "score": float, "payload": dict}]
        filters: {"field": str, "value": any} — normalized format
        defined in filters.py and translated per backend.
        """

    @abstractmethod
    async def delete_by_normalized_filter(
        self,
        collection_name: str,
        filters: dict | None,
    ) -> int:
        """
        Delete all points matching normalized metadata filter (``must`` list).
        Used for bulk operations (e.g. full re-sync by ``ac_source_id``).
        Returns count of deleted points.
        """

    async def delete_by_doc_id(self, collection_name: str, doc_id: str) -> int:
        """
        Deletes all points where payload.doc_id == doc_id.
        Returns count of deleted points.
        """
        return await self.delete_by_normalized_filter(
            collection_name,
            {"must": [{"field": "doc_id", "value": doc_id}]},
        )

    @abstractmethod
    async def get_by_doc_id(
        self,
        collection_name: str,
        doc_id: str,
    ) -> list[dict]:
        """
        Returns all points with matching doc_id in payload.
        Each item: {"id": str, "payload": dict}
        """

    @abstractmethod
    async def count(
        self,
        collection_name: str,
        filters: dict | None = None,
    ) -> int:
        """Returns total point count, optionally filtered."""

    @abstractmethod
    async def scroll(
        self,
        collection_name: str,
        filters: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Paginated full scan without vector similarity.
        Returns list of {"id": str, "payload": dict}
        Used for admin listing and health checks.
        """

    async def set_payload_by_filter(
        self,
        collection_name: str,
        filters: dict | None,
        payload: dict,
    ) -> int:
        """
        Merge ``payload`` keys into every point matching the normalized filter.
        Used by retention to flag docs archived/restored without re-embedding.
        Returns matched-point count (best effort). Optional per backend.
        """
        raise VectorDBError(
            "set_payload_by_filter not supported", self.backend_name
        )
