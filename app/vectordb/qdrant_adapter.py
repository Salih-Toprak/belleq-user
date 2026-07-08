# Shared adapter — keep in sync with belleq-master/app/vectordb/
"""Qdrant implementation of VectorDBAdapter (async client)."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.vectordb import filters as filter_utils
from app.vectordb.base import VectorDBAdapter, VectorDBError

logger = logging.getLogger(__name__)


def _distance_to_qdrant(distance: str) -> qmodels.Distance:
    d = (distance or "Cosine").strip().lower()
    if d == "cosine":
        return qmodels.Distance.COSINE
    if d in ("euclidean", "l2"):
        return qmodels.Distance.EUCLID
    if d in ("dot", "dotproduct"):
        return qmodels.Distance.DOT
    return qmodels.Distance.COSINE


def _qdrant_distance_to_str(dist: Any) -> str:
    if dist == qmodels.Distance.COSINE:
        return "Cosine"
    if dist == qmodels.Distance.EUCLID:
        return "Euclidean"
    if dist == qmodels.Distance.DOT:
        return "Dot"
    return str(dist)


class QdrantAdapter(VectorDBAdapter):
    """Qdrant Cloud or local cluster via AsyncQdrantClient."""

    backend_name = "qdrant"

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._client = self._make_client()
        logger.info("qdrant_adapter_init url=%s has_api_key=%s", url, bool(api_key))

    def _make_client(self) -> AsyncQdrantClient:
        kwargs: dict[str, Any] = {"url": self._url, "timeout": self._timeout}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return AsyncQdrantClient(**kwargs)

    async def _reconnect(self) -> None:
        """Recreate the async client after a stale event loop error."""
        try:
            await self._client.close()
        except Exception:
            pass
        self._client = self._make_client()
        logger.info("qdrant_adapter_reconnected url=%s", self._url)

    async def health(self) -> dict:
        try:
            await self._client.get_collections()
            return {"status": "ok", "backend": self.backend_name, "detail": self._url}
        except Exception as e:  # noqa: BLE001
            logger.warning("qdrant_health_failed error=%s", e)
            return {
                "status": "error",
                "backend": self.backend_name,
                "detail": f"{self._url}: {e!s}",
            }

    async def list_collections(self) -> list[str]:
        try:
            res = await self._client.get_collections()
            return [c.name for c in res.collections]
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def get_collection_info(self, collection_name: str) -> dict:
        try:
            info = await self._client.get_collection(collection_name=collection_name)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "not found" in msg or "404" in msg:
                raise VectorDBError(
                    f"collection not found: {collection_name}",
                    self.backend_name,
                    detail=str(e),
                ) from e
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e
        params = info.config.params
        vs = 0
        dist = "Cosine"
        if params and params.vectors:
            v = next(iter(params.vectors.values()))
            if hasattr(v, "size"):
                vs = int(v.size or 0)
            if hasattr(v, "distance"):
                dist = _qdrant_distance_to_str(v.distance)
        counts = info.points_count or 0
        indexed = getattr(info, "indexed_vectors_count", None) or counts
        status = getattr(info.status, "value", None) or str(info.status or "unknown")
        return {
            "name": collection_name,
            "vector_count": int(counts),
            "indexed_vector_count": int(indexed),
            "status": str(status),
            "vector_size": int(vs),
            "distance": dist,
        }

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        try:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=_distance_to_qdrant(distance),
                ),
            )
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def delete_collection(self, collection_name: str) -> None:
        try:
            await self._client.delete_collection(collection_name=collection_name)
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def upsert(self, collection_name: str, points: list[dict]) -> int:
        try:
            batch = []
            for p in points:
                pid = p.get("id")
                vec = p.get("vector") or []
                payload = p.get("payload") or {}
                batch.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))
            await self._client.upsert(collection_name=collection_name, points=batch)
            return len(batch)
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        for attempt in range(2):
            try:
                flt = filter_utils.build_qdrant_filter(filters)
                res = await self._client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=flt,
                    with_payload=True,
                )
                out: list[dict] = []
                for hit in res:
                    out.append(
                        {
                            "id": str(hit.id),
                            "score": float(hit.score),
                            "payload": hit.payload or {},
                        }
                    )
                return out
            except Exception as e:  # noqa: BLE001
                if attempt == 0 and "event loop is closed" in str(e).lower():
                    logger.warning(
                        "qdrant_search_stale_loop attempt=%d — reconnecting", attempt
                    )
                    await self._reconnect()
                    continue
                raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e
        raise VectorDBError("search failed after reconnect", self.backend_name)

    async def delete_by_normalized_filter(
        self,
        collection_name: str,
        filters: dict | None,
    ) -> int:
        flt = filter_utils.build_qdrant_filter(filters)
        total_deleted = 0
        try:
            while True:
                points, next_offset = await self._client.scroll(
                    collection_name=collection_name,
                    scroll_filter=flt,
                    limit=256,
                    with_payload=False,
                    with_vectors=False,
                )
                if not points:
                    break
                ids = [pt.id for pt in points]
                for i in range(0, len(ids), 100):
                    chunk = ids[i : i + 100]
                    await self._client.delete(
                        collection_name=collection_name,
                        points_selector=qmodels.PointIdsList(points=chunk),
                    )
                    total_deleted += len(chunk)
                if next_offset is None:
                    break
            return total_deleted
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def get_by_doc_id(self, collection_name: str, doc_id: str) -> list[dict]:
        flt = filter_utils.build_qdrant_filter(
            {"must": [{"field": "doc_id", "value": doc_id}]}
        )
        try:
            points, _ = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )
            return [{"id": str(pt.id), "payload": pt.payload or {}} for pt in points]
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def count(self, collection_name: str, filters: dict | None = None) -> int:
        flt = filter_utils.build_qdrant_filter(filters)
        try:
            res = await self._client.count(
                collection_name=collection_name,
                count_filter=flt,
                exact=True,
            )
            return int(res.count)
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def set_payload_by_filter(
        self,
        collection_name: str,
        filters: dict | None,
        payload: dict,
    ) -> int:
        """Merge payload keys into all points matching the normalized filter."""
        flt = filter_utils.build_qdrant_filter(filters)
        try:
            matched = await self.count(collection_name, filters)
            await self._client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=qmodels.FilterSelector(filter=flt),
            )
            return matched
        except VectorDBError:
            raise
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e

    async def scroll(
        self,
        collection_name: str,
        filters: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Paginate using Qdrant cursor scroll; ``offset`` skips that many points in order."""
        flt = filter_utils.build_qdrant_filter(filters)
        try:
            skip_first = max(0, int(offset))
            page_limit = max(1, int(limit))
            collected: list[dict] = []
            stream_index = 0
            next_page: Any = None
            while len(collected) < page_limit:
                points, next_page = await self._client.scroll(
                    collection_name=collection_name,
                    scroll_filter=flt,
                    limit=256,
                    offset=next_page,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break
                for pt in points:
                    if stream_index < skip_first:
                        stream_index += 1
                        continue
                    collected.append({"id": str(pt.id), "payload": pt.payload or {}})
                    stream_index += 1
                    if len(collected) >= page_limit:
                        return collected
                if next_page is None:
                    break
            return collected
        except Exception as e:  # noqa: BLE001
            raise VectorDBError(str(e), self.backend_name, detail=str(e)) from e
