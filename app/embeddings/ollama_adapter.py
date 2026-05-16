# Shared adapter — keep in sync with mnemo-master/app/embeddings/
"""Ollama /api/embed embedding backend."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.embeddings.base import EmbeddingAdapter, EmbeddingError

logger = logging.getLogger(__name__)


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """Local Ollama embedding API."""

    backend_name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        vector_size: int,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._vector_size = int(vector_size)
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        logger.info(
            "ollama_embedding_init base_url=%s model=%s vector_size=%s",
            self._base_url,
            model,
            vector_size,
        )

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def model_name(self) -> str:
        return self._model

    def _parse_embeddings(self, data: dict[str, Any], n_expected: int) -> list[list[float]]:
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs and isinstance(embs[0], list):
            return [list(map(float, e)) for e in embs]
        single = data.get("embedding")
        if isinstance(single, list) and n_expected <= 1:
            return [list(map(float, single))]
        raise EmbeddingError("unexpected Ollama embed response shape", self.backend_name)

    async def embed_one(self, text: str) -> list[float]:
        vecs = await self.embed_batch([text])
        return vecs[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/api/embed"
        body: dict[str, Any] = {"model": self._model, "input": texts}
        try:
            r = await self._client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            out = self._parse_embeddings(data, len(texts))
            if len(out) != len(texts):
                raise EmbeddingError(
                    f"embedding count mismatch: got {len(out)} expected {len(texts)}",
                    self.backend_name,
                )
            return out
        except EmbeddingError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("ollama_embed_batch_failed error=%s", e)
            out_seq: list[list[float]] = []
            for t in texts:
                try:
                    one = await self._embed_one_raw(t)
                    out_seq.append(one)
                except Exception as e2:  # noqa: BLE001
                    raise EmbeddingError(str(e2), self.backend_name, detail=str(e2)) from e2
            return out_seq

    async def _embed_one_raw(self, text: str) -> list[float]:
        url = f"{self._base_url}/api/embed"
        r = await self._client.post(url, json={"model": self._model, "input": text})
        r.raise_for_status()
        data = r.json()
        vecs = self._parse_embeddings(data, 1)
        return vecs[0]

    async def health(self) -> dict:
        try:
            v = await self.embed_one("health check")
            if len(v) != self._vector_size:
                return {
                    "status": "error",
                    "backend": self.backend_name,
                    "model": self._model,
                    "vector_size": self._vector_size,
                    "detail": f"vector length mismatch: got {len(v)} expected {self._vector_size}",
                }
            return {
                "status": "ok",
                "backend": self.backend_name,
                "model": self._model,
                "vector_size": self._vector_size,
                "detail": self._base_url,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "status": "error",
                "backend": self.backend_name,
                "model": self._model,
                "vector_size": self._vector_size,
                "detail": str(e),
            }

    async def aclose(self) -> None:
        await self._client.aclose()

    def embed_query(self, text: str) -> list[float]:
        """
        Sync wrapper required by rag-wiki / LangChain embedding interface.

        Uses a short-lived httpx.Client (synchronous) instead of asyncio.run()
        to avoid creating and destroying event loops, which would corrupt the
        shared async httpx client used by embed_batch() and the Qdrant adapter.
        """
        url = f"{self._base_url}/api/embed"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(url, json={"model": self._model, "input": [text]})
                r.raise_for_status()
                vecs = self._parse_embeddings(r.json(), 1)
                return vecs[0]
        except EmbeddingError:
            raise
        except Exception as e:  # noqa: BLE001
            raise EmbeddingError(str(e), self.backend_name) from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Sync wrapper required by rag-wiki / LangChain embedding interface.

        Uses a short-lived httpx.Client (synchronous) instead of asyncio.run()
        to avoid event loop lifecycle issues (see embed_query docstring).
        """
        if not texts:
            return []
        url = f"{self._base_url}/api/embed"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(url, json={"model": self._model, "input": texts})
                r.raise_for_status()
                return self._parse_embeddings(r.json(), len(texts))
        except EmbeddingError:
            raise
        except Exception as e:  # noqa: BLE001
            raise EmbeddingError(str(e), self.backend_name) from e
