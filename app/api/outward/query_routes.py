"""Dış API: kullanıcı sorguları."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

import app.config as app_config
from app import state as app_state
from app.api.deps import get_pipeline, require_user
from app.query.pipeline import QueryPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query — User Facing"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        t = (v or "").strip()
        if not t:
            raise ValueError("Query must not be empty")
        return t


@router.get("/health")
async def query_health() -> dict[str, str]:
    """Returns retrieval subsystem health."""
    s = app_config.settings
    return {
        "status": "ok",
        "user_id": s.user_id,
        "vectordb_backend": s.vectordb_backend,
        "embedding_backend": s.embedding_backend,
    }


@router.post("", dependencies=[Depends(require_user)])
async def post_query(
    body: QueryRequest,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """
    Returns relevant document chunks from the rag-wiki lifecycle retriever.
    No answer generation — the caller is responsible for using the chunks.
    """
    if app_state.get_vectordb() is None:
        raise HTTPException(
            status_code=503,
            detail="Vector database is not available. Check VECTORDB_* settings and connectivity.",
        )
    if app_state.get_embedder() is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not available. Check EMBEDDING_* settings and connectivity.",
        )
    try:
        result = await pipeline.query(
            body.query.strip(),
            top_k=body.top_k,
        )
    except Exception as e:
        msg = str(e).lower()
        if "vector" in msg or "qdrant" in msg or "pinecone" in msg:
            raise HTTPException(
                status_code=503,
                detail="Vector database query failed.",
            ) from e
        raise
    return result
