"""İç API: KB işlemlerinin REST karşılığı (MCP olmayan sağlayıcılar için).

These mirror the four MCP tools (recall / query / record / flush) as plain JSON
REST endpoints so AI providers that don't speak MCP — ChatGPT (Actions /
function calling), Gemini (function calling), or any HTTP client — can use the
same Belleq memory loop. They call exactly the same pipeline/capture/session
methods the MCP server wraps, so behaviour is identical across transports.

Reached only via the master on the private docker network; auth is the stable
``X-Master-Key`` (same as the other ``/internal/*`` routes), NOT the per-context
api_key. Public auth (the regeneratable api_key) is enforced upstream at the
backend bridge, so rotating the api_key never requires recreating this container.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_pipeline, require_master
from app.query.pipeline import QueryPipeline

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/kb",
    tags=["Internal — Knowledge Base REST"],
    dependencies=[Depends(require_master)],
)


class RecallBody(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)


class QueryBody(BaseModel):
    query: str = Field(min_length=1)


class RecordBody(BaseModel):
    user_message: str
    assistant_message: str
    conversation_id: str = ""


@router.post("/recall")
async def kb_recall(
    body: RecallBody,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Load recently saved conversation facts (mirror of recall_context)."""
    return await pipeline.recent_context(body.limit)


@router.post("/query")
async def kb_query(
    body: QueryBody,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    """Search the knowledge base (mirror of query_knowledge_base)."""
    return await pipeline.query(body.query)


@router.post("/record")
async def kb_record(body: RecordBody, request: Request) -> dict[str, Any]:
    """Save a verbatim exchange (mirror of record_exchange)."""
    capture = getattr(request.app.state, "conversation_capture", None)
    if capture is None:
        raise HTTPException(status_code=503, detail="Conversation capture is disabled")
    ack = await asyncio.to_thread(
        capture.record_exchange,
        body.user_message,
        body.assistant_message,
        body.conversation_id or None,
    )
    return ack if isinstance(ack, dict) else {"recorded": True}


@router.post("/flush")
async def kb_flush(request: Request) -> dict[str, Any]:
    """Ingest buffered exchanges into the KB now (mirror of flush_knowledge_base)."""
    sm = getattr(request.app.state, "session_manager", None)
    if sm is None:
        raise HTTPException(status_code=503, detail="Conversation extraction is disabled")
    return await asyncio.to_thread(sm.flush_now)
