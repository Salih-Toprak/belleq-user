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


class UploadBody(BaseModel):
    filename: str
    content_base64: str = ""
    text: str = ""
    title: str = ""
    replace: bool = False


class CaptureBody(BaseModel):
    text: str
    title: str = ""
    source_label: str = "connector"
    tool: str = ""


class AgentWriteBody(BaseModel):
    content: str
    tags: list[str] = Field(default_factory=list)
    scope: str = "shared"
    source: str = "agent"


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


def _touch_activity(request: Request) -> None:
    """Best-effort: writes also mark today an active day for retention."""
    tracker = getattr(request.app.state, "activity_tracker", None)
    if tracker is not None:
        try:
            tracker.mark_active()
        except Exception:  # noqa: BLE001
            pass


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
    await asyncio.to_thread(_touch_activity, request)
    return ack if isinstance(ack, dict) else {"recorded": True}


@router.post("/flush")
async def kb_flush(request: Request) -> dict[str, Any]:
    """Ingest buffered exchanges into the KB now (mirror of flush_knowledge_base)."""
    sm = getattr(request.app.state, "session_manager", None)
    if sm is None:
        raise HTTPException(status_code=503, detail="Conversation extraction is disabled")
    return await asyncio.to_thread(sm.flush_now)


@router.post("/upload")
async def kb_upload(body: UploadBody, request: Request) -> dict[str, Any]:
    """Add a document to the KB (mirror of the upload_document MCP tool).

    Accepts either base64 file bytes or plain text; extracts, dedups, and queues
    it for chunk → embed → index.
    """
    import base64

    from app.ingestion.extractors import ExtractionError
    from app.ingestion.service import enqueue_document

    queue = getattr(request.app.state, "ingestion_queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="Ingestion is disabled")

    if body.content_base64.strip():
        try:
            raw = base64.b64decode(body.content_base64, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}")
    elif body.text.strip():
        raw = body.text.encode("utf-8")
    else:
        raise HTTPException(status_code=422, detail="Provide either text or content_base64")

    filename = body.filename or "upload.txt"
    import app.config as app_config

    max_mb = int(getattr(app_config.settings, "ingestion_max_upload_mb", 25) or 25)
    if len(raw) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds the {max_mb} MB limit")

    try:
        result = await asyncio.to_thread(
            enqueue_document,
            queue,
            raw=raw,
            filename=filename,
            title=body.title,
            replace=body.replace,
        )
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await asyncio.to_thread(_touch_activity, request)
    return result


@router.post("/agent_write")
async def kb_agent_write(body: AgentWriteBody, request: Request) -> dict[str, Any]:
    """Upsert agent-authored content into the KB (used when the backend approves
    a queued ``scope="shared"`` review item). Reuses the shared KBWriter so it
    behaves identically to the in-process agent kb_write."""
    kb_writer = getattr(request.app.state, "kb_writer", None)
    if kb_writer is None or not kb_writer.available():
        raise HTTPException(status_code=503, detail="KB writer is unavailable")
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="content is required")
    doc_key = abs(hash(content)) % 10_000_000
    written = await asyncio.to_thread(
        kb_writer.write_chunks,
        doc_id=f"agent-shared-{doc_key}",
        doc_title="Agent shared note",
        doc_path=f"agent-shared:{doc_key}",
        source=body.source or "agent",
        chunks=[content],
        extra_payload={"tags": body.tags, "scope": body.scope},
    )
    return {"written": written}


@router.post("/capture")
async def kb_capture(body: CaptureBody, request: Request) -> dict[str, Any]:
    """Queue a document-like MCP tool response (4C MCP response capture).

    Called by the master's proxy middleware when a connector returns a chunk of
    document-like text. Dedup + chunk + embed happen via the same queue/worker.
    """
    from app.ingestion.service import enqueue_capture

    queue = getattr(request.app.state, "ingestion_queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="Ingestion is disabled")
    try:
        return await asyncio.to_thread(
            enqueue_capture,
            queue,
            text=body.text,
            title=body.title,
            source_label=body.source_label,
            extra={"tool": body.tool} if body.tool else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
