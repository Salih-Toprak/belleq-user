"""İç API: konuşma arşivi gözlemlenebilirliği (dashboard konuşma günlüğü)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import get_conversation_store, require_master

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/conversations",
    tags=["Internal — Conversations"],
    dependencies=[Depends(require_master)],
)


def _serialize(obj: Any) -> dict[str, Any]:
    d = asdict(obj)
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat() if v else None
    return d


def _require_store(request: Request) -> Any:
    store = get_conversation_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Conversation capture is disabled")
    return store


@router.get("/stats")
async def conversation_stats(request: Request) -> dict[str, Any]:
    store = _require_store(request)
    return await asyncio.to_thread(store.stats)


@router.get("")
async def list_conversations(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=5000),
) -> dict[str, Any]:
    store = _require_store(request)
    sessions = await asyncio.to_thread(store.list_sessions, status, limit)
    return {
        "count": len(sessions),
        "sessions": [_serialize(s) for s in sessions],
    }


@router.get("/{session_id}")
async def get_conversation(request: Request, session_id: str) -> dict[str, Any]:
    store = _require_store(request)
    session = await asyncio.to_thread(store.get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    turns = await asyncio.to_thread(store.get_session_turns, session_id)
    return {
        "session": _serialize(session),
        "turns": [_serialize(t) for t in turns],
    }
