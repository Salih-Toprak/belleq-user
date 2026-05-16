"""İç API: sağlık ve istatistik."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Request

import app.config as app_config
from app.api.deps import get_global_store, get_state_store, require_master
from app.lifecycle.lifecycle_stats import lifecycle_aggregates
from app import state as app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["Internal — Health & Stats"])


async def _db_ping(db_path: str) -> bool:
    def _run() -> None:
        con = sqlite3.connect(db_path, timeout=2.0)
        try:
            con.execute("SELECT 1")
        finally:
            con.close()

    try:
        await asyncio.wait_for(asyncio.to_thread(_run), timeout=2.5)
        return True
    except Exception:
        logger.debug("internal_health_db_ping_failed", exc_info=True)
        return False


@router.get("/health", dependencies=[Depends(require_master)])
async def internal_health() -> dict[str, Any]:
    s = app_config.settings
    await _db_ping(s.db_path)
    up = app_state.get_uptime()
    return {
        "status": "ok",
        "user_id": s.user_id,
        "display_name": s.display_name,
        "container_type": s.container_type,
        "vectordb_backend": s.vectordb_backend,
        "embedding_backend": s.embedding_backend,
        "rag_wiki_fetch_threshold": s.rag_wiki_fetch_threshold,
        "data_dir": s.data_dir,
        "uptime_seconds": up,
    }


@router.get("/stats", dependencies=[Depends(require_master)])
async def internal_stats(request: Request) -> dict[str, Any]:
    s = app_config.settings
    up = app_state.get_uptime()
    base: dict[str, Any] = {
        "user_id": s.user_id,
        "display_name": s.display_name,
        "container_type": s.container_type,
        "reachable": True,
        "documents": {},
        "lifecycle": {},
        "system": {
            "vectordb_backend": s.vectordb_backend,
            "embedding_backend": s.embedding_backend,
            "db_path": s.db_path,
            "uptime_seconds": up,
        },
    }
    gs = get_global_store(request)
    st = get_state_store(request)
    try:

        def _doc_stats() -> dict:
            stats = gs.get_stats()
            total = int(stats.get("total_documents", 0))
            flagged = int(stats.get("flagged_count", 0))
            avg = stats.get("average_decay_score")
            return {
                "total": total,
                "flagged": flagged,
                "healthy": max(0, total - flagged),
                "needs_review": flagged,
                "average_decay_score": float(avg) if avg is not None else 0.0,
                "by_department": dict(stats.get("by_department") or {}),
                "by_source": dict(stats.get("by_source") or {}),
                "total_chunks": int(stats.get("total_chunks", 0)),
            }

        doc_block = await asyncio.to_thread(_doc_stats)
        base["documents"] = doc_block
        life = await asyncio.to_thread(lifecycle_aggregates, st, s.user_id)
        base["lifecycle"] = life
    except Exception as e:
        logger.warning("internal_stats_store_error", exc_info=True)
        base["documents"] = {
            "total": 0,
            "flagged": 0,
            "healthy": 0,
            "needs_review": 0,
            "average_decay_score": 0.0,
            "by_department": {},
            "by_source": {},
            "total_chunks": 0,
        }
        base["lifecycle"] = {
            "claimed_docs": 0,
            "pinned_docs": 0,
            "surfaced_docs": 0,
            "demoted_docs": 0,
            "total_fetch_count": 0,
        }
        base["store_error"] = str(e)
    return base
