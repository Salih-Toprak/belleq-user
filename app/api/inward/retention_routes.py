"""İç API: retention — stale-doc archive/purge status, preview, restore.

Reached only by the master (X-Master-Key). Thresholds are changed via the
regular PATCH /internal/config route (they're RUNTIME_CONFIG_KEYS); these
routes cover everything else the dashboard Settings page needs:

    GET  /internal/retention/status    config + last sweep + counts
    GET  /internal/retention/archived  archived docs (restorable)
    POST /internal/retention/sweep     run now ({"dry_run": true} = preview)
    POST /internal/retention/restore   {"doc_id": ...} un-archive
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import app.config as app_config
from app.api.deps import require_master

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/retention",
    tags=["Internal — Retention"],
    dependencies=[Depends(require_master)],
)


def _sweeper(request: Request) -> Any:
    sweeper = getattr(request.app.state, "retention_sweeper", None)
    if sweeper is None:
        raise HTTPException(status_code=503, detail="Retention is unavailable (no vector db)")
    return sweeper


class SweepBody(BaseModel):
    dry_run: bool = False


class RestoreBody(BaseModel):
    doc_id: str


@router.get("/status")
async def retention_status(request: Request) -> dict[str, Any]:
    s = app_config.settings
    sweeper = getattr(request.app.state, "retention_sweeper", None)
    tracker = getattr(request.app.state, "activity_tracker", None)
    archived_count = 0
    if sweeper is not None:
        try:
            archived_count = len(await sweeper.list_archived())
        except Exception:  # noqa: BLE001 — status must not 500 on a cold collection
            logger.debug("retention_status_archived_count_failed", exc_info=True)
    return {
        "retention_enabled": s.retention_enabled,
        "retention_archive_after_days": s.retention_archive_after_days,
        "retention_purge_enabled": s.retention_purge_enabled,
        "retention_purge_after_days": s.retention_purge_after_days,
        "sweep_interval_hours": s.retention_sweep_interval_hours,
        "active_days_total": tracker.total_active_days() if tracker else 0,
        "archived_count": archived_count,
        "last_sweep": sweeper.last_sweep() if sweeper else None,
    }


@router.get("/archived")
async def retention_archived(request: Request) -> dict[str, Any]:
    sweeper = _sweeper(request)
    try:
        docs = await sweeper.list_archived()
    except Exception as exc:  # noqa: BLE001 — empty/cold collection = no archive
        logger.debug("retention_archived_list_failed", exc_info=True)
        if "not found" in str(exc).lower():
            docs = []
        else:
            raise HTTPException(status_code=502, detail=str(exc)[:200])
    return {"archived": docs, "count": len(docs)}


@router.post("/sweep")
async def retention_sweep(body: SweepBody, request: Request) -> dict[str, Any]:
    sweeper = _sweeper(request)
    try:
        return await sweeper.sweep(dry_run=body.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retention_manual_sweep_failed", exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc)[:200])


@router.post("/restore")
async def retention_restore(body: RestoreBody, request: Request) -> dict[str, Any]:
    sweeper = _sweeper(request)
    doc_id = (body.doc_id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=422, detail="doc_id is required")
    try:
        n = await sweeper.restore(doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retention_restore_failed doc=%s", doc_id, exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc)[:200])
    return {"restored": doc_id, "chunks": n}
