"""İç API: yutma kuyruğu gözlemlenebilirliği (admin queue inspection)."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import require_master

router = APIRouter(
    prefix="/internal/ingestion",
    tags=["Internal — Ingestion"],
    dependencies=[Depends(require_master)],
)


def _queue(request: Request) -> Any:
    q = getattr(request.app.state, "ingestion_queue", None)
    if q is None:
        raise HTTPException(status_code=503, detail="Ingestion is disabled")
    return q


def _serialize_job(job: Any) -> dict[str, Any]:
    d = asdict(job)
    # Don't ship the full document text in admin listings — just its size.
    payload = d.pop("payload", {}) or {}
    d["doc_id"] = payload.get("doc_id", "")
    d["doc_title"] = payload.get("doc_title", "")
    d["chars"] = len(payload.get("text", "") or "")
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat() if v else None
    return d


@router.get("/stats")
async def ingestion_stats(request: Request) -> dict[str, Any]:
    return await asyncio.to_thread(_queue(request).stats)


@router.get("/jobs")
async def list_jobs(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    jobs = await asyncio.to_thread(_queue(request).list_jobs, status, limit)
    return {"count": len(jobs), "jobs": [_serialize_job(j) for j in jobs]}


@router.post("/jobs/{job_id}/retry")
async def retry_job(request: Request, job_id: str) -> dict[str, Any]:
    ok = await asyncio.to_thread(_queue(request).retry_dead, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No dead job with that id")
    return {"requeued": True, "job_id": job_id}
