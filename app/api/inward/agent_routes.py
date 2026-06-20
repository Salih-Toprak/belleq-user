"""İç API: agent task yürütme (master üzerinden backend tarafından çağrılır).

The backend assembles the run spec (agent config, decrypted BYOK key, task
instruction, scoped KB info, connectors_mcp_url) and POSTs it here through the
master. We run the full agentic loop in-process — reusing this container's
QueryPipeline + KBWriter — and return the result, cost, KB writes, and a
step-by-step run log for the backend to persist.

Auth is the stable ``X-Master-Key`` (same as the other ``/internal/*`` routes).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

import app.config as app_config
from app.agents.runner import run_agent_task
from app.api.deps import require_master

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/agents",
    tags=["Internal — Agent execution"],
    dependencies=[Depends(require_master)],
)


@router.post("/run")
async def run_agent(request: Request, payload: dict = Body(default_factory=dict)) -> dict[str, Any]:
    pipeline = getattr(request.app.state, "pipeline", None)
    kb_writer = getattr(request.app.state, "kb_writer", None)
    if pipeline is None or kb_writer is None:
        raise HTTPException(status_code=503, detail="Agent runtime is unavailable")

    task_id = (payload.get("task") or {}).get("id", "?")
    agent_id = (payload.get("agent") or {}).get("id", "?")
    logger.info("agent_run_started task=%s agent=%s", task_id, agent_id)
    result = await run_agent_task(
        payload,
        pipeline=pipeline,
        kb_writer=kb_writer,
        settings=app_config.settings,
    )
    logger.info(
        "agent_run_finished task=%s status=%s cost=%.4f steps=%d",
        task_id, result.get("status"), result.get("cost_usd", 0), len(result.get("runs", [])),
    )
    return result
