"""The agentic execution loop: run one task end-to-end.

Called by the ``/internal/agents/run`` route with the spec the backend assembled.
Reuses the container's QueryPipeline (kb_read + initial context) and KBWriter
(kb_write). Returns the final result, cost/tokens, the things written to the KB,
and a step-by-step run log — all persisted by the backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.agents.llm import call_llm
from app.agents.prompt import build_system_prompt
from app.agents.tools import Toolbox

logger = logging.getLogger(__name__)

MAX_STEPS = 16  # hard cap on agentic iterations (safety against tool-loop runaway)
INITIAL_KB_TOP_K = 12


def _format_kb_context(result: dict) -> str:
    chunks = result.get("chunks", []) if isinstance(result, dict) else []
    if not chunks:
        return "(No relevant knowledge base entries found.)"
    lines = []
    for i, c in enumerate(chunks, 1):
        title = c.get("doc_title") or c.get("source") or "entry"
        lines.append(f"[{i}] ({title}) {c.get('text', '')}")
    return "\n".join(lines)


async def run_agent_task(
    payload: dict,
    *,
    pipeline: Any,
    kb_writer: Any,
    settings: Any,
) -> dict:
    """Execute one agent task. Returns the result dict the backend persists."""
    task = payload.get("task", {})
    agent = payload.get("agent", {})
    context = payload.get("context", {})
    connectors_mcp_url = payload.get("connectors_mcp_url", "")
    instruction = task.get("instruction", "")
    budget_remaining = agent.get("budget_remaining_usd")

    toolbox = Toolbox(
        pipeline=pipeline,
        kb_writer=kb_writer,
        agent=agent,
        task=task,
        connectors_mcp_url=connectors_mcp_url,
    )
    await toolbox.load_connector_tools()

    system_prompt = build_system_prompt(agent, context, toolbox.connector_tool_names())

    # Seed the conversation with the instruction + a first KB retrieval.
    kb_result = await pipeline.query(instruction, top_k=INITIAL_KB_TOP_K)
    toolbox.record_step("kb_read", f"initial: {instruction[:200]}",
                        f"{len(kb_result.get('chunks', []))} chunks")
    seed = (
        f"Task:\n{instruction}\n\n"
        f"Relevant knowledge base context:\n{_format_kb_context(kb_result)}"
    )
    conv: list[dict] = [{"role": "user", "text": seed}]

    tool_specs = toolbox.specs()
    total_tokens = 0
    total_cost = 0.0
    final_text = ""
    status = "completed"

    for _ in range(MAX_STEPS):
        resp = await asyncio.to_thread(call_llm, agent, system_prompt, conv, tool_specs, settings)
        total_tokens += resp.tokens_used
        total_cost += resp.cost_usd
        final_text = resp.final_text or final_text
        toolbox.record_step(
            "llm_call",
            f"{len(conv)} msgs, {len(tool_specs)} tools",
            f"stop={resp.stop_reason} text={(resp.final_text or '')[:200]} "
            f"tool_calls={[t['name'] for t in resp.tool_calls]}",
        )

        # Mid-run budget stop (the backend pre-checks; this guards a single
        # expensive run from blowing past the remaining daily budget).
        if budget_remaining is not None and total_cost >= float(budget_remaining):
            status = "failed"
            final_text = (final_text + "\n\n").strip() + "\n[Stopped: daily budget limit reached]"
            break

        # Record the assistant turn (text + any tool requests).
        conv.append(
            {"role": "assistant", "text": resp.final_text, "tool_uses": resp.tool_calls}
        )

        if resp.stop_reason != "tool_use" or not resp.tool_calls:
            break

        # Execute each requested tool and feed results back.
        results = []
        for call in resp.tool_calls:
            out = await toolbox.execute(call["name"], call.get("input", {}))
            results.append({"id": call["id"], "name": call["name"], "content": out})
        conv.append({"role": "tool", "results": results})
    else:
        # Loop exhausted without an end_turn.
        logger.info("agent_run_max_steps task=%s", task.get("id"))

    return {
        "status": status,
        "result": final_text,
        "final_text": final_text,
        "tokens_used": total_tokens,
        "cost_usd": round(total_cost, 6),
        "kb_writes": toolbox.kb_writes,
        "runs": toolbox.steps,
    }
