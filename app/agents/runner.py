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

# Hard cap on agentic iterations — only a runaway-loop safety net, NOT a feature
# limit. Set high so big tasks (e.g. reading dozens of files) finish; cost is
# bounded separately by the agent's daily budget. Override via AGENT_MAX_STEPS.
DEFAULT_MAX_STEPS = 100
INITIAL_KB_TOP_K = 12

# Connector name fragments that identify a "send a message to a human" tool, used
# to deliver run notifications through a Slack/Discord/etc. connector.
_MESSAGING_HINTS = ("slack", "discord", "telegram", "teams", "mattermost", "message", "send", "post", "chat")


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
        tavily_api_key=getattr(settings, "tavily_api_key", "") or "",
        step_callback=payload.get("step_callback"),
    )
    await toolbox.load_connector_tools()

    system_prompt = build_system_prompt(
        agent, context, toolbox.connector_tool_names(), web_tools=toolbox.web_tool_names()
    )

    # Seed the conversation with the instruction + a first KB retrieval.
    kb_result = await pipeline.query(instruction, top_k=INITIAL_KB_TOP_K)
    toolbox.record_step("kb_read", f"initial: {instruction[:200]}",
                        f"{len(kb_result.get('chunks', []))} chunks")
    await toolbox.flush_steps()
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
    max_steps = int(getattr(settings, "agent_max_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS)

    for _ in range(max_steps):
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
        await toolbox.flush_steps()

        # User pressed Stop (signaled back via the step callback).
        if toolbox.cancel_requested():
            status = "cancelled"
            final_text = (final_text + "\n\n").strip() + "\n[Stopped by user]"
            break

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
        await toolbox.flush_steps()
        if toolbox.cancel_requested():
            status = "cancelled"
            final_text = (final_text + "\n\n").strip() + "\n[Stopped by user]"
            break
    else:
        # Loop exhausted without the agent finishing — make it explicit instead of
        # returning a mid-thought message that looks like a silent stop.
        logger.info("agent_run_max_steps task=%s steps=%s", task.get("id"), max_steps)
        final_text = (final_text + "\n\n").strip() + (
            f"\n[Reached the {max_steps}-step limit before finishing. "
            f"Raise AGENT_MAX_STEPS or split the task into smaller runs.]"
        )

    return {
        "status": status,
        "result": final_text,
        "final_text": final_text,
        "tokens_used": total_tokens,
        "cost_usd": round(total_cost, 6),
        "kb_writes": toolbox.kb_writes,
        "runs": toolbox.steps,
    }


async def send_notification(
    payload: dict,
    *,
    pipeline: Any,
    kb_writer: Any,
    settings: Any,
) -> dict:
    """Deliver a run-completion message through the agent's messaging connector.

    Runs a single, tightly-scoped LLM turn whose only tools are the connector's
    own tools, so it can call the right "send message" tool with the right args
    (channel, etc.) regardless of which Slack/Discord/… connector is attached.
    Best-effort: returns {sent: bool} and never raises into the caller.
    """
    agent = payload.get("agent", {})
    message = (payload.get("message") or "").strip()
    connectors_mcp_url = payload.get("connectors_mcp_url", "")
    if not message or not agent.get("connector_ids"):
        return {"sent": False, "reason": "no message or no messaging connector"}

    toolbox = Toolbox(
        pipeline=pipeline,
        kb_writer=kb_writer,
        agent=agent,
        task={},
        connectors_mcp_url=connectors_mcp_url,
    )
    await toolbox.load_connector_tools()
    # Only offer tools that look like "send a message to a human".
    specs = [s for s in toolbox.connector_specs() if any(h in s["name"].lower() for h in _MESSAGING_HINTS)]
    if not specs:
        return {"sent": False, "reason": "no messaging tool found on attached connectors"}

    system = (
        "You deliver a notification. Use one of the available messaging tools to "
        "send the message below to the user, exactly as written, then stop. Do not "
        "rephrase it, do not add commentary, and do not call any other tool. If a "
        "channel or recipient is required, use the connector's default or most "
        "general one."
    )
    conv = [{"role": "user", "text": f"Send this message now:\n\n{message}"}]
    try:
        resp = await asyncio.to_thread(call_llm, agent, system, conv, specs, settings)
        sent = False
        for call in resp.tool_calls or []:
            await toolbox.execute(call["name"], call.get("input", {}))
            sent = True
        return {"sent": sent}
    except Exception as exc:  # noqa: BLE001 — notifications are best-effort
        logger.warning("agent_notify_send_failed err=%s", exc, exc_info=True)
        return {"sent": False, "reason": str(exc)}
