"""System-prompt builder for an agent run.

Combines the agent's role, the context it lives in, its KB access, the tools it
may use, and a standing instruction to write findings back to the KB.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _kb_access_line(kb_scope: str, kb_section_ids: list[str]) -> str:
    scope = (kb_scope or "scoped").lower()
    if scope == "master":
        return "- The full context knowledge base (master scope)."
    if scope == "both":
        return "- The full context knowledge base, including all scoped sections."
    if kb_section_ids:
        return "- Scoped knowledge base sections: " + ", ".join(kb_section_ids) + "."
    return "- The scoped knowledge base for this context."


def build_system_prompt(agent: dict, context: dict, connector_tool_names: list[str]) -> str:
    """Return the system prompt string for this agent in this context.

    ``connector_tool_names`` are the human-facing connector tool names the agent
    is permitted to use (already filtered to the agent's connectors).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = agent.get("name") or "Agent"
    ctx_name = context.get("name") or "this"
    role = agent.get("role_description") or "Assist with tasks in this workspace."

    tool_lines = []
    for t in connector_tool_names:
        tool_lines.append(f"- {t}")
    tool_lines.append("- kb_read: search the knowledge base for relevant information")
    tool_lines.append(
        "- kb_write: save important findings, decisions, or learned information "
        "back to the knowledge base"
    )
    tools_block = "\n".join(tool_lines)

    return (
        f"You are {name}, working inside the {ctx_name} workspace.\n\n"
        f"Your role: {role}\n\n"
        f"You have access to the following knowledge:\n"
        f"{_kb_access_line(agent.get('kb_scope', 'scoped'), agent.get('kb_section_ids', []) or [])}\n\n"
        f"You can use these tools:\n"
        f"{tools_block}\n\n"
        f"Always write back anything important you learn or decide using kb_write. "
        f"Use scope=\"private\" for working notes scoped to you, and scope=\"shared\" "
        f"for findings worth promoting to the shared knowledge base (these are queued "
        f"for human review, not written immediately).\n\n"
        f"Work step by step. When the task is complete, give a clear final answer. "
        f"Today is {today}."
    )
