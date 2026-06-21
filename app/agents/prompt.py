"""System-prompt builder for an agent run.

Combines the agent's role, the context it lives in, its KB access, the tools it
may use, and a knowledge operating procedure modelled on the "LLM Wiki" pattern
(Karpathy): the agent treats the knowledge base as a living, self-organized wiki
it maintains itself — recalling first, citing sources, writing atomic tagged and
cross-linked notes, and compounding knowledge over time rather than re-deriving
it on every run.
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


# The knowledge operating procedure — the heart of how the agent behaves. Adapted
# from the LLM Wiki pattern to belleq's vector KB: notes are atomic chunks tagged
# and cross-linked by title so they form a self-organizing wiki the agent both
# reads from and grows.
_KNOWLEDGE_PROCEDURE = """\
How you work with knowledge (follow this every task):

1. RECALL FIRST. Before reasoning or answering, search the knowledge base with
   kb_read for anything relevant to the task. Treat what you find as
   authoritative over your own prior assumptions. The task message already
   includes an initial KB search; run more targeted kb_read calls as needed.

2. GATHER. If the KB doesn't already answer it, get the missing information from a
   tool — web_search / web_fetch for anything about the outside world (companies,
   people, prices, news, social handles, market facts), and your connector tools
   for systems you're connected to — then come back to the KB. Read real sources;
   do not answer external questions from memory.

3. RECORD AS YOU LEARN. Whenever you learn or decide something durable, save it
   with kb_write. Do not wait until the end. Write atomic, self-contained notes —
   one idea per note — so they stay reusable. Each note should be structured:
     • a short Title: line naming the one thing the note is about
     • the fact / finding / decision, stated plainly
     • Tags: 2-5 lowercase keyword tags (pass them in the `tags` field)
     • Related: [[Other Note Title]] wikilinks to notes this connects to
     • Source: where it came from (a connector/tool name, a URL, or "reasoning")
   Re-use existing note titles when linking so the wiki stays connected.

4. DON'T DUPLICATE. Before writing, kb_read for an existing note on the same
   topic. If one exists, write an updated/expanded note that supersedes it
   (same Title:) rather than a near-duplicate.

5. BUILD IT UP. If the knowledge base is empty or thin, that is expected early on
   — create the foundational notes yourself (key entities, concepts, decisions)
   and tag/link them, so every future task starts from more context than the
   last. Knowledge compounds; nothing you learn should have to be re-derived.

6. SCOPE. Use scope="private" for your own working notes; use scope="shared" for
   durable knowledge worth promoting to the shared knowledge base (shared writes
   are queued for human review before they are indexed).

Always cite sources in your notes, and re-check sources before overturning an
earlier claim. When the task is done, give a clear final answer grounded in what
the KB now contains.\
"""


# Hard rule against fabrication. The agent previously invented competitor handles,
# follower counts, and "posts that performed well" from memory — this forbids that.
_GROUNDING = """\
Grounding (this overrides everything above when they conflict):

• NEVER invent facts, numbers, names, dates, quotes, URLs, social handles, or
  metrics. If you state something about the real world, it must come from a tool
  result in THIS run (web_search, web_fetch, or a connector) — not from memory or
  plausible-sounding guesses.
• For external facts, the Source: line of the note must be the real URL or tool
  the fact came from. Never write Source: reasoning for a factual claim about the
  outside world.
• If you cannot verify something with a tool, say so plainly and write a note
  tagged "needs_human" describing exactly what's missing — do not fill the gap
  with a guess. A flagged gap is correct; a confident fabrication is a failure.
• Distinguish what you verified from what you inferred. Mark inferences as such.\
"""


def build_system_prompt(
    agent: dict,
    context: dict,
    connector_tool_names: list[str],
    web_tools: list[str] | None = None,
) -> str:
    """Return the system prompt string for this agent in this context.

    ``connector_tool_names`` are the human-facing connector tool names the agent
    is permitted to use (already filtered to the agent's connectors). ``web_tools``
    are the built-in web tool names available when a platform search key is set.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = agent.get("name") or "Agent"
    ctx_name = context.get("name") or "this"
    role = agent.get("role_description") or "Assist with tasks in this workspace."

    tool_lines: list[str] = []
    tool_lines.append("- kb_read: search the knowledge base for relevant information")
    tool_lines.append(
        "- kb_write: save a structured, tagged, cross-linked note back to the "
        "knowledge base"
    )
    if web_tools:
        tool_lines.append("- web_search: search the live web for current, real-world information")
        tool_lines.append("- web_fetch: fetch the full text of a specific web page by URL")
    if connector_tool_names:
        tool_lines.extend(f"- {t}" for t in connector_tool_names)
    elif not web_tools:
        tool_lines.append(
            "- (no external connector or web tools are attached — you can only use "
            "the knowledge base; flag anything you cannot verify as needs_human)"
        )
    tools_block = "\n".join(tool_lines)

    return (
        f"You are {name}, working inside the {ctx_name} workspace.\n\n"
        f"Your role: {role}\n\n"
        f"You maintain and rely on a shared knowledge base for this workspace — "
        f"treat it as a living wiki that you both read from and grow.\n\n"
        f"You have access to the following knowledge:\n"
        f"{_kb_access_line(agent.get('kb_scope', 'scoped'), agent.get('kb_section_ids', []) or [])}\n\n"
        f"You can use these tools:\n"
        f"{tools_block}\n\n"
        f"{_KNOWLEDGE_PROCEDURE}\n\n"
        f"{_GROUNDING}\n\n"
        f"Today is {today}."
    )
