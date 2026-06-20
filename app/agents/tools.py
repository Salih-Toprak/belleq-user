"""Agent tools: kb_read, kb_write, and connector tools.

kb_read / kb_write reuse the container's in-process retrieval + writer
(QueryPipeline, KBWriter) — no reimplementation. Connector tools are reached
through the master's aggregated MCP endpoint for this container
(``connectors_mcp_url``), filtered to the agent's permitted connectors by their
namespace prefix (matching the master dispatcher's ``_safe_namespace``).

Every tool call appends a step to ``self.steps`` (persisted by the backend as an
AgentRun) and, for kb_write, to ``self.kb_writes`` (mirrored to the task).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _safe_namespace(connector_id: str) -> str:
    """Connector id -> tool-name-safe prefix (must match the master dispatcher)."""
    ns = re.sub(r"[^a-zA-Z0-9_]", "_", connector_id).strip("_")
    return ns or "tool"


def _truncate(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "…"


KB_READ_SPEC = {
    "name": "kb_read",
    "description": "Search the knowledge base for relevant information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default 8).", "default": 8},
        },
        "required": ["query"],
    },
}

KB_WRITE_SPEC = {
    "name": "kb_write",
    "description": (
        "Save important findings, decisions, or learned information back to the "
        "knowledge base. scope='private' writes to your scoped KB now; "
        "scope='shared' queues it for human review before promotion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to save."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Labels."},
            "scope": {"type": "string", "enum": ["private", "shared"], "default": "private"},
        },
        "required": ["content"],
    },
}


class Toolbox:
    def __init__(
        self,
        *,
        pipeline: Any,
        kb_writer: Any,
        agent: dict,
        task: dict,
        connectors_mcp_url: str,
    ) -> None:
        self._pipeline = pipeline
        self._kb_writer = kb_writer
        self._agent = agent
        self._task = task
        self._mcp_url = (connectors_mcp_url or "").strip()
        self._allowed_prefixes = tuple(
            f"{_safe_namespace(cid)}_" for cid in (agent.get("connector_ids") or [])
        )
        self._connector_specs: list[dict] = []
        self._connector_names: set[str] = set()
        # Run-state collected for the backend.
        self.steps: list[dict] = []
        self.kb_writes: list[dict] = []
        self._step_no = 0

    # ── setup ────────────────────────────────────────────────────────────────
    async def load_connector_tools(self) -> None:
        """Discover the agent's permitted connector tools from the aggregated MCP."""
        if not self._mcp_url or not self._allowed_prefixes:
            return
        try:
            client = self._client()
            async with client:
                tools = await client.list_tools()
            for t in tools:
                name = getattr(t, "name", "")
                if not name.startswith(self._allowed_prefixes):
                    continue
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {
                    "type": "object",
                    "properties": {},
                }
                self._connector_specs.append(
                    {
                        "name": name,
                        "description": getattr(t, "description", "") or "",
                        "input_schema": schema,
                    }
                )
                self._connector_names.add(name)
            logger.info("agent_connector_tools_loaded count=%d", len(self._connector_specs))
        except Exception:  # noqa: BLE001 — connectors are best-effort
            logger.warning("agent_connector_tools_load_failed url=%s", self._mcp_url, exc_info=True)

    def _client(self):
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        return Client(StreamableHttpTransport(url=self._mcp_url))

    def specs(self) -> list[dict]:
        return [KB_READ_SPEC, KB_WRITE_SPEC, *self._connector_specs]

    def connector_tool_names(self) -> list[str]:
        return sorted(self._connector_names)

    # ── execution ──────────────────────────────────────────────────────────--
    def _log(self, type_: str, input_summary: str, output_summary: str) -> None:
        self._step_no += 1
        self.steps.append(
            {
                "step_number": self._step_no,
                "type": type_,
                "input_summary": _truncate(input_summary),
                "output_summary": _truncate(output_summary),
            }
        )

    def next_step_number(self) -> int:
        return self._step_no + 1

    def record_step(self, type_: str, input_summary: str, output_summary: str) -> None:
        """Public hook for the runner to log non-tool steps (e.g. llm_call)."""
        self._log(type_, input_summary, output_summary)

    async def execute(self, name: str, args: dict) -> str:
        """Run one tool call and return its result text (for the LLM tool result)."""
        if name == "kb_read":
            return await self._kb_read(args)
        if name == "kb_write":
            return await self._kb_write(args)
        if name in self._connector_names:
            return await self._connector_call(name, args)
        msg = f"Unknown tool: {name}"
        self._log("connector_call", name, msg)
        return msg

    async def _kb_read(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        top_k = int(args.get("top_k", 8) or 8)
        if not query:
            self._log("kb_read", "(empty query)", "error: query required")
            return "Error: query is required."
        result = await self._pipeline.query(query, top_k=top_k)
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        out = json.dumps({"chunks": chunks}, ensure_ascii=False)
        self._log("kb_read", f"query={query!r} top_k={top_k}", f"{len(chunks)} chunks")
        return out

    async def _kb_write(self, args: dict) -> str:
        content = str(args.get("content", "")).strip()
        tags = [str(t) for t in (args.get("tags") or [])]
        scope = str(args.get("scope", "private")).lower()
        if scope not in ("private", "shared"):
            scope = "private"
        if not content:
            self._log("kb_write", "(empty)", "error: content required")
            return "Error: content is required."

        record = {"content": content, "tags": tags, "scope": scope}
        self.kb_writes.append(record)

        if scope == "shared":
            # Queued for review by the backend (propagate_to_master_kb); not
            # written to the KB here.
            self._log("kb_write", f"scope=shared tags={tags}", "queued for review")
            return "Saved (shared) — queued for human review before promotion."

        # Private: write to the context KB now, tagged with agent + task source.
        written = await asyncio.to_thread(self._write_private, content, tags)
        self._log("kb_write", f"scope=private tags={tags}", f"wrote {written} point(s)")
        return f"Saved to the knowledge base ({written} point)."

    def _write_private(self, content: str, tags: list[str]) -> int:
        agent_name = self._agent.get("name") or "agent"
        task_id = self._task.get("id", "")
        return self._kb_writer.write_chunks(
            doc_id=f"agent-{self._agent.get('id','')}-task-{task_id}-{abs(hash(content)) % 10_000_000}",
            doc_title=f"Agent note: {agent_name}",
            doc_path=f"agent:{agent_name}:{task_id}",
            source="agent",
            chunks=[content],
            extra_payload={
                "agent_id": self._agent.get("id", ""),
                "agent_name": agent_name,
                "task_id": task_id,
                "tags": tags,
                "scope": "private",
            },
        )

    async def _connector_call(self, name: str, args: dict) -> str:
        try:
            client = self._client()
            async with client:
                result = await client.call_tool(name, args)
            text = _extract_mcp_text(result)
            self._log("connector_call", f"{name} args={json.dumps(args)[:300]}", text)
            return text
        except Exception as exc:  # noqa: BLE001
            msg = f"Connector tool '{name}' failed: {exc}"
            self._log("connector_call", name, msg)
            return msg


def _extract_mcp_text(result: Any) -> str:
    """Pull text out of a FastMCP call_tool result across version shapes."""
    # Newer fastmcp exposes `.data` (structured) and `.content` (blocks).
    data = getattr(result, "data", None)
    if isinstance(data, (str, int, float, bool)):
        return str(data)
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    content = getattr(result, "content", None)
    if content:
        parts = [getattr(b, "text", "") for b in content if getattr(b, "text", "")]
        if parts:
            return "\n".join(parts)
    return str(result)
