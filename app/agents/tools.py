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
        "Save a structured note to the knowledge base — this is how you grow a "
        "self-organizing wiki. Write ONE idea per note, with a clear title, tags, "
        "and [[wikilinks]] to related notes so knowledge stays connected and "
        "compounds across tasks. Before writing, kb_read for an existing note on "
        "the same topic and reuse its title to supersede it rather than "
        "duplicating. scope='private' writes to your scoped KB now; scope='shared' "
        "queues it for human review before promotion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title naming the single thing this note is about.",
            },
            "content": {
                "type": "string",
                "description": (
                    "The note body: the fact/finding/decision stated plainly. May "
                    "include [[Other Note Title]] wikilinks and a Source: line."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-5 lowercase keyword tags for retrieval.",
            },
            "related": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Titles of related notes to cross-link (wikilinks).",
            },
            "source": {
                "type": "string",
                "description": "Where this came from: a connector/tool name, URL, or 'reasoning'.",
            },
            "scope": {"type": "string", "enum": ["private", "shared"], "default": "private"},
        },
        "required": ["content"],
    },
}

WEB_SEARCH_SPEC = {
    "name": "web_search",
    "description": (
        "Search the live web for current, real-world information — competitors, "
        "people, companies, prices, news, social handles, anything you can't get "
        "from the knowledge base or a connector. Returns ranked results with the "
        "page title, URL, and an extracted content snippet. Use this (not your own "
        "memory) for any external fact, and cite the result URL as the source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

WEB_FETCH_SPEC = {
    "name": "web_fetch",
    "description": (
        "Fetch the full clean text of a specific web page by URL (e.g. a result "
        "from web_search, a company's pricing page, a docs page). Use this to read "
        "a source in depth before writing a grounded note that cites it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The full URL to fetch."},
        },
        "required": ["url"],
    },
}

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


def _compose_note(title: str, content: str, tags: list[str], related: list[str], source: str) -> str:
    """Assemble a self-contained wiki note (frontmatter-ish header + body).

    Stored verbatim as the chunk text so retrieval surfaces the structure and the
    [[wikilinks]], keeping the KB navigable like an LLM-wiki page.
    """
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
    meta: list[str] = []
    if tags:
        meta.append("tags: " + ", ".join(tags))
    if related:
        meta.append("related: " + ", ".join(f"[[{r}]]" for r in related))
    if source:
        meta.append(f"source: {source}")
    if meta:
        lines.append("\n".join(meta))
    lines.append(content)
    return "\n\n".join(lines).strip()


class Toolbox:
    def __init__(
        self,
        *,
        pipeline: Any,
        kb_writer: Any,
        agent: dict,
        task: dict,
        connectors_mcp_url: str,
        tavily_api_key: str = "",
        step_callback: dict | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._kb_writer = kb_writer
        self._agent = agent
        self._task = task
        self._mcp_url = (connectors_mcp_url or "").strip()
        self._tavily_key = (tavily_api_key or "").strip()
        # Live-progress sink (url/token/task_id) the backend handed us; when None
        # the run still works, the dashboard just sees steps after completion.
        self._step_callback = step_callback or None
        self._emitted = 0  # how many steps have been streamed so far
        self._cancelled = False  # set when the backend signals a user Stop
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
        """Load connector tools from the context's aggregated MCP.

        Agents inherit every connector enabled for their context (the aggregated
        endpoint only exposes those), so we load them all — no per-agent picking.
        When ``_allowed_prefixes`` is set (the notifier passes specific connector
        ids) we filter to just those; otherwise we take everything."""
        if not self._mcp_url:
            return
        try:
            client = self._client()
            async with client:
                tools = await client.list_tools()
            for t in tools:
                name = getattr(t, "name", "")
                if self._allowed_prefixes and not name.startswith(self._allowed_prefixes):
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
        web = [WEB_SEARCH_SPEC, WEB_FETCH_SPEC] if self._tavily_key else []
        return [KB_READ_SPEC, KB_WRITE_SPEC, *web, *self._connector_specs]

    def web_tool_names(self) -> list[str]:
        return ["web_search", "web_fetch"] if self._tavily_key else []

    def connector_tool_names(self) -> list[str]:
        return sorted(self._connector_names)

    def connector_specs(self) -> list[dict]:
        """Just the connector tool specs (no kb/web) — used by the notifier."""
        return list(self._connector_specs)

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

    async def flush_steps(self) -> None:
        """Stream any steps logged since the last flush to the backend's live-
        progress sink. Best-effort: failures never interrupt the run (the final
        result still carries the full step log for authoritative persistence)."""
        cb = self._step_callback
        if not cb:
            return
        new = self.steps[self._emitted:]
        if not new:
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    cb["url"],
                    json={"task_id": cb.get("task_id"), "token": cb.get("token"), "steps": new},
                )
            self._emitted = len(self.steps)
            # The backend tells us here if the user pressed Stop.
            try:
                if resp.json().get("cancel"):
                    self._cancelled = True
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 — live progress is best-effort
            logger.debug("agent_step_flush_failed", exc_info=True)

    def cancel_requested(self) -> bool:
        return self._cancelled

    async def execute(self, name: str, args: dict) -> str:
        """Run one tool call and return its result text (for the LLM tool result)."""
        if name == "kb_read":
            return await self._kb_read(args)
        if name == "kb_write":
            return await self._kb_write(args)
        if name == "web_search":
            return await self._web_search(args)
        if name == "web_fetch":
            return await self._web_fetch(args)
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
        body = str(args.get("content", "")).strip()
        title = str(args.get("title", "")).strip()
        tags = [str(t) for t in (args.get("tags") or [])]
        related = [str(r) for r in (args.get("related") or [])]
        source = str(args.get("source", "")).strip()
        scope = str(args.get("scope", "private")).lower()
        if scope not in ("private", "shared"):
            scope = "private"
        if not body:
            self._log("kb_write", "(empty)", "error: content required")
            return "Error: content is required."

        note = _compose_note(title, body, tags, related, source)
        # Record the assembled note (what's stored) plus structure for the task log.
        record = {
            "content": note,
            "title": title,
            "tags": tags,
            "related": related,
            "scope": scope,
        }
        self.kb_writes.append(record)

        if scope == "shared":
            # Queued for review by the backend (propagate_to_master_kb); not
            # written to the KB here.
            self._log("kb_write", f"shared title={title!r} tags={tags}", "queued for review")
            return "Saved (shared) — queued for human review before promotion."

        # Private: write to the context KB now, tagged with agent + task source.
        written = await asyncio.to_thread(self._write_private, title, note, tags, related)
        self._log("kb_write", f"private title={title!r} tags={tags}", f"wrote {written} point(s)")
        return f"Saved note to the knowledge base ({written} point)."

    def _write_private(self, title: str, note: str, tags: list[str], related: list[str]) -> int:
        agent_name = self._agent.get("name") or "agent"
        task_id = self._task.get("id", "")
        doc_title = f"{title} (agent note)" if title else f"Agent note: {agent_name}"
        return self._kb_writer.write_chunks(
            doc_id=f"agent-{self._agent.get('id','')}-task-{task_id}-{abs(hash(note)) % 10_000_000}",
            doc_title=doc_title,
            doc_path=f"agent:{agent_name}:{task_id}",
            source="agent",
            chunks=[note],
            extra_payload={
                "agent_id": self._agent.get("id", ""),
                "agent_name": agent_name,
                "task_id": task_id,
                "note_title": title,
                "tags": tags,
                "related": related,
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

    # ── web (Tavily) ─────────────────────────────────────────────────────────
    async def _web_search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            self._log("web_search", "(empty query)", "error: query required")
            return "Error: query is required."
        max_results = max(1, min(int(args.get("max_results", 5) or 5), 10))
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    TAVILY_SEARCH_URL,
                    json={
                        "api_key": self._tavily_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            msg = f"web_search failed: {exc}"
            self._log("web_search", query, msg)
            return msg

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": _truncate(r.get("content", ""), 800),
            }
            for r in (data.get("results") or [])
        ]
        out = json.dumps(
            {"answer": data.get("answer"), "results": results}, ensure_ascii=False
        )
        self._log("web_search", f"query={query!r} n={max_results}", f"{len(results)} results")
        return out

    async def _web_fetch(self, args: dict) -> str:
        url = str(args.get("url", "")).strip()
        if not url:
            self._log("web_fetch", "(empty url)", "error: url required")
            return "Error: url is required."
        try:
            import httpx

            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    TAVILY_EXTRACT_URL,
                    json={"api_key": self._tavily_key, "urls": [url]},
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            msg = f"web_fetch failed: {exc}"
            self._log("web_fetch", url, msg)
            return msg

        results = data.get("results") or []
        content = results[0].get("raw_content", "") if results else ""
        if not content:
            failed = data.get("failed_results") or []
            reason = failed[0].get("error") if failed else "no content extracted"
            self._log("web_fetch", url, f"empty: {reason}")
            return f"Could not extract content from {url}: {reason}"
        out = json.dumps({"url": url, "content": _truncate(content, 12000)}, ensure_ascii=False)
        self._log("web_fetch", url, f"{len(content)} chars")
        return out


# Keys whose values are plumbing, not information — connector/REST APIs (GitHub in
# particular) bury the useful fields under piles of these. We drop them so the
# agent (and the run log) keep the substance, not URLs/hashes/internal ids. Any
# key ending in "_url" is dropped too.
_NOISE_KEYS = frozenset({
    "url", "sha", "node_id", "gravatar_id", "_links", "etag",
    "blob_id", "tree_id", "commit_sha",
})  # NB: "id"/"number" are kept — the agent may need them for follow-up calls.
_MAX_LIST_ITEMS = 50      # cap long arrays (e.g. a 300-file directory listing)
_MAX_STR_LEN = 1500       # cap any single long string value
_MAX_DEPTH = 8


def _prune_json(obj: Any, depth: int = 0) -> Any:
    """Strip plumbing keys, cap long arrays/strings, and limit depth so a
    connector result carries information instead of noise."""
    if depth >= _MAX_DEPTH:
        return "…"
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in _NOISE_KEYS or kl.endswith("_url"):
                continue
            out[k] = _prune_json(v, depth + 1)
        return out
    if isinstance(obj, list):
        pruned = [_prune_json(x, depth + 1) for x in obj[:_MAX_LIST_ITEMS]]
        if len(obj) > _MAX_LIST_ITEMS:
            pruned.append(f"…(+{len(obj) - _MAX_LIST_ITEMS} more items)")
        return pruned
    if isinstance(obj, str) and len(obj) > _MAX_STR_LEN:
        return obj[:_MAX_STR_LEN] + "…"
    return obj


def _clean_text(s: str) -> str:
    """If a text payload is really JSON, prune it; otherwise return it as-is."""
    st = s.strip()
    if st[:1] in ("{", "["):
        try:
            return json.dumps(_prune_json(json.loads(st)), ensure_ascii=False)
        except (ValueError, TypeError):
            pass
    return s


def _extract_mcp_text(result: Any) -> str:
    """Pull text out of a FastMCP call_tool result across version shapes, pruning
    plumbing fields so we keep the information, not the raw dump."""
    # Newer fastmcp exposes `.data` (structured) and `.content` (blocks).
    data = getattr(result, "data", None)
    if isinstance(data, (str, int, float, bool)):
        return _clean_text(str(data))
    if isinstance(data, (dict, list)):
        return json.dumps(_prune_json(data), ensure_ascii=False)
    content = getattr(result, "content", None)
    if content:
        parts = [getattr(b, "text", "") for b in content if getattr(b, "text", "")]
        if parts:
            return _clean_text("\n".join(parts))
    return str(result)
