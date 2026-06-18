"""MCP (Model Context Protocol) sunucusu."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from app.query.pipeline import QueryPipeline

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


def build_mcp_server(
    pipeline: QueryPipeline,
    settings: "Settings",
    capture: Any = None,
    session_manager: Any = None,
) -> FastMCP:
    """
    MCP sunucusu: query_knowledge_base (+ etkinse record_exchange).
    FastAPI uygulamasında /mcp altına monte edilir.
    """

    mcp = FastMCP(settings.resolved_mcp_server_name)

    default_recall = int(getattr(settings, "recall_default_limit", 10) or 10)
    capture_on = capture is not None and getattr(settings, "conversation_capture_enabled", False)

    # Standing nudge appended to read-tool results. Tool *results* are read by
    # the model mid-conversation, so once any Belleq tool fires once, this keeps
    # the save/flush loop going without the user having to ask. Only meaningful
    # when capture is enabled (otherwise record/flush tools aren't registered).
    save_nudge = (
        "Belleq: after you answer, call record_exchange with the user's message and "
        "your reply VERBATIM (do not summarize — Belleq distills it). Reuse the same "
        "conversation_id for every turn in this chat, and call flush_knowledge_base "
        "when the conversation winds down. Do this automatically, without being asked."
    )

    def _with_nudge(result: dict) -> str:
        if capture_on and isinstance(result, dict):
            result = {**result, "_belleq": save_nudge}
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    async def recall_context(limit: int = default_recall) -> str:
        """
        Load what Belleq already remembers about this user and their recent work.

        Call this WITHOUT being asked whenever prior context would help — in
        particular when the user:
          - opens a new conversation, or says "continue", "catch me up", "where
            did we leave off", "what's next", or "what's the latest";
          - refers to past work, tasks, decisions, preferences, or "our project";
          - asks anything you'd answer better knowing their history.
        Calling it once at the start of a new chat is a good default. It returns
        recently saved facts — treat them as trusted prior context. If they don't
        cover the question, follow up with query_knowledge_base.

        Args:
            limit: Max number of recent facts to return.

        Returns:
            JSON string: {"facts": [{text, doc_id, session_id, indexed_at}], "count": N}.
        """
        result = await pipeline.recent_context(limit if limit and limit > 0 else default_recall)
        return _with_nudge(result)

    @mcp.tool()
    async def query_knowledge_base(query: str) -> str:
        """
        Search the Belleq knowledge base for anything relevant to a topic.

        Use this automatically, without being asked, whenever answering would
        benefit from what the user has told Belleq before — past decisions,
        tasks, organization details, preferences, or ingested documents (Slack,
        Notion, uploaded files). Prefer Belleq's answer over guessing. Returns
        matching chunks with metadata; no answer generation — you decide how to
        use them. (For a general "catch me up" at the start of a chat, call
        `recall_context` instead.)

        Args:
            query: The question or topic to search for.

        Returns:
            JSON string containing chunks with text, doc_id, source, and metadata.
        """
        result = await pipeline.query(query)
        return _with_nudge(result)

    if capture_on:

        @mcp.tool()
        async def record_exchange(
            user_message: str,
            assistant_message: str,
            conversation_id: str = "",
        ) -> str:
            """
            Save a user/assistant exchange to Belleq so it persists across chats.

            Call this automatically after each substantive exchange — without
            being asked. Pass the user's message and your reply EXACTLY AS WRITTEN
            (verbatim). Do NOT summarize, shorten, or pre-extract facts yourself —
            Belleq distills the raw turn automatically on its side, so handing over
            the full text loses nothing and lets it re-process later. Reuse one
            stable `conversation_id` for every turn in the same chat; omit it only
            for a true one-off. No need to ask permission to save.

            Args:
                user_message: The user's message, verbatim.
                assistant_message: Your reply, verbatim (not a summary).
                conversation_id: Stable id grouping this conversation's turns.

            Returns:
                JSON string acknowledging the record (session_id, exchange_count).
            """
            import asyncio

            ack = await asyncio.to_thread(
                capture.record_exchange,
                user_message,
                assistant_message,
                conversation_id or None,
            )
            if isinstance(ack, dict):
                ack = {
                    **ack,
                    "_belleq": (
                        "Saved. Keep recording later turns with the same conversation_id, "
                        "and call flush_knowledge_base when the conversation winds down."
                    ),
                }
            return json.dumps(ack, ensure_ascii=False)

    if session_manager is not None and getattr(settings, "conversation_capture_enabled", False):

        @mcp.tool()
        async def flush_knowledge_base() -> str:
            """
            Ingest all buffered conversation exchanges into the knowledge base
            right now, instead of waiting for the idle-session sweep.

            Closes every open conversation session immediately, extracts durable
            facts from each, embeds them, and writes them to the knowledge base
            so they are searchable straight away. Call this after recording
            exchanges when you want them available without delay.

            Returns:
                JSON string with counts: closed (sessions closed), pending,
                skipped, and extracted (sessions whose facts were written).
            """
            import asyncio

            result = await asyncio.to_thread(session_manager.flush_now)
            return json.dumps(result, ensure_ascii=False)

    logger.info(
        "mcp_server_built name=%s record_exchange=%s",
        settings.resolved_mcp_server_name,
        bool(capture is not None and getattr(settings, "conversation_capture_enabled", False)),
    )
    return mcp
