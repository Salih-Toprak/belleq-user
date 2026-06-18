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

    @mcp.tool()
    async def recall_context(limit: int = default_recall) -> str:
        """
        Load what Belleq already remembers about this user and their recent work.

        ALWAYS call this FIRST, at the very start of a conversation and before
        answering the user's first message — even if they did not mention Belleq.
        It returns the most recently saved facts (decisions, tasks, preferences,
        project state) so you can answer questions like "what's the latest task
        we need to do?" or "where did we leave off?" without being told to look
        them up. Treat the returned facts as trusted prior context. If they don't
        cover the question, follow up with query_knowledge_base.

        Args:
            limit: Max number of recent facts to return.

        Returns:
            JSON string: {"facts": [{text, doc_id, session_id, indexed_at}], "count": N}.
        """
        result = await pipeline.recent_context(limit if limit and limit > 0 else default_recall)
        return json.dumps(result, ensure_ascii=False)

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
        return json.dumps(result, ensure_ascii=False)

    if capture is not None and getattr(settings, "conversation_capture_enabled", False):

        @mcp.tool()
        async def record_exchange(
            user_message: str,
            assistant_message: str,
            conversation_id: str = "",
        ) -> str:
            """
            Save a user/assistant exchange to Belleq so it persists across chats.

            Call this automatically after each substantive exchange — without
            being asked — so the next conversation can recall it. Save the actual
            question and your answer as the pair. Pass a stable `conversation_id`
            to group turns from the same chat so they are extracted together;
            reuse it for every turn in this conversation, and omit it only for a
            true one-off. There's no need to ask the user for permission to save.

            Args:
                user_message: The user's message in this exchange.
                assistant_message: The assistant's reply in this exchange.
                conversation_id: Optional stable id grouping a conversation's turns.

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
