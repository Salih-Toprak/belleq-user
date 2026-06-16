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

    @mcp.tool()
    async def query_knowledge_base(query: str) -> str:
        """
        Retrieve relevant document chunks from the Belleq knowledge base.

        Search through your organization's ingested documents
        (Slack messages, Notion pages, uploaded files) and return
        matching chunks with metadata. No answer generation —
        the caller decides how to use the chunks.

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
            Save a notable user/assistant exchange to the Belleq knowledge base.

            Call this after exchanges that contain durable, reusable facts about
            the user, their organization, decisions, preferences, or project
            details — information worth remembering for future conversations.
            Pass a stable `conversation_id` to group turns from the same chat so
            they are extracted together; omit it for a one-off exchange.

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
