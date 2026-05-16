"""MCP (Model Context Protocol) sunucusu."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from app.query.pipeline import QueryPipeline

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


def build_mcp_server(pipeline: QueryPipeline, settings: "Settings") -> FastMCP:
    """
    Tek araçlı MCP sunucusu: query_knowledge_base.
    FastAPI uygulamasında /mcp altına monte edilir.
    """

    mcp = FastMCP(settings.resolved_mcp_server_name)

    @mcp.tool()
    async def query_knowledge_base(query: str) -> str:
        """
        Retrieve relevant document chunks from the Mnemo knowledge base.

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

    logger.info("mcp_server_built name=%s", settings.resolved_mcp_server_name)
    return mcp
