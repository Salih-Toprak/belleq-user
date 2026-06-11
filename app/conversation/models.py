"""Veri modelleri: konuşma turu ve oturumu."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Turn roles.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_QUERY = "query"  # passive query-stream capture (no assistant answer seen)

# Turn sources.
SOURCE_MCP_TOOL = "mcp_tool"
SOURCE_QUERY_STREAM = "query_stream"

# Session lifecycle states.
STATUS_OPEN = "open"
STATUS_CLOSED = "closed"
STATUS_PENDING_EXTRACTION = "pending_extraction"
STATUS_EXTRACTED = "extracted"
STATUS_SKIPPED = "skipped"

SESSION_STATUSES = frozenset(
    {
        STATUS_OPEN,
        STATUS_CLOSED,
        STATUS_PENDING_EXTRACTION,
        STATUS_EXTRACTED,
        STATUS_SKIPPED,
    }
)


@dataclass
class ConversationTurn:
    """A single recorded turn in a conversation session."""

    turn_id: str
    session_id: str
    role: str
    content: str
    source: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """A group of turns treated as one conversation for extraction purposes."""

    session_id: str
    source: str
    status: str
    exchange_count: int
    started_at: datetime
    last_activity_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
