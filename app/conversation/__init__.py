"""Conversation history capture & archive.

Records user/assistant exchanges (via the ``record_exchange`` MCP tool) and
passive query traffic, groups them into sessions, detects session end by idle
gap, and routes closed sessions to fact extraction or skip. Fact extraction
itself (LLM -> chunk -> embed -> KB) is built in a later slice; this package
provides the capture + raw archive + session lifecycle.
"""

from app.conversation.capture import ConversationCapture
from app.conversation.models import ConversationSession, ConversationTurn
from app.conversation.store import ConversationStore

__all__ = [
    "ConversationCapture",
    "ConversationSession",
    "ConversationStore",
    "ConversationTurn",
]
