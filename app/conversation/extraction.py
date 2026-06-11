"""Konuşma oturumlarından kalıcı gerçek çıkarımı.

Etkinse (``conversation_extraction_enabled``) ``HaikuFactExtractor`` Claude
Haiku ile gerçekleri çıkarır; ``ExtractionWorker`` bunları ``KBWriter`` ile
chunk+embed edip KB'ye yazar. Devre dışıysa ``NoopFactExtractor`` hiçbir şey
çıkarmaz/yazmaz (oturumları yalnızca ``extracted`` olarak işaretler).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from app.conversation.models import (
    ROLE_ASSISTANT,
    ROLE_QUERY,
    ROLE_USER,
    STATUS_EXTRACTED,
    STATUS_PENDING_EXTRACTION,
)

if TYPE_CHECKING:
    from app.config import Settings
    from app.conversation.kb_writer import KBWriter
    from app.conversation.models import ConversationSession, ConversationTurn
    from app.conversation.store import ConversationStore

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = (
    "You extract durable, reusable facts from a conversation between a user and "
    "an AI assistant. A durable fact is information worth remembering for future "
    "conversations: stable facts about the user, their organization, decisions, "
    "preferences, commitments, or project details. Ignore small talk, transient "
    "context, questions, and anything that will not be true next week. Write each "
    "fact as a single self-contained sentence that makes sense without the "
    "conversation. Return an empty list if there is nothing durable."
)

# Structured-output schema: force a JSON object with a `facts` string array.
_FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["facts"],
    "additionalProperties": False,
}


class FactExtractor(Protocol):
    """Turns a closed session's turns into durable fact strings."""

    def extract(
        self,
        session: "ConversationSession",
        turns: list["ConversationTurn"],
    ) -> list[str]:
        ...


class NoopFactExtractor:
    """Active when extraction is disabled: extracts nothing, writes nothing."""

    def extract(
        self,
        session: "ConversationSession",
        turns: list["ConversationTurn"],
    ) -> list[str]:
        logger.info(
            "noop_extract session=%s turns=%d exchanges=%d (extraction disabled)",
            session.session_id,
            len(turns),
            session.exchange_count,
        )
        return []


def _format_transcript(turns: list["ConversationTurn"], max_chars: int = 24000) -> str:
    label = {ROLE_USER: "User", ROLE_ASSISTANT: "Assistant", ROLE_QUERY: "User (query)"}
    lines = []
    for t in turns:
        who = label.get(t.role, t.role)
        lines.append(f"{who}: {t.content}".strip())
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


class HaikuFactExtractor:
    """Extract facts with Claude Haiku (sync client; called from the sweep thread)."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._model = settings.extraction_model or "claude-haiku-4-5"

    def _client(self):
        import anthropic

        key = (self._settings.anthropic_api_key or "").strip()
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def extract(
        self,
        session: "ConversationSession",
        turns: list["ConversationTurn"],
    ) -> list[str]:
        transcript = _format_transcript(turns)
        if not transcript.strip():
            return []
        try:
            client = self._client()
            resp = client.messages.create(
                model=self._model,
                max_tokens=2000,
                system=_EXTRACTION_SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": _FACTS_SCHEMA}},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract durable facts from this conversation:\n\n" + transcript
                        ),
                    }
                ],
            )
        except Exception:  # noqa: BLE001
            logger.warning("haiku_extract_failed session=%s", session.session_id, exc_info=True)
            return []

        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            data = json.loads(text)
            facts = data.get("facts", []) if isinstance(data, dict) else []
            return [str(f).strip() for f in facts if str(f).strip()]
        except (json.JSONDecodeError, TypeError):
            logger.warning("haiku_extract_parse_failed session=%s", session.session_id)
            return []


class ExtractionWorker:
    """Drives extraction + KB-write over ``pending_extraction`` sessions."""

    def __init__(
        self,
        store: "ConversationStore",
        extractor: FactExtractor,
        kb_writer: "KBWriter | None" = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._kb_writer = kb_writer

    def run_pending(self, limit: int = 50) -> int:
        sessions = self._store.list_sessions(
            status=STATUS_PENDING_EXTRACTION, limit=limit
        )
        handled = 0
        for session in sessions:
            try:
                turns = self._store.get_session_turns(session.session_id)
                facts = self._extractor.extract(session, turns)
                if facts and self._kb_writer is not None:
                    self._kb_writer.write_facts(session, facts)
                self._store.mark_session(session.session_id, STATUS_EXTRACTED)
                handled += 1
                logger.debug(
                    "extraction_done session=%s facts=%d",
                    session.session_id,
                    len(facts),
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "extraction_failed session=%s", session.session_id, exc_info=True
                )
        if handled:
            logger.info("extraction_worker_ran handled=%d", handled)
        return handled


def build_extractor(settings: "Settings") -> FactExtractor:
    """Select the active extractor based on config."""
    if settings.conversation_extraction_enabled:
        logger.info("extraction_backend=haiku model=%s", settings.extraction_model)
        return HaikuFactExtractor(settings)
    return NoopFactExtractor()
