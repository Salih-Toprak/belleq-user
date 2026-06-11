"""Konuşma yakalama servisi: açık MCP aracı + pasif sorgu akışı.

İki giriş noktası:
- ``record_exchange`` — AI istemcisinin çağırdığı MCP aracı (kullanıcı + asistan).
- ``record_query``    — QueryPipeline'dan pasif sorgu akışı (yalnızca sorgu).

En iyi çaba ilkesi: yakalama hataları asla çağırana sızmaz.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.conversation.models import (
    ROLE_ASSISTANT,
    ROLE_QUERY,
    ROLE_USER,
    SOURCE_MCP_TOOL,
    SOURCE_QUERY_STREAM,
)

if TYPE_CHECKING:
    from app.config import Settings
    from app.conversation.store import ConversationStore

logger = logging.getLogger(__name__)


class ConversationCapture:
    """Routes recorded turns into the conversation archive."""

    def __init__(self, store: "ConversationStore", settings: "Settings") -> None:
        self._store = store
        self._settings = settings

    def bind_settings(self, settings: "Settings") -> None:
        """runtime_config güncellemelerinde ayar referansını yeniler."""
        self._settings = settings

    # --- explicit capture (MCP tool) ----------------------------------

    def record_exchange(
        self,
        user_message: str,
        assistant_message: str,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one user/assistant exchange (two turns) into a session.

        ``conversation_id`` groups turns into a session; when omitted a new
        session id is minted (one-off exchange). Returns a small ack dict.
        Never raises — capture must not break the caller's tool call.
        """
        try:
            session_id = (conversation_id or "").strip() or f"exch-{uuid.uuid4().hex}"
            self._store.ensure_session(
                session_id,
                source=SOURCE_MCP_TOOL,
                metadata={"conversation_id": conversation_id} if conversation_id else {},
            )
            self._store.append_turn(
                session_id, ROLE_USER, user_message, SOURCE_MCP_TOOL, metadata
            )
            self._store.append_turn(
                session_id, ROLE_ASSISTANT, assistant_message, SOURCE_MCP_TOOL, metadata
            )
            self._store.touch_session(session_id, exchange_delta=1)
            count = self._store.count_exchanges(session_id)
            logger.info(
                "conversation_exchange_recorded session=%s exchanges=%d",
                session_id,
                count,
            )
            return {"recorded": True, "session_id": session_id, "exchange_count": count}
        except Exception:  # noqa: BLE001
            logger.warning("record_exchange_failed", exc_info=True)
            return {"recorded": False, "session_id": conversation_id or "", "exchange_count": 0}

    # --- passive capture (query stream) -------------------------------

    def record_query(self, query: str, conversation_id: str | None = None) -> None:
        """Record a passive query turn.

        With no explicit ``conversation_id`` the turn lands in a rolling
        query-stream session keyed by a time window, so a burst of queries
        groups into one session while idle gaps start a fresh one.
        Best-effort and silent on failure.
        """
        try:
            session_id = (conversation_id or "").strip() or self._query_window_session_id()
            self._store.ensure_session(
                session_id, source=SOURCE_QUERY_STREAM, metadata={"window": True}
            )
            self._store.append_turn(session_id, ROLE_QUERY, query, SOURCE_QUERY_STREAM, None)
            # A standalone query counts as one exchange for the >=N threshold.
            self._store.touch_session(session_id, exchange_delta=1)
        except Exception:  # noqa: BLE001
            logger.debug("record_query_failed", exc_info=True)

    def _query_window_session_id(self) -> str:
        """Rolling session id bucketed by the idle window length.

        Buckets queries by floor(now / idle_window). Within one window all
        passive queries share a session; the sweep closes it once the window
        passes with no activity.
        """
        window_min = max(1, int(self._settings.conversation_session_idle_minutes))
        now = datetime.now(timezone.utc)
        bucket = int(now.timestamp()) // (window_min * 60)
        return f"qs-{bucket}"
