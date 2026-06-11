"""Konuşma yakalama + arşiv + oturum yaşam döngüsü testleri.

rag_wiki bağımlılığı gerektirmez — yalnızca app.conversation.* + stdlib.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.conversation.capture import ConversationCapture
from app.conversation.extraction import ExtractionWorker, NoopFactExtractor
from app.conversation.models import (
    ROLE_QUERY,
    STATUS_EXTRACTED,
    STATUS_OPEN,
    STATUS_PENDING_EXTRACTION,
    STATUS_SKIPPED,
)
from app.conversation.session_manager import SessionManager
from app.conversation.store import ConversationStore


def make_settings(**over):
    base = dict(
        conversation_min_exchanges=10,
        conversation_session_idle_minutes=0,
        conversation_sweep_interval_minutes=10,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture()
def store(tmp_path):
    s = ConversationStore(str(tmp_path / "conversations.db"))
    yield s
    s.close()


def test_store_records_exchange_and_turns(store):
    store.ensure_session("s1", source="mcp_tool")
    store.append_turn("s1", "user", "hi", "mcp_tool")
    store.append_turn("s1", "assistant", "hello", "mcp_tool")
    store.touch_session("s1", exchange_delta=1)

    session = store.get_session("s1")
    assert session is not None
    assert session.status == STATUS_OPEN
    assert session.exchange_count == 1

    turns = store.get_session_turns("s1")
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].content == "hi"


def test_idle_detection_by_cutoff(store):
    store.ensure_session("s1", source="mcp_tool")
    now = datetime.now(timezone.utc)
    # Cutoff in the future -> session counts as idle.
    assert [s.session_id for s in store.get_idle_open_sessions(now + timedelta(minutes=1))] == ["s1"]
    # Cutoff in the past -> not idle yet.
    assert store.get_idle_open_sessions(now - timedelta(minutes=1)) == []


def test_sweep_routes_by_exchange_threshold(store):
    # Busy session: 10 exchanges -> pending_extraction.
    store.ensure_session("busy", source="mcp_tool")
    for _ in range(10):
        store.touch_session("busy", exchange_delta=1)
    # Quiet session: 3 exchanges -> skipped.
    store.ensure_session("quiet", source="mcp_tool")
    for _ in range(3):
        store.touch_session("quiet", exchange_delta=1)

    worker = ExtractionWorker(store, NoopFactExtractor())
    mgr = SessionManager("u", store, worker, make_settings())
    time.sleep(0.01)  # ensure last_activity < sweep cutoff
    result = mgr.sweep_once()

    assert result["closed"] == 2
    assert result["pending"] == 1
    assert result["skipped"] == 1
    # Worker ran immediately on the pending session -> extracted.
    assert result["extracted"] == 1
    assert store.get_session("busy").status == STATUS_EXTRACTED
    assert store.get_session("quiet").status == STATUS_SKIPPED


def test_noop_extractor_writes_nothing(store):
    store.ensure_session("s1", source="mcp_tool")
    store.mark_session("s1", STATUS_PENDING_EXTRACTION)
    worker = ExtractionWorker(store, NoopFactExtractor())
    handled = worker.run_pending()
    assert handled == 1
    assert store.get_session("s1").status == STATUS_EXTRACTED
    # No new turns were written by extraction.
    assert store.get_session_turns("s1") == []


def test_capture_record_exchange(store):
    cap = ConversationCapture(store, make_settings())
    ack = cap.record_exchange("what's our refund policy?", "30 days.", conversation_id="chat-1")
    assert ack["recorded"] is True
    assert ack["session_id"] == "chat-1"
    assert ack["exchange_count"] == 1
    turns = store.get_session_turns("chat-1")
    assert [t.role for t in turns] == ["user", "assistant"]


def test_capture_query_stream_windowing(store):
    cap = ConversationCapture(store, make_settings(conversation_session_idle_minutes=30))
    cap.record_query("first question")
    cap.record_query("second question")
    # Both land in the same rolling window session.
    sessions = store.list_sessions()
    qs = [s for s in sessions if s.session_id.startswith("qs-")]
    assert len(qs) == 1
    turns = store.get_session_turns(qs[0].session_id)
    assert len(turns) == 2
    assert all(t.role == ROLE_QUERY for t in turns)


def test_worker_writes_facts_to_kb(store):
    """When an extractor returns facts, the worker hands them to the KB writer."""
    store.ensure_session("s1", source="mcp_tool")
    store.mark_session("s1", STATUS_PENDING_EXTRACTION)

    class FakeExtractor:
        def extract(self, session, turns):
            return ["The user prefers Python.", "The team ships on Fridays."]

    written = {}

    class FakeKBWriter:
        def write_facts(self, session, facts):
            written[session.session_id] = list(facts)
            return len(facts)

    worker = ExtractionWorker(store, FakeExtractor(), kb_writer=FakeKBWriter())
    assert worker.run_pending() == 1
    assert written == {"s1": ["The user prefers Python.", "The team ships on Fridays."]}
    assert store.get_session("s1").status == STATUS_EXTRACTED


def test_haiku_extractor_parses_facts(monkeypatch):
    from app.conversation.extraction import HaikuFactExtractor
    from app.conversation.models import ConversationSession, ConversationTurn

    settings = make_settings(extraction_model="claude-haiku-4-5", anthropic_api_key="")
    extractor = HaikuFactExtractor(settings)

    class _Block:
        type = "text"
        text = '{"facts": ["User is the CTO of FAIRBANK.", "Budget approved for Q3."]}'

    class _Resp:
        content = [_Block()]

    class _Msgs:
        def create(self, **kwargs):
            return _Resp()

    class _FakeClient:
        messages = _Msgs()

    monkeypatch.setattr(extractor, "_client", lambda: _FakeClient())

    now = datetime.now(timezone.utc)
    session = ConversationSession("s1", "mcp_tool", STATUS_PENDING_EXTRACTION, 12, now, now)
    turns = [ConversationTurn("t1", "s1", "user", "I'm the CTO of FAIRBANK.", "mcp_tool", now)]
    facts = extractor.extract(session, turns)
    assert facts == ["User is the CTO of FAIRBANK.", "Budget approved for Q3."]


def test_capture_never_raises_on_bad_store():
    class BoomStore:
        def ensure_session(self, *a, **k):
            raise RuntimeError("boom")

    cap = ConversationCapture(BoomStore(), make_settings())
    ack = cap.record_exchange("u", "a")
    assert ack["recorded"] is False
    # Passive path swallows errors too.
    cap.record_query("q")  # must not raise
