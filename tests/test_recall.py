"""`recent_facts` recall tests — rag_wiki-isolated.

Exercises ordering (newest doc first, by chunk_index within a doc), the limit
cap, and graceful skipping of a doc whose vector lookup fails. Uses a custom
``runner`` that awaits the coroutine directly, so the persistent loop / rag_wiki
are never imported.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.conversation.recall import recent_facts

NOW = datetime(2026, 6, 18, tzinfo=timezone.utc)


async def _direct_runner(coro):
    return await coro


def _doc(doc_id, source="conversation", age_minutes=0):
    return SimpleNamespace(
        doc_id=doc_id,
        source=source,
        ingested_at=NOW - timedelta(minutes=age_minutes),
        last_updated_at=NOW - timedelta(minutes=age_minutes),
    )


class FakeGlobalStore:
    def __init__(self, docs):
        self._docs = docs

    def list_all(self, limit=50):
        return list(self._docs)[:limit]


class FakeVectorDB:
    """Maps doc_id -> list of {id, payload} chunks; can raise for a given id."""

    def __init__(self, by_doc, fail_ids=()):
        self._by_doc = by_doc
        self._fail = set(fail_ids)

    async def get_by_doc_id(self, collection, doc_id):
        if doc_id in self._fail:
            raise RuntimeError("boom")
        return list(self._by_doc.get(doc_id, []))


def _chunk(doc_id, idx, text):
    return {"id": f"{doc_id}:{idx}", "payload": {"doc_id": doc_id, "chunk_index": idx, "text": text, "session_id": doc_id}}


@pytest.mark.asyncio
async def test_recent_facts_orders_newest_first_and_by_chunk_index():
    docs = [_doc("conv-old", age_minutes=100), _doc("conv-new", age_minutes=1)]
    vd = FakeVectorDB({
        "conv-new": [_chunk("conv-new", 1, "second"), _chunk("conv-new", 0, "first")],
        "conv-old": [_chunk("conv-old", 0, "older fact")],
    })
    out = await recent_facts(FakeGlobalStore(docs), vd, "c_x", limit=10, runner=_direct_runner)
    texts = [f["text"] for f in out["facts"]]
    # Newest doc first; within a doc, chunk_index order.
    assert texts == ["first", "second", "older fact"]
    assert out["count"] == 3


@pytest.mark.asyncio
async def test_recent_facts_respects_limit():
    docs = [_doc("conv-a", age_minutes=1), _doc("conv-b", age_minutes=2)]
    vd = FakeVectorDB({
        "conv-a": [_chunk("conv-a", 0, "a0"), _chunk("conv-a", 1, "a1")],
        "conv-b": [_chunk("conv-b", 0, "b0")],
    })
    out = await recent_facts(FakeGlobalStore(docs), vd, "c_x", limit=2, runner=_direct_runner)
    assert [f["text"] for f in out["facts"]] == ["a0", "a1"]
    assert out["count"] == 2


@pytest.mark.asyncio
async def test_recent_facts_ignores_non_conversation_and_handles_failures():
    docs = [
        _doc("doc-slack", source="slack", age_minutes=0),  # filtered out
        _doc("conv-bad", age_minutes=1),                   # vector lookup fails -> skipped
        _doc("conv-ok", age_minutes=2),
    ]
    vd = FakeVectorDB(
        {"conv-ok": [_chunk("conv-ok", 0, "kept")]},
        fail_ids=["conv-bad"],
    )
    out = await recent_facts(FakeGlobalStore(docs), vd, "c_x", limit=10, runner=_direct_runner)
    assert [f["text"] for f in out["facts"]] == ["kept"]


@pytest.mark.asyncio
async def test_recent_facts_empty_when_no_docs_or_no_vectordb():
    assert await recent_facts(FakeGlobalStore([]), FakeVectorDB({}), "c", runner=_direct_runner) == {
        "facts": [],
        "count": 0,
    }
    assert await recent_facts(FakeGlobalStore([_doc("conv-a")]), None, "c", runner=_direct_runner) == {
        "facts": [],
        "count": 0,
    }
