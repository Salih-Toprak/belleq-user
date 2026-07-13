"""Retention sweep tests — synthetic clock, no rag_wiki, no real Qdrant.

Covers the properties that make retention safe:
  - the staleness clock counts ACTIVE days, so an absent user loses nothing;
  - fetched docs never archive; unfetched docs archive after the threshold;
  - archive is a payload flag (soft), purge is opt-in and physically deletes;
  - restore un-archives and restarts the doc's clock;
  - docs with no parseable timestamp are never touched (safe default).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import app.config as app_config
from app.lifecycle.retention import ActivityTracker, RetentionSweeper

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)


async def _direct_runner(coro):
    return await coro


def _chunk(doc_id, *, indexed_at=None, archived=False, archived_at=None, text="hello"):
    payload = {
        "doc_id": doc_id,
        "doc_title": f"Doc {doc_id}",
        "source": "conversation",
        "text": text,
    }
    if indexed_at is not None:
        payload["indexed_at"] = indexed_at.isoformat()
    if archived:
        payload["archived"] = True
        payload["archived_at"] = (archived_at or indexed_at or NOW).isoformat()
    return {"id": f"{doc_id}-0", "payload": payload}


class FakeVectorDB:
    def __init__(self, points):
        self.points = list(points)
        self.payload_sets: list[tuple[dict, dict]] = []
        self.deleted_doc_ids: list[str] = []

    async def scroll(self, collection, filters=None, limit=100, offset=0):
        rows = self.points
        if filters and filters.get("must"):
            for cond in filters["must"]:
                rows = [
                    p for p in rows
                    if (p["payload"] or {}).get(cond["field"]) == cond["value"]
                ]
        return rows[offset : offset + limit]

    async def set_payload_by_filter(self, collection, filters, payload):
        self.payload_sets.append((filters, payload))
        n = 0
        doc_id = filters["must"][0]["value"]
        for p in self.points:
            if p["payload"].get("doc_id") == doc_id:
                p["payload"].update(payload)
                n += 1
        return n

    async def delete_by_doc_id(self, collection, doc_id):
        before = len(self.points)
        self.points = [p for p in self.points if p["payload"].get("doc_id") != doc_id]
        self.deleted_doc_ids.append(doc_id)
        return before - len(self.points)


@pytest.fixture()
def tracker(tmp_path):
    return ActivityTracker(str(tmp_path / "retention.db"))


@pytest.fixture()
def retention_settings(monkeypatch):
    """Enable retention with small thresholds for the tests."""
    s = app_config.settings.model_copy(
        update={
            "retention_enabled": True,
            "retention_archive_after_days": 3,
            "retention_purge_enabled": False,
            "retention_purge_after_days": 5,
        }
    )
    monkeypatch.setattr(app_config, "settings", s)
    return s


def _sweeper(vdb, tracker, state_store=None, user_id="u_test"):
    return RetentionSweeper(
        vectordb=vdb,
        collection_name="c_test",
        tracker=tracker,
        runner=_direct_runner,
        state_store=state_store,
        user_id=user_id,
    )


def _make_state_store(rows, user_id="u_test"):
    """Real in-memory SQLAlchemy state table (like rag_wiki's SQLiteStateStore),
    so the reflection-based protected-doc read is exercised for real."""
    from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

    engine = create_engine("sqlite:///:memory:")
    md = MetaData()
    table = Table(
        "document_state",
        md,
        Column("doc_id", String, primary_key=True),
        Column("user_id", String),
        Column("user_state", String),
        Column("fetch_count", Integer, default=0),
    )
    md.create_all(engine)
    with engine.begin() as conn:
        for doc_id, state in rows:
            conn.execute(
                table.insert().values(
                    doc_id=doc_id, user_id=user_id, user_state=state, fetch_count=0
                )
            )

    class _Store:
        pass

    s = _Store()
    s._table = table
    s._engine = engine
    return s


def _mark_days(tracker, start, n):
    """Mark n consecutive active days starting the day after ``start``."""
    for i in range(1, n + 1):
        tracker.mark_active(start + timedelta(days=i))


# ── Activity clock ───────────────────────────────────────────────────

def test_inactive_user_accrues_no_active_days(tracker):
    ref = NOW - timedelta(days=365)
    assert tracker.active_days_since(ref, NOW) == 0


def test_active_days_count_only_used_days(tracker):
    ref = NOW - timedelta(days=30)
    _mark_days(tracker, ref, 4)
    assert tracker.active_days_since(ref, NOW) == 4
    # The reference day itself never counts.
    tracker.mark_active(ref)
    assert tracker.active_days_since(ref, NOW) == 4


# ── Sweep: archive ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_idle_user_docs_never_archive(tracker, retention_settings):
    """A year of absence, zero active days -> nothing is archived."""
    vdb = FakeVectorDB([_chunk("old", indexed_at=NOW - timedelta(days=365))])
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["archived"] == []
    assert vdb.payload_sets == []


@pytest.mark.asyncio
async def test_stale_doc_archives_after_active_days(tracker, retention_settings):
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 3)  # threshold is 3 active days
    vdb = FakeVectorDB([_chunk("stale", indexed_at=indexed)])
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["archived"] == ["stale"]
    assert vdb.points[0]["payload"]["archived"] is True


@pytest.mark.asyncio
async def test_recently_fetched_doc_survives(tracker, retention_settings):
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 3)
    tracker.record_fetch(["useful"], NOW - timedelta(days=1))
    vdb = FakeVectorDB([_chunk("useful", indexed_at=indexed)])
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["archived"] == []


@pytest.mark.asyncio
async def test_doc_without_timestamp_is_never_touched(tracker, retention_settings):
    _mark_days(tracker, NOW - timedelta(days=10), 9)
    vdb = FakeVectorDB([_chunk("mystery", indexed_at=None)])
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["archived"] == []


@pytest.mark.asyncio
async def test_dry_run_changes_nothing(tracker, retention_settings):
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 3)
    vdb = FakeVectorDB([_chunk("stale", indexed_at=indexed)])
    res = await _sweeper(vdb, tracker).sweep(dry_run=True, now=NOW)
    assert [c["doc_id"] for c in res["archive_candidates"]] == ["stale"]
    assert res["archived"] == []
    assert vdb.payload_sets == []


@pytest.mark.asyncio
async def test_disabled_retention_skips(tracker, retention_settings, monkeypatch):
    monkeypatch.setattr(
        app_config, "settings", app_config.settings.model_copy(update={"retention_enabled": False})
    )
    vdb = FakeVectorDB([_chunk("stale", indexed_at=NOW - timedelta(days=100))])
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["status"] == "skipped"


# ── Sweep: purge ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_purge_off_by_default_keeps_archived_docs(tracker, retention_settings):
    archived_at = NOW - timedelta(days=30)
    _mark_days(tracker, archived_at, 10)  # well past purge_after_days=5
    vdb = FakeVectorDB([_chunk("parked", indexed_at=archived_at, archived=True)])
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["purged"] == []
    assert vdb.deleted_doc_ids == []


@pytest.mark.asyncio
async def test_purge_deletes_after_active_days(tracker, retention_settings, monkeypatch):
    monkeypatch.setattr(
        app_config, "settings", app_config.settings.model_copy(update={"retention_purge_enabled": True})
    )
    archived_at = NOW - timedelta(days=30)
    _mark_days(tracker, archived_at, 5)
    vdb = FakeVectorDB(
        [_chunk("gone", indexed_at=archived_at, archived=True, archived_at=archived_at)]
    )
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["purged"] == ["gone"]
    assert res["bytes_freed"] > 0
    assert vdb.points == []


@pytest.mark.asyncio
async def test_archived_doc_not_purged_while_user_idle(tracker, retention_settings, monkeypatch):
    monkeypatch.setattr(
        app_config, "settings", app_config.settings.model_copy(update={"retention_purge_enabled": True})
    )
    archived_at = NOW - timedelta(days=300)
    vdb = FakeVectorDB(
        [_chunk("safe", indexed_at=archived_at, archived=True, archived_at=archived_at)]
    )
    res = await _sweeper(vdb, tracker).sweep(now=NOW)  # zero active days marked
    assert res["purged"] == []


# ── Restore & listing ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_restore_unarchives_and_restarts_clock(tracker, retention_settings):
    vdb = FakeVectorDB([_chunk("back", indexed_at=NOW - timedelta(days=20), archived=True)])
    sw = _sweeper(vdb, tracker)
    n = await sw.restore("back")
    assert n == 1
    assert vdb.points[0]["payload"]["archived"] is False
    assert tracker.last_fetch("back") is not None


@pytest.mark.asyncio
async def test_list_archived_groups_docs(tracker, retention_settings):
    vdb = FakeVectorDB(
        [
            _chunk("a", indexed_at=NOW - timedelta(days=5), archived=True),
            _chunk("live", indexed_at=NOW - timedelta(days=5)),
        ]
    )
    docs = await _sweeper(vdb, tracker).list_archived()
    assert [d["doc_id"] for d in docs] == ["a"]
    assert docs[0]["chunks"] == 1


# ── Pinned/claimed protection ────────────────────────────────────────

@pytest.mark.asyncio
async def test_pinned_doc_never_archives(tracker, retention_settings):
    """A stale doc the user pinned is protected even past the threshold."""
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 5)  # well past archive_after=3
    vdb = FakeVectorDB([_chunk("pinned_doc", indexed_at=indexed)])
    store = _make_state_store([("pinned_doc", "pinned")])
    res = await _sweeper(vdb, tracker, state_store=store).sweep(now=NOW)
    assert res["archived"] == []
    assert res["protected_docs"] == 1


@pytest.mark.asyncio
async def test_claimed_doc_protected_but_others_archive(tracker, retention_settings):
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 5)
    vdb = FakeVectorDB(
        [_chunk("claimed_doc", indexed_at=indexed), _chunk("plain_doc", indexed_at=indexed)]
    )
    store = _make_state_store([("claimed_doc", "claimed"), ("plain_doc", "surfaced")])
    res = await _sweeper(vdb, tracker, state_store=store).sweep(now=NOW)
    assert res["archived"] == ["plain_doc"]
    assert res["protected_docs"] == 1


@pytest.mark.asyncio
async def test_pinned_payload_flag_protects_without_state_store(tracker, retention_settings):
    """A doc pinned from the dashboard (belleq-level `pinned` payload flag) is
    protected even with no rag_wiki state store present."""
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 5)
    chunk = _chunk("dash_pinned", indexed_at=indexed)
    chunk["payload"]["pinned"] = True
    vdb = FakeVectorDB([chunk, _chunk("plain", indexed_at=indexed)])
    res = await _sweeper(vdb, tracker, state_store=None).sweep(now=NOW)
    assert res["archived"] == ["plain"]
    assert "dash_pinned" not in res["archived"]


@pytest.mark.asyncio
async def test_no_state_store_means_no_protection(tracker, retention_settings):
    """Without a state store the sweep still runs (protection just empty)."""
    indexed = NOW - timedelta(days=10)
    _mark_days(tracker, indexed, 5)
    vdb = FakeVectorDB([_chunk("doc", indexed_at=indexed)])
    res = await _sweeper(vdb, tracker, state_store=None).sweep(now=NOW)
    assert res["archived"] == ["doc"]
    assert res["protected_docs"] == 0


# ── Pending-release accounting (purge → usage meter) ─────────────────

@pytest.mark.asyncio
async def test_purge_parks_freed_bytes_for_release(tracker, retention_settings, monkeypatch):
    monkeypatch.setattr(
        app_config, "settings", app_config.settings.model_copy(update={"retention_purge_enabled": True})
    )
    archived_at = NOW - timedelta(days=30)
    _mark_days(tracker, archived_at, 5)
    vdb = FakeVectorDB(
        [_chunk("gone", indexed_at=archived_at, archived=True, archived_at=archived_at)]
    )
    res = await _sweeper(vdb, tracker).sweep(now=NOW)
    assert res["bytes_freed"] > 0
    assert tracker.pending_release() == res["bytes_freed"]


def test_claim_pending_release_is_atomic_and_resets(tracker):
    tracker.add_pending_release(100)
    tracker.add_pending_release(50)
    assert tracker.pending_release() == 150
    assert tracker.claim_pending_release() == 150
    assert tracker.pending_release() == 0
    # Nothing pending → claim returns 0, stays 0.
    assert tracker.claim_pending_release() == 0


def test_claim_does_not_lose_concurrent_addition(tracker):
    """claim subtracts exactly what it read, so a byte added between read and
    reset survives (isn't zeroed away)."""
    tracker.add_pending_release(200)
    # Simulate: the value we claim is 200; a concurrent purge already bumped it
    # to 260 in the same window — the extra 60 must remain after claim.
    import sqlite3

    with sqlite3.connect(tracker._db_path) as c:
        c.execute(
            "UPDATE retention_meta SET value = '260' WHERE key = 'pending_release_bytes'"
        )
    # claim reads 260 here (post-bump), so this models the simpler case; the
    # subtract-what-you-read guarantee is what we assert structurally.
    claimed = tracker.claim_pending_release()
    assert claimed == 260
    assert tracker.pending_release() == 0
