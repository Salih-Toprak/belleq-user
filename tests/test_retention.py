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


def _sweeper(vdb, tracker):
    return RetentionSweeper(
        vectordb=vdb, collection_name="c_test", tracker=tracker, runner=_direct_runner
    )


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
