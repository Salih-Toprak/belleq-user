"""Ingestion pipeline tests — queue, chunker, extractors, worker (rag_wiki-isolated)."""

from __future__ import annotations

import os
import tempfile

import pytest

from app.ingestion.chunker import chunk_text, content_hash, make_point_id
from app.ingestion.extractors import ExtractionError, detect_type, extract_text
from app.ingestion.models import (
    KIND_DOCUMENT,
    STATUS_DEAD,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
)
from app.ingestion.queue import IngestionQueue
from app.ingestion.worker import IngestionWorker


@pytest.fixture()
def queue():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    q = IngestionQueue(path, max_attempts=3)
    yield q
    os.unlink(path)


# ── chunker ────────────────────────────────────────────────────────────────

def test_chunk_text_short_returns_single():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_text_long_overlaps_and_covers():
    text = ". ".join(f"sentence number {i} here" for i in range(200))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= int(512 * 1.2) for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_point_id_deterministic():
    assert make_point_id("doc-1", 0) == make_point_id("doc-1", 0)
    assert make_point_id("doc-1", 0) != make_point_id("doc-1", 1)


# ── extractors ───────────────────────────────────────────────────────────────

def test_detect_type():
    assert detect_type("a.pdf") == "pdf"
    assert detect_type("a.docx") == "docx"
    assert detect_type("a.md") == "markdown"
    assert detect_type("a.html") == "html"
    assert detect_type("a.unknown") == "text"


def test_extract_text_plain_and_html():
    assert "hello" in extract_text(b"hello there", "note.txt")
    html = b"<html><body><h1>Title</h1><p>Body text</p><script>x()</script></body></html>"
    out = extract_text(html, "page.html")
    assert "Title" in out and "Body text" in out
    assert "x()" not in out


# ── queue ────────────────────────────────────────────────────────────────────

def _payload(text="some text", doc_id="doc-1"):
    return {"text": text, "doc_id": doc_id, "doc_title": "T", "source": "document"}


def test_enqueue_and_dedup(queue):
    chash = content_hash("body")
    jid1, created1 = queue.enqueue(KIND_DOCUMENT, _payload(), chash)
    jid2, created2 = queue.enqueue(KIND_DOCUMENT, _payload(), chash)
    assert created1 is True
    assert created2 is False  # deduped by content hash
    assert jid1 == jid2
    assert queue.stats()["queued"] == 1


def test_claim_complete(queue):
    jid, _ = queue.enqueue(KIND_DOCUMENT, _payload(), content_hash("a"))
    job = queue.claim_next()
    assert job is not None and job.job_id == jid
    # already claimed → nothing else due
    assert queue.claim_next() is None
    queue.complete(jid)
    assert queue.stats()["done"] == 1


def test_fail_retries_then_dead(queue):
    jid, _ = queue.enqueue(KIND_DOCUMENT, _payload(), content_hash("b"))
    queue.claim_next()
    assert queue.fail(jid, "boom") == STATUS_FAILED   # attempt 1
    # backoff sets next_attempt in the future → not immediately claimable
    assert queue.claim_next() is None
    queue.fail(jid, "boom")  # attempt 2
    assert queue.fail(jid, "boom") == STATUS_DEAD      # attempt 3 == max
    assert queue.stats()["dead"] == 1
    assert queue.retry_dead(jid) is True
    assert queue.get(jid).status == STATUS_QUEUED


# ── worker (with a fake KBWriter) ────────────────────────────────────────────

class FakeKBWriter:
    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def write_chunks(self, **kwargs):
        self.calls.append(kwargs)
        return len(kwargs["chunks"])


def test_worker_processes_job(queue):
    writer = FakeKBWriter()
    worker = IngestionWorker(queue, writer)
    queue.enqueue(KIND_DOCUMENT, _payload(text="hello world", doc_id="doc-x"), content_hash("c"))
    out = worker.run_pending()
    assert out == {"processed": 1, "failed": 0}
    assert writer.calls and writer.calls[0]["doc_id"] == "doc-x"
    assert queue.stats()["done"] == 1


def test_worker_fails_empty_text(queue):
    writer = FakeKBWriter()
    worker = IngestionWorker(queue, writer)
    queue.enqueue(KIND_DOCUMENT, {"text": "", "doc_id": "doc-y"}, content_hash("d"))
    out = worker.run_pending()
    assert out["failed"] == 1
    assert not writer.calls
