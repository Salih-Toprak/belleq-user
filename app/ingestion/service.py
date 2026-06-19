"""Enqueue helpers: turn an upload or an MCP capture into a queued job.

Extraction + dedup hashing happen here (at enqueue time) so callers get
immediate feedback on a bad file, and the worker only does the heavy lifting.
"""

from __future__ import annotations

import logging
from typing import Any

from app.ingestion.chunker import content_hash
from app.ingestion.extractors import extract_text
from app.ingestion.models import KIND_DOCUMENT, KIND_MCP_CAPTURE
from app.ingestion.queue import IngestionQueue

logger = logging.getLogger(__name__)


def enqueue_document(
    queue: IngestionQueue,
    *,
    raw: bytes,
    filename: str,
    content_type: str = "",
    title: str = "",
) -> dict[str, Any]:
    """Extract text from an uploaded file and queue it. Raises ExtractionError."""
    text = extract_text(raw, filename, content_type).strip()
    if not text:
        from app.ingestion.extractors import ExtractionError
        raise ExtractionError(f"No text extracted from {filename}")

    chash = content_hash(text)
    doc_id = f"doc-{chash[:16]}"
    display = (title or filename or doc_id).strip()
    job_id, created = queue.enqueue(
        KIND_DOCUMENT,
        {
            "text": text,
            "doc_id": doc_id,
            "doc_title": display,
            "doc_path": f"upload:{filename or doc_id}",
            "source": "document",
            "extra": {"filename": filename},
        },
        chash,
    )
    return {
        "job_id": job_id,
        "doc_id": doc_id,
        "queued": created,
        "duplicate": not created,
        "chars": len(text),
        "title": display,
    }


def enqueue_capture(
    queue: IngestionQueue,
    *,
    text: str,
    title: str,
    source_label: str,
    doc_path: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue a document-like MCP tool response (4C)."""
    body = (text or "").strip()
    if not body:
        raise ValueError("capture has no text")
    chash = content_hash(body)
    doc_id = f"cap-{chash[:16]}"
    job_id, created = queue.enqueue(
        KIND_MCP_CAPTURE,
        {
            "text": body,
            "doc_id": doc_id,
            "doc_title": (title or doc_id).strip(),
            "doc_path": doc_path or f"capture:{source_label}",
            "source": "mcp_capture",
            "extra": {"connector": source_label, **(extra or {})},
        },
        chash,
    )
    return {"job_id": job_id, "doc_id": doc_id, "queued": created, "duplicate": not created}
