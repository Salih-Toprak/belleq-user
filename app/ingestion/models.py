"""Ingestion queue job model + status constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Job kinds.
KIND_DOCUMENT = "document"      # an uploaded file's extracted text
KIND_MCP_CAPTURE = "mcp_capture"  # a document-like MCP tool response

# Job statuses.
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"   # transient failure, will retry until max_attempts
STATUS_DEAD = "dead"       # exhausted retries (dead-letter)


@dataclass
class IngestionJob:
    job_id: str
    kind: str
    status: str
    content_hash: str
    payload: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 5
    last_error: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    next_attempt_at: datetime | None = None
