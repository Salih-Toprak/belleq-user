"""Text chunking — mirrors the master ingestion chunker (512/64, char-based).

Kept byte-compatible with belleq-master/app/ingestion/chunker.py so documents
chunked here land the same way master-ingested documents would.
"""

from __future__ import annotations

import hashlib
import uuid

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
_MAX_CHUNK_SOFT = int(CHUNK_SIZE * 1.2)


def chunk_text(text: str) -> list[str]:
    """Split text into ~CHUNK_SIZE-char overlapping chunks on sentence bounds."""
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= CHUNK_SIZE:
        return [t[:_MAX_CHUNK_SOFT]]

    chunks: list[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        window = t[start:end]
        if end < n:
            split_at = window.rfind(". ")
            if split_at >= max(0, CHUNK_SIZE // 4):
                end = start + split_at + 2
                window = t[start:end]
        if len(window) > _MAX_CHUNK_SOFT:
            window = window[:_MAX_CHUNK_SOFT]
        chunks.append(window.strip() or window)
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def make_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID point id (Qdrant requires UUID or uint)."""
    h = hashlib.sha256(f"{doc_id}::{chunk_index}".encode()).digest()
    return str(uuid.UUID(bytes=h[:16]))


def content_hash(text: str) -> str:
    """Stable hash of normalized content for dedup."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()
