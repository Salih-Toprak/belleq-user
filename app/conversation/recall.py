"""Recent-facts recall for the zero-instruction `recall_context` MCP tool.

Returns the most recently saved conversation facts so a connected AI can prime
itself at the start of a chat (no LLM, cheap). Kept free of any rag_wiki import
at module load (the persistent-loop helper is imported lazily) so it can be unit
tested in isolation by passing a custom ``runner``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

# A runner marshals a vector-db coroutine to wherever its client lives and
# returns the awaited result. Default (production) sends it to the retriever's
# persistent loop; tests pass one that simply awaits the coroutine.
Runner = Callable[[Awaitable[Any]], Awaitable[Any]]


async def _persistent_runner(coro: Awaitable[Any]) -> Any:
    # Imported lazily so this module stays importable without rag_wiki.
    from app.lifecycle.retriever import _get_persistent_loop

    loop = _get_persistent_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return await asyncio.wrap_future(fut)


async def recent_facts(
    global_store: Any,
    vectordb: Any,
    collection_name: str,
    limit: int = 10,
    runner: Runner | None = None,
) -> dict:
    """Most recently saved conversation facts, newest first, capped at ``limit``."""
    run = runner or _persistent_runner
    limit = max(1, int(limit or 0) or 10)

    def _recent_docs() -> list:
        rows = global_store.list_all(limit=max(limit * 5, 50)) or []
        convs = [r for r in rows if getattr(r, "source", "") == "conversation"]
        convs.sort(
            key=lambda r: (
                getattr(r, "last_updated_at", None)
                or getattr(r, "ingested_at", None)
                or _EPOCH
            ),
            reverse=True,
        )
        return convs

    docs = await asyncio.to_thread(_recent_docs)
    if not docs or vectordb is None:
        return {"facts": [], "count": 0}

    facts: list[dict] = []
    for rec in docs:
        if len(facts) >= limit:
            break
        try:
            chunks = await run(vectordb.get_by_doc_id(collection_name, rec.doc_id))
        except Exception:
            logger.debug(
                "recall_get_by_doc_id_failed doc_id=%s",
                getattr(rec, "doc_id", None),
                exc_info=True,
            )
            continue
        chunks.sort(key=lambda c: (c.get("payload") or {}).get("chunk_index", 0))
        for c in chunks:
            payload = c.get("payload") or {}
            # Retention-archived facts stay out of recall until restored.
            if payload.get("archived") is True:
                continue
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            facts.append({
                "text": text,
                "doc_id": payload.get("doc_id", rec.doc_id),
                "session_id": payload.get("session_id", ""),
                "indexed_at": payload.get("indexed_at", ""),
            })
            if len(facts) >= limit:
                break

    return {"facts": facts[:limit], "count": len(facts[:limit])}
