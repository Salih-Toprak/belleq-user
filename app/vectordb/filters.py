# Shared adapter — keep in sync with belleq-master/app/vectordb/
"""Normalized metadata filters shared across vector DB adapters."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


def _field_conditions(conds: list) -> list[qmodels.FieldCondition]:
    out: list[qmodels.FieldCondition] = []
    for cond in conds:
        if not isinstance(cond, dict):
            continue
        field = cond.get("field")
        value = cond.get("value")
        if field is None:
            continue
        out.append(
            qmodels.FieldCondition(key=str(field), match=qmodels.MatchValue(value=value))
        )
    return out


def build_qdrant_filter(filters: dict | None) -> qmodels.Filter | None:
    """Translate normalized filter dict ({"must": [...], "must_not": [...]})
    to a qdrant_client Filter object."""
    if not filters:
        return None
    must = _field_conditions(filters.get("must") or [])
    must_not = _field_conditions(filters.get("must_not") or [])
    if not must and not must_not:
        return None
    kwargs: dict[str, Any] = {}
    if must:
        kwargs["must"] = must
    if must_not:
        kwargs["must_not"] = must_not
    return qmodels.Filter(**kwargs)


def build_pinecone_filter(filters: dict | None) -> dict | None:
    """Translate normalized filter dict to Pinecone metadata filter dict."""
    if not filters:
        return None
    parts: list[dict[str, Any]] = []
    for cond in filters.get("must") or []:
        if not isinstance(cond, dict):
            continue
        field = cond.get("field")
        value = cond.get("value")
        if field is None:
            continue
        parts.append({str(field): {"$eq": value}})
    for cond in filters.get("must_not") or []:
        if not isinstance(cond, dict):
            continue
        field = cond.get("field")
        value = cond.get("value")
        if field is None:
            continue
        parts.append({str(field): {"$ne": value}})
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
