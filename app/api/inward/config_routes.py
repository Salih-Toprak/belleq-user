"""İç API: yapılandırma okuma / güncelleme."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import app.config as app_config
from app.api.deps import require_master
from app.config import RUNTIME_CONFIG_KEYS, replace_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/config",
    tags=["Internal — Config"],
    dependencies=[Depends(require_master)],
)


def _redact_value(key: str, value: Any) -> Any:
    lk = key.lower()
    if any(x in lk for x in ("api_key", "secret", "password", "token")):
        if value is None or value == "":
            return ""
        return "***set***"
    return value


def _public_config_dict(s: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "user_id": s.user_id,
        "display_name": s.display_name,
        "container_type": s.container_type,
        "vectordb_backend": s.vectordb_backend,
        "qdrant_url": s.qdrant_url,
        "qdrant_collection": s.qdrant_collection,
        "embedding_backend": s.embedding_backend,
        "ollama_embed_model": s.ollama_embed_model,
        "embedding_vector_size": s.embedding_vector_size,
        "rag_wiki_fetch_threshold": s.rag_wiki_fetch_threshold,
        "rag_wiki_decay_interval_hours": s.rag_wiki_decay_interval_hours,
        "rag_wiki_top_k": s.rag_wiki_top_k,
        "mcp_enabled": s.mcp_enabled,
        "mcp_server_name": s.mcp_server_name or f"mnemo-{s.user_id}",
        "data_dir": s.data_dir,
        "app_port": s.app_port,
    }
    return {k: _redact_value(k, v) for k, v in out.items()}


@router.get("")
async def get_config() -> dict[str, Any]:
    return _public_config_dict(app_config.settings)


class RuntimePatch(BaseModel):
    display_name: str | None = None
    rag_wiki_fetch_threshold: int | None = Field(default=None, ge=1)
    rag_wiki_top_k: int | None = Field(default=None, ge=1)
    mcp_enabled: bool | None = None


@router.patch("")
async def patch_config(body: RuntimePatch, request: Request) -> dict[str, Any]:
    cur = app_config.settings
    updates: dict[str, Any] = {}
    if body.display_name is not None:
        updates["display_name"] = body.display_name
    if body.rag_wiki_fetch_threshold is not None:
        updates["rag_wiki_fetch_threshold"] = body.rag_wiki_fetch_threshold
    if body.rag_wiki_top_k is not None:
        updates["rag_wiki_top_k"] = body.rag_wiki_top_k
    if body.mcp_enabled is not None:
        updates["mcp_enabled"] = body.mcp_enabled
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    new_settings = cur.model_copy(update=updates)
    path = Path(new_settings.runtime_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("runtime_config_corrupt_resetting", exc_info=True)
    merged = {k: v for k, v in existing.items() if k in RUNTIME_CONFIG_KEYS}
    merged.update({k: v for k, v in updates.items() if k in RUNTIME_CONFIG_KEYS})
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    replace_settings(new_settings)
    lr = getattr(request.app.state, "lifecycle_retriever", None)
    if lr is not None:
        lr.reconfigure(new_settings)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        pipeline.bind_settings(new_settings)
    notes = (
        "Yeniden başlatma gerektiren ayarlar: vectordb_backend, qdrant_url, "
        "embedding_backend, ollama_embed_model, app_port. "
        "mcp_enabled değişikliği MCP HTTP montajını tam olarak yansıtmayabilir; "
        "üretimde MCP aç/kapa için konteyneri yeniden başlatın."
    )
    out = _public_config_dict(new_settings)
    out["patch_notes"] = notes
    return out
