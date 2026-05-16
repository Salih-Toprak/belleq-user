"""Uygulama ayarları — ortam değişkenleri ve isteğe bağlı runtime_config.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

RUNTIME_CONFIG_KEYS = frozenset(
    {
        "display_name",
        "rag_wiki_fetch_threshold",
        "rag_wiki_top_k",
        "mcp_enabled",
    }
)


class Settings(BaseSettings):
    """Mnemo kullanıcı konteyneri yapılandırması."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Identity ─────────────────────────────────────────────────
    user_id: str = Field(
        ...,
        min_length=1,
        description="Bu konteyner örneği için benzersiz kimlik.",
    )
    display_name: str = ""
    container_type: str = "user"

    # ── Data directory ───────────────────────────────────────────
    data_dir: str = "/app/data"

    # ── Vector DB ────────────────────────────────────────────────
    vectordb_backend: str = "qdrant"
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        validation_alias=AliasChoices("QDRANT_URL", "VECTORDB_URL"),
    )
    qdrant_api_key: str = ""
    qdrant_collection: str = "company_knowledge"

    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = ""
    pinecone_cloud: str = "aws"

    # ── Embeddings ───────────────────────────────────────────────
    embedding_backend: str = "ollama"
    ollama_base_url: str = "http://ollama:11434"
    ollama_embed_model: str = "nomic-embed-text"
    embedding_vector_size: int = 768

    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"

    # ── rag-wiki lifecycle ────────────────────────────────────────
    rag_wiki_fetch_threshold: int = 3
    rag_wiki_decay_interval_hours: int = 24
    rag_wiki_top_k: int = 5

    # ── Auth ─────────────────────────────────────────────────────
    master_api_key: str = ""
    user_api_key: str = ""

    # ── MCP ──────────────────────────────────────────────────────
    mcp_enabled: bool = True
    mcp_server_name: str = ""

    # ── App ──────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    @property
    def db_path(self) -> str:
        return f"{self.data_dir}/{self.user_id}/mnemo.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def runtime_config_path(self) -> str:
        return f"{self.data_dir}/{self.user_id}/runtime_config.json"

    @property
    def resolved_mcp_server_name(self) -> str:
        return self.mcp_server_name or f"mnemo-{self.user_id}"

    @model_validator(mode="after")
    def _normalize_user_id(self) -> Settings:
        uid = (self.user_id or "").strip()
        if not uid:
            raise ValueError(
                "USER_ID zorunludur ve boş olamaz. "
                "Örnek: USER_ID=chatbot veya USER_ID=user-salih",
            )
        if uid != self.user_id:
            return self.model_copy(update={"user_id": uid})
        return self


def _read_runtime_overrides(path: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if k in RUNTIME_CONFIG_KEYS}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("runtime_config okunamadı path=%s", path, exc_info=True)
        return {}


def _merge_runtime(base: Settings) -> Settings:
    overrides = _read_runtime_overrides(base.runtime_config_path)
    if not overrides:
        return base
    return base.model_copy(update=overrides)


def build_settings() -> Settings:
    """Ortam + isteğe bağlı runtime_config.json ile etkin ayarları üretir."""
    base = Settings()
    return _merge_runtime(base)


settings = build_settings()


def replace_settings(new: Settings) -> None:
    """PATCH /internal/config sonrası modül düzeyindeki settings örneğini günceller."""
    global settings
    settings = new
