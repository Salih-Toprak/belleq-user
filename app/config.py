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
    """Belleq kullanıcı konteyneri yapılandırması."""

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

    # ── Conversation capture & archive ───────────────────────────
    # The conversation history subsystem records user/assistant exchanges
    # (via the record_exchange MCP tool) and passive query traffic, groups
    # them into sessions, detects session end by idle gap, and routes closed
    # sessions to fact extraction (>= min_exchanges) or skip (< min_exchanges).
    conversation_capture_enabled: bool = True
    conversation_min_exchanges: int = 10
    conversation_session_idle_minutes: int = 30
    conversation_sweep_interval_minutes: int = 10
    # How many recent saved facts `recall_context` returns by default (the
    # zero-instruction "load my situation" primer the connected AI calls first).
    recall_default_limit: int = 10
    # Fact extraction (LLM -> chunk -> embed -> KB). While disabled the worker
    # only classifies/marks sessions and writes nothing.
    conversation_extraction_enabled: bool = False
    # Provider for the extraction LLM: "gemini" (Gemini Flash) or "anthropic"
    # (Claude Haiku). NOTE: the *free* Gemini/AI-Studio tier may train on the
    # data you send — link billing (paid tier) before routing customer
    # conversations through it. Claude never trains on API data.
    extraction_backend: str = "gemini"
    # Gemini (google-genai) — reads GEMINI_API_KEY / GOOGLE_API_KEY from env if blank.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    # Anthropic (claude) — reads ANTHROPIC_API_KEY from env if blank.
    anthropic_api_key: str = ""
    extraction_model: str = "claude-haiku-4-5"

    # ── Agent execution ──────────────────────────────────────────
    # Max agentic loop iterations — a runaway-loop safety net, not a feature cap.
    # High by default so large tasks finish; daily budget bounds cost separately.
    agent_max_steps: int = 100

    # ── Agent web access (Tavily) ────────────────────────────────
    # Platform-wide search/fetch key injected at provision time (like the LLM
    # keys above). When blank, agents simply have no web tools — everything else
    # keeps working. See app/agents/tools.py (web_search / web_fetch).
    tavily_api_key: str = ""

    # ── Ingestion (documents + MCP captures → queue → worker → KB) ──
    ingestion_enabled: bool = True
    ingestion_sweep_interval_seconds: int = 20
    ingestion_max_attempts: int = 5
    # Max accepted upload size (decoded bytes) before extraction.
    ingestion_max_upload_mb: int = 25

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
        return f"{self.data_dir}/{self.user_id}/belleq.db"

    @property
    def conversations_db_path(self) -> str:
        """Separate SQLite file for conversation capture (rag-wiki owns belleq.db)."""
        return f"{self.data_dir}/{self.user_id}/conversations.db"

    @property
    def ingestion_db_path(self) -> str:
        """Separate SQLite file for the ingestion queue."""
        return f"{self.data_dir}/{self.user_id}/ingestion.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def runtime_config_path(self) -> str:
        return f"{self.data_dir}/{self.user_id}/runtime_config.json"

    @property
    def resolved_mcp_server_name(self) -> str:
        return self.mcp_server_name or f"belleq-{self.user_id}"

    @model_validator(mode="after")
    def _normalize_user_id(self) -> Settings:
        uid = (self.user_id or "").strip()
        if not uid:
            raise ValueError(
                "USER_ID zorunludur ve boş olamaz. "
                "Örnek: USER_ID=chatbot veya USER_ID=user-salih",
            )

        self.user_id = uid

        # URL envs are copied verbatim from the master/backend; an empty or
        # scheme-less value (e.g. a bare "host:11434") otherwise reaches the
        # httpx client only at query time as "URL is missing an 'http://' …".
        # Heal it here: blank → field default, scheme-less → prefix http://.
        for field in ("ollama_base_url", "qdrant_url"):
            setattr(
                self,
                field,
                _normalize_url(
                    getattr(self, field),
                    default=self.model_fields[field].default,
                ),
            )

        return self


def _normalize_url(value: str, *, default: str) -> str:
    """Make a service URL usable: blank → default, scheme-less → http://."""
    v = (value or "").strip()
    if not v:
        return default
    if "://" not in v:
        return f"http://{v}"
    return v


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
