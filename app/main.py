"""FastAPI giriş noktası ve uygulama ömrü."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import app.config as app_config
import app.state as state
from app.api.inward.agent_routes import router as internal_agent_router
from app.api.inward.config_routes import router as internal_config_router
from app.api.inward.conversation_routes import router as internal_conversation_router
from app.api.inward.kb_routes import router as internal_kb_router
from app.api.inward.ingestion_routes import router as internal_ingestion_router
from app.api.inward.docs_routes import router as internal_docs_router
from app.api.inward.health_routes import router as internal_health_router
from app.api.outward.query_routes import router as query_router
from app.config import settings
from app.conversation.capture import ConversationCapture
from app.conversation.extraction import ExtractionWorker, build_extractor
from app.conversation.kb_writer import KBWriter
from app.conversation.session_manager import SessionManager
from app.conversation.store import ConversationStore
from app.ingestion.queue import IngestionQueue
from app.ingestion.scheduler import IngestionScheduler
from app.ingestion.worker import IngestionWorker
from app.embeddings.factory import get_embedding_adapter
from app.lifecycle.decay_scheduler import DecayScheduler
from app.lifecycle.retriever import LifecycleRetriever
from app.lifecycle.store import init_stores
from app.mcp.server import build_mcp_server
from app.query.pipeline import QueryPipeline
from app.vectordb.factory import get_vector_db_adapter
from rag_wiki import RagWikiRetrieverConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state._startup_time = time.monotonic()
    logger.info(
        "Starting Belleq Container: user_id=%s type=%s",
        settings.user_id,
        settings.container_type,
    )

    state_store, global_store = init_stores(settings)
    state._state_store = state_store
    state._global_store = global_store
    logger.info("Stores initialized: %s", settings.db_path)

    try:
        vectordb = get_vector_db_adapter(settings)
        health = await vectordb.health()
        state._vectordb = vectordb
        if health.get("status") == "ok":
            logger.info("Vector DB connected: backend=%s", settings.vectordb_backend)
        else:
            logger.warning("Vector DB unhealthy: %s", health)
    except Exception as e:
        logger.warning("Vector DB init failed: %s", e)
        state._vectordb = None

    try:
        embedder = get_embedding_adapter(settings)
        eh = await embedder.health()
        state._embedder = embedder
        logger.info(
            "Embedder ready: backend=%s model=%s",
            settings.embedding_backend,
            eh.get("model"),
        )
    except Exception as e:
        logger.warning("Embedder init failed: %s", e)
        state._embedder = None

    cfg = RagWikiRetrieverConfig(
        fetch_threshold=settings.rag_wiki_fetch_threshold,
    )
    retriever = LifecycleRetriever(
        user_id=settings.user_id,
        vectordb=state._vectordb,
        embedder=state._embedder,
        state_store=state_store,
        config=cfg,
        collection_name=(
            settings.pinecone_index_name.strip()
            if settings.vectordb_backend.strip().lower() == "pinecone"
            and (settings.pinecone_index_name or "").strip()
            else settings.qdrant_collection
        ),
        top_k=settings.rag_wiki_top_k,
    )
    retriever.build()
    state._lifecycle_retriever = retriever

    # ── Conversation capture & archive ───────────────────────────
    conversation_store = None
    conversation_capture = None
    session_manager = None
    if settings.conversation_capture_enabled:
        conversation_store = ConversationStore(settings.conversations_db_path)
        conversation_capture = ConversationCapture(conversation_store, settings)
        state._conversation_store = conversation_store
        state._conversation_capture = conversation_capture
        app.state.conversation_store = conversation_store
        app.state.conversation_capture = conversation_capture
        logger.info("Conversation capture enabled: %s", settings.conversations_db_path)

    pipeline = QueryPipeline(
        user_id=settings.user_id,
        lifecycle_retriever=retriever,
        global_store=global_store,
        settings=settings,
        capture=conversation_capture,
    )
    state._pipeline = pipeline
    app.state.pipeline = pipeline
    app.state.state_store = state_store
    app.state.global_store = global_store
    app.state.lifecycle_retriever = retriever
    app.state.vectordb = state._vectordb
    app.state.embedder = state._embedder

    decay = DecayScheduler(
        user_id=settings.user_id,
        state_store=state_store,
        interval_hours=settings.rag_wiki_decay_interval_hours,
    )
    decay.start()
    app.state.decay_scheduler = decay

    # Shared KB writer (embed → Qdrant → GlobalDocStore). Used by both the
    # conversation extraction worker and the document/MCP ingestion worker.
    kb_collection = (
        settings.pinecone_index_name.strip()
        if settings.vectordb_backend.strip().lower() == "pinecone"
        and (settings.pinecone_index_name or "").strip()
        else settings.qdrant_collection
    )
    kb_writer = KBWriter(
        embedder=state._embedder,
        vectordb=state._vectordb,
        global_store=global_store,
        collection_name=kb_collection,
        settings=settings,
    )
    app.state.kb_writer = kb_writer

    if conversation_capture is not None and conversation_store is not None:
        extractor = build_extractor(settings)
        worker = ExtractionWorker(conversation_store, extractor, kb_writer=kb_writer)
        session_manager = SessionManager(
            user_id=settings.user_id,
            store=conversation_store,
            worker=worker,
            settings=settings,
        )
        session_manager.start()
        app.state.session_manager = session_manager

    # ── Ingestion queue + worker (documents + MCP captures) ──────────
    ingestion_queue = None
    ingestion_scheduler = None
    if settings.ingestion_enabled and kb_writer.available():
        ingestion_queue = IngestionQueue(
            settings.ingestion_db_path, max_attempts=settings.ingestion_max_attempts
        )
        ingestion_worker = IngestionWorker(ingestion_queue, kb_writer)
        ingestion_scheduler = IngestionScheduler(
            ingestion_worker, interval_seconds=settings.ingestion_sweep_interval_seconds
        )
        ingestion_scheduler.start()
        app.state.ingestion_queue = ingestion_queue
        app.state.ingestion_worker = ingestion_worker
        logger.info("Ingestion enabled: %s", settings.ingestion_db_path)

    if settings.mcp_enabled:
        mcp_server = build_mcp_server(
            pipeline, settings, conversation_capture, session_manager, ingestion_queue
        )
        app.mount("/mcp", mcp_server.http_app(transport="sse"))
        logger.info(
            "MCP server mounted at /mcp (SSE) name=%s",
            settings.resolved_mcp_server_name,
        )

    logger.info(
        "Belleq Container ready: user_id=%s port=%d",
        settings.user_id,
        settings.app_port,
    )
    yield

    app.state.decay_scheduler.stop()
    if session_manager is not None:
        session_manager.stop()
    if ingestion_scheduler is not None:
        ingestion_scheduler.stop()
    if conversation_store is not None:
        try:
            conversation_store.close()
        except Exception:
            logger.debug("conversation_store_close_failed", exc_info=True)
    await pipeline.close()
    emb = state._embedder
    if emb is not None and hasattr(emb, "aclose"):
        try:
            await emb.aclose()
        except Exception:
            logger.debug("embedder_aclose_failed", exc_info=True)
    logger.info("Belleq Container shut down: user_id=%s", settings.user_id)


app = FastAPI(
    title="Belleq Container",
    description="""
    Per-user knowledge lifecycle container for the Belleq platform.

    Two API surfaces:
    - `/internal/*` — called by the Belleq master API (dashboard control)
    - `/query` — called by end users via API key or MCP

    Authentication:
    - `/internal/*` requires `X-Master-Key` header
    - `POST /query` requires `X-Api-Key` header
    - Both keys can be left empty for dev mode
    """,
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.error("unhandled_exception path=%s", request.url.path, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def public_health() -> dict:
    s = app_config.settings
    return {
        "status": "ok",
        "user_id": s.user_id,
        "container_type": s.container_type,
        "mcp_enabled": s.mcp_enabled,
    }


app.include_router(internal_health_router)
app.include_router(internal_docs_router)
app.include_router(internal_config_router)
app.include_router(internal_conversation_router)
app.include_router(internal_kb_router)
app.include_router(internal_ingestion_router)
app.include_router(internal_agent_router)
app.include_router(query_router)
