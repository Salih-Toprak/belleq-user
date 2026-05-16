# mnemo-user

Per-user **retrieval-as-a-service** container for the [Mnemo](https://github.com/sstprk) platform. It runs the **rag-wiki** lifecycle system and returns relevant document chunks to whoever calls it. It has no LLM, no answer generation, no use-case-specific logic.

Each deployment runs **one** `USER_ID`. All rag-wiki lifecycle state is stored under `/app/data/{USER_ID}/` on disk (SQLite + JSON overrides) and survives restarts.

---

## What this container does

- Accepts a plain text query
- Runs it through rag-wiki (personal cache → master vector DB fallback)
- Returns matching document chunks with metadata
- Tracks retrieval counts and lifecycle state (SURFACED/CLAIMED/PINNED)
- Exposes health and stats to the master API

It does **not** call any LLM or generate answers. The caller decides what to do with the chunks.

---

## Two API surfaces

| Surface | Base path | Auth | Purpose |
|--------|-----------|------|---------|
| **Inward** (master / dashboard) | `/internal/*` | `X-Master-Key` | Health, stats, document CRUD, runtime config |
| **Outward** (users / bots / MCP) | `POST /query`, MCP | `X-Api-Key` on HTTP query (`/query/health` has no auth) | Chunk retrieval |

OpenAPI: **`/docs`**, **`/redoc`**.

---

## Quick start

1. Create a Docker network (once), e.g. from **mnemo-master**: `docker network create mnemo-net`
2. `cp .env.example .env` and set at least **`USER_ID`**
3. `docker compose up --build`
4. Call **`POST /query`** with a JSON body and optional `X-Api-Key` if `USER_API_KEY` is set

Example:

```bash
curl -sS -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-user-key" \
  -d '{"query":"What is our vacation policy?"}'
```

Response:

```json
{
  "chunks": [
    {
      "text": "Our vacation policy allows...",
      "doc_id": "doc-123",
      "doc_title": "HR Policies",
      "source": "notion",
      "channel": "",
      "department": "hr",
      "chunk_index": 0,
      "total_chunks": 3,
      "state": "GLOBAL",
      "metadata": {}
    }
  ],
  "user_id": "user-001",
  "query_id": "uuid-here",
  "latency_ms": 42,
  "provenance": {
    "cache_hits": 1,
    "global_hits": 2,
    "total_retrieved": 3
  }
}
```

---

## Claude Desktop (MCP over SSE)

Set `MCP_ENABLED=true` (default). The MCP app is mounted at **`/mcp`** using FastMCP's **SSE** transport. The SSE entry is typically:

`http://<container-host>:8000/mcp/sse`

Add to **Claude Desktop** config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mnemo": {
      "url": "http://your-container-host:8000/mcp/sse"
    }
  }
}
```

MCP shares the **same** `QueryPipeline` instance as the HTTP API (no duplicate pipeline). The MCP tool returns chunks as JSON.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `USER_ID` | **Yes** | Unique instance id (directory name + rag-wiki scope) |
| `DISPLAY_NAME` | No | Dashboard label |
| `CONTAINER_TYPE` | No | `user` \| `chatbot` \| `agent` (default `user`) |
| `DATA_DIR` | No | Root data dir (default `/app/data`) |
| `VECTORDB_BACKEND` | No | `qdrant` (default) or `pinecone` |
| `QDRANT_URL` / `VECTORDB_URL` | No* | Qdrant HTTP URL (*required in practice for Qdrant) |
| `QDRANT_API_KEY` | No | Qdrant API key if enabled |
| `QDRANT_COLLECTION` | No | Collection / logical index name |
| `PINECONE_*` | If Pinecone | See `.env.example` |
| `EMBEDDING_BACKEND` | No | `ollama` or `openai` |
| `OLLAMA_BASE_URL` | If Ollama | Ollama service URL for embeddings |
| `OLLAMA_EMBED_MODEL` | If Ollama | Embedding model name |
| `OPENAI_API_KEY` | If OpenAI | OpenAI API key for embeddings |
| `OPENAI_EMBED_MODEL` | If OpenAI | OpenAI embedding model |
| `EMBEDDING_VECTOR_SIZE` | No | Vector dimensions (default 768) |
| `RAG_WIKI_*` | No | Fetch threshold, decay interval, top-k |
| `MASTER_API_KEY` | No | Master `X-Master-Key` (empty = open) |
| `USER_API_KEY` | No | User `X-Api-Key` (empty = open) |
| `MCP_ENABLED` | No | Mount MCP sub-app |
| `MCP_SERVER_NAME` | No | Defaults to `mnemo-{USER_ID}` |
| `APP_PORT` | No | Host port in compose mapping (container listens on **8000**) |
| `LOG_LEVEL` | No | Python logging level |

---

## Data persistence

- **Volume**: mount host dir to **`/app/data`**
- **Files**:
  - `{DATA_DIR}/{USER_ID}/mnemo.db` — rag-wiki `SQLiteStateStore` + `GlobalDocStore`
  - `{DATA_DIR}/{USER_ID}/runtime_config.json` — PATCHable runtime overrides

If the volume is **lost**, per-user lifecycle + registry rows are gone; **vectors in Qdrant/Pinecone** remain until explicitly deleted.

---

## Multiple containers (example)

`docker-compose.yml` maps host `${APP_PORT:-8000}` → container `8000`. Run several stacks with different `USER_ID` and host ports:

```yaml
services:
  user001:
    build: .
    env_file: .env.user001
    ports: ["8100:8000"]
    volumes: ["./data-user001:/app/data"]
    networks: [mnemo-net]
  user002:
    build: .
    env_file: .env.user002
    ports: ["8200:8000"]
    volumes: ["./data-user002:/app/data"]
    networks: [mnemo-net]
networks:
  mnemo-net: { external: true, name: mnemo-net }
```

---

## Runtime vs restart configuration

**Adjustable at runtime** (via `PATCH /internal/config` and persisted in `runtime_config.json`):

- `display_name`, `rag_wiki_fetch_threshold`, `rag_wiki_top_k`, `mcp_enabled`

**Requires container restart** (documented in PATCH response `patch_notes`):

- Vector DB backend/URL, embedding backend/model, `app_port` / image CMD port

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export USER_ID=local-dev
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Repository layout

- `app/vectordb/` and `app/embeddings/` — duplicated from **mnemo-master** adapters (see file headers); keep in sync manually.
- `app/lifecycle/` — rag-wiki stores, retriever, APScheduler decay job
- `app/query/` — retrieval pipeline (chunks only, no LLM)
- `app/mcp/` — FastMCP SSE server
- `app/api/` — inward + outward routers

Master repo path (local): `../mnemo-master`.
