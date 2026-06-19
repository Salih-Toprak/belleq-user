"""Belge & MCP yanıt yutma hattı (queue + worker + chunk + embed → KB).

Per-container ingestion: documents (uploads) and MCP tool responses are queued
in a local SQLite ``ingestion.db``, then a scheduler-driven worker chunks,
embeds, and writes them into the same KB (Qdrant + GlobalDocStore) the
conversation pipeline uses. Decoupled from the request path so large uploads or
bursty captures never block a query or a tool call.
"""
