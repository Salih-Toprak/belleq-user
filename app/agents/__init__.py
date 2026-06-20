"""Agent execution engine for the belleq-user container.

The backend owns agent/task state (Supabase) and triggers a run through the
master; this package runs the actual agentic loop here, where the context KB,
the LLM SDKs, and (via the master's aggregated MCP endpoint) the connectors are
reachable.
"""
