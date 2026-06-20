"""LLM client router for agent runs (Anthropic / OpenAI / Google).

``call_llm`` takes a provider-agnostic conversation and tool list and returns a
normalized response, so the agentic loop in runner.py never sees provider
specifics. Provider is the platform's Anthropic key for ``provider="belleq"``,
or the agent's own key (detected from the model prefix) for ``provider="byok"``.

Canonical conversation turns (built by runner.py):
    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_uses": [{"id","name","input"}]}
    {"role": "tool", "results": [{"id","name","content": str}]}

Tools: [{"name", "description", "input_schema": <JSON schema dict>}].
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BELLEQ_MODEL = "claude-sonnet-4-6"
# A solid free OpenRouter model with tool-calling, used when an openrouter agent
# has no explicit model set.
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4096

# Best-effort USD pricing per 1M tokens (input, output). Unknown models -> 0.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gemini-2.5-flash": (0.3, 2.5),
    "gemini-2.5-pro": (1.25, 10.0),
}


@dataclass
class LLMResponse:
    final_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {id,name,input}
    stop_reason: str = "end_turn"
    tokens_used: int = 0
    cost_usd: float = 0.0


def detect_provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("gemini-"):
        return "google"
    return "anthropic"  # claude-* and default


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    key = (model or "").lower()
    rate = _PRICING.get(key)
    if rate is None:
        # Try a prefix match (e.g. dated model ids).
        for k, v in _PRICING.items():
            if key.startswith(k):
                rate = v
                break
    if rate is None:
        return 0.0
    return (in_tok / 1_000_000) * rate[0] + (out_tok / 1_000_000) * rate[1]


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def resolve_model_and_key(agent: dict, settings: Any) -> tuple[str, str, str]:
    """Return (provider_family, model, api_key) for this agent.

    BYOK uses the agent's own key (provider detected from the model prefix).
    The managed ``belleq`` provider uses whatever platform LLM credential the
    container actually has — preferring Anthropic, falling back to Gemini (the
    same key the conversation-extraction pipeline uses) — so a managed agent
    works regardless of how the deployment configured its extraction backend.
    """
    provider = agent.get("provider") or "belleq"
    if provider == "openrouter":
        # OpenRouter is OpenAI-compatible; the model id is its own (e.g.
        # "meta-llama/llama-3.3-70b-instruct:free"), routed via base_url override.
        model = agent.get("model") or DEFAULT_OPENROUTER_MODEL
        return "openrouter", model, (agent.get("api_key") or "")
    if provider == "byok":
        model = agent.get("model") or DEFAULT_BELLEQ_MODEL
        return detect_provider(model), model, (agent.get("api_key") or "")

    # belleq (managed): pick the platform key that exists.
    anthropic_key = (getattr(settings, "anthropic_api_key", "") or "").strip()
    # Anthropic keys always start with sk-ant-; reject anything else (stale env
    # vars, placeholder strings, or keys for other providers injected by mistake).
    if anthropic_key and anthropic_key.startswith("sk-ant-"):
        model = agent.get("model") or DEFAULT_BELLEQ_MODEL
        family = detect_provider(model)
        if family != "anthropic":  # platform Anthropic key only serves claude-*
            model, family = DEFAULT_BELLEQ_MODEL, "anthropic"
        return family, model, anthropic_key

    gemini_key = (getattr(settings, "gemini_api_key", "") or "").strip()
    if gemini_key:
        model = (getattr(settings, "gemini_model", "") or DEFAULT_GEMINI_MODEL).strip()
        return "google", model, gemini_key

    raise RuntimeError(
        "No platform LLM key is configured for the managed 'belleq' provider on "
        "this context. Set the platform Anthropic or Gemini key, or switch the "
        "agent to BYOK (Bring your own key) in its settings."
    )


def call_llm(agent: dict, system_prompt: str, conv: list[dict], tools: list[dict], settings: Any) -> LLMResponse:
    """Dispatch one LLM turn. Synchronous (called via asyncio.to_thread)."""
    family, model, api_key = resolve_model_and_key(agent, settings)
    if family == "openai":
        return _call_openai(model, api_key, system_prompt, conv, tools)
    if family == "openrouter":
        return _call_openai(model, api_key, system_prompt, conv, tools, base_url=OPENROUTER_BASE_URL)
    if family == "google":
        return _call_google(model, api_key, system_prompt, conv, tools)
    return _call_anthropic(model, api_key, system_prompt, conv, tools)


# ── Anthropic ────────────────────────────────────────────────────────────────
def _anthropic_messages(conv: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for turn in conv:
        role = turn["role"]
        if role == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": turn.get("text", "")}]})
        elif role == "assistant":
            blocks: list[dict] = []
            if turn.get("text"):
                blocks.append({"type": "text", "text": turn["text"]})
            for tu in turn.get("tool_uses", []):
                blocks.append(
                    {"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu.get("input", {})}
                )
            msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            content = [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r.get("content", "")}
                for r in turn.get("results", [])
            ]
            msgs.append({"role": "user", "content": content})
    return msgs


def _call_anthropic(model: str, api_key: str, system: str, conv: list[dict], tools: list[dict]) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    tool_specs = [
        {"name": t["name"], "description": t.get("description", ""), "input_schema": t["input_schema"]}
        for t in tools
    ]
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=_anthropic_messages(conv),
        tools=tool_specs or anthropic.NOT_GIVEN,
    )
    text_parts, tool_calls = [], []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": dict(block.input or {})})
    in_tok = getattr(resp.usage, "input_tokens", 0) or 0
    out_tok = getattr(resp.usage, "output_tokens", 0) or 0
    stop = "tool_use" if tool_calls else (resp.stop_reason or "end_turn")
    return LLMResponse(
        final_text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=stop,
        tokens_used=in_tok + out_tok,
        cost_usd=_cost(model, in_tok, out_tok),
    )


# ── OpenAI ───────────────────────────────────────────────────────────────────
def _openai_messages(system: str, conv: list[dict]) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system}]
    for turn in conv:
        role = turn["role"]
        if role == "user":
            msgs.append({"role": "user", "content": turn.get("text", "")})
        elif role == "assistant":
            m: dict = {"role": "assistant", "content": turn.get("text", "") or None}
            if turn.get("tool_uses"):
                m["tool_calls"] = [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {"name": tu["name"], "arguments": json.dumps(tu.get("input", {}))},
                    }
                    for tu in turn["tool_uses"]
                ]
            msgs.append(m)
        elif role == "tool":
            for r in turn.get("results", []):
                msgs.append({"role": "tool", "tool_call_id": r["id"], "content": r.get("content", "")})
    return msgs


def _call_openai(
    model: str,
    api_key: str,
    system: str,
    conv: list[dict],
    tools: list[dict],
    base_url: str | None = None,
) -> LLMResponse:
    from openai import OpenAI

    # ``base_url`` switches the same OpenAI-compatible code path to OpenRouter
    # (or any compatible gateway); None keeps the default OpenAI endpoint.
    if api_key or base_url:
        client = OpenAI(api_key=api_key or "", base_url=base_url)
    else:
        client = OpenAI()
    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]
    resp = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=_openai_messages(system, conv),
        tools=tool_specs or None,
    )
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    stop = "tool_use" if tool_calls else "end_turn"
    return LLMResponse(
        final_text=msg.content or "",
        tool_calls=tool_calls,
        stop_reason=stop,
        tokens_used=in_tok + out_tok,
        cost_usd=_cost(model, in_tok, out_tok),
    )


# ── Google (Gemini) ──────────────────────────────────────────────────────────
_GEMINI_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


def _gemini_schema(node: Any) -> dict | None:
    """Coerce a JSON-Schema node into a Gemini-safe schema.

    Gemini's FunctionDeclaration is much stricter than JSON Schema (and than
    Anthropic/OpenAI): ``type`` must be a single supported value (not a list /
    union), ``enum`` must be all-strings, and unknown keywords are rejected. MCP
    connector tools routinely emit list-typed fields and boolean enums, so we
    rebuild a clean schema with only the keys Gemini accepts and drop what it
    can't represent. Returns None for an un-coercible node.
    """
    if not isinstance(node, dict):
        return None
    out: dict = {}

    t = node.get("type")
    if isinstance(t, list):  # union type -> first supported, non-null wins
        picks = [x for x in t if x in _GEMINI_TYPES and x != "null"] or [x for x in t if x in _GEMINI_TYPES]
        t = picks[0] if picks else None
    if isinstance(t, str) and t in _GEMINI_TYPES:
        out["type"] = t

    if isinstance(node.get("description"), str):
        out["description"] = node["description"]

    enum = node.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(e, str) for e in enum):
        out["enum"] = enum
        out.setdefault("type", "string")

    props = node.get("properties")
    if isinstance(props, dict):
        cleaned = {k: _gemini_schema(v) for k, v in props.items()}
        cleaned = {k: v for k, v in cleaned.items() if v}
        if cleaned:
            out["properties"] = cleaned
            out.setdefault("type", "object")
            req = node.get("required")
            if isinstance(req, list):
                kept = [r for r in req if r in cleaned]
                if kept:
                    out["required"] = kept

    if out.get("type") == "array" or "items" in node:
        ci = _gemini_schema(node.get("items")) if isinstance(node.get("items"), dict) else None
        out["items"] = ci or {"type": "string"}
        out.setdefault("type", "array")

    if "type" not in out:
        out["type"] = "object" if "properties" in out else "string"
    return out


def _gemini_parameters(input_schema: Any) -> dict | None:
    """Top-level parameters for a FunctionDeclaration, or None for no-arg tools."""
    schema = _gemini_schema(input_schema)
    if not schema or schema.get("type") != "object" or not schema.get("properties"):
        # Gemini rejects an object schema with no properties; omit parameters.
        return None
    return schema


def _call_google(model: str, api_key: str, system: str, conv: list[dict], tools: list[dict]) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    contents = []
    for turn in conv:
        role = turn["role"]
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=turn.get("text", ""))]))
        elif role == "assistant":
            parts = []
            if turn.get("text"):
                parts.append(types.Part(text=turn["text"]))
            for tu in turn.get("tool_uses", []):
                parts.append(types.Part(function_call=types.FunctionCall(name=tu["name"], args=tu.get("input", {}))))
            contents.append(types.Content(role="model", parts=parts or [types.Part(text="")]))
        elif role == "tool":
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        name=r.get("name", "tool"), response={"result": r.get("content", "")}
                    )
                )
                for r in turn.get("results", [])
            ]
            contents.append(types.Content(role="user", parts=parts))

    fn_decls = []
    for t in tools:
        try:
            fn_decls.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=_gemini_parameters(t.get("input_schema")),
                )
            )
        except Exception:  # noqa: BLE001 — skip a tool Gemini can't represent rather than failing the run
            logger.warning("gemini_tool_schema_skipped tool=%s", t.get("name"), exc_info=True)
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=fn_decls)] if fn_decls else None,
        max_output_tokens=MAX_TOKENS,
    )
    resp = client.models.generate_content(model=model, contents=contents, config=config)

    text_parts, tool_calls = [], []
    cand = (resp.candidates or [None])[0]
    if cand and cand.content and cand.content.parts:
        for part in cand.content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append(
                    {"id": f"call_{uuid.uuid4().hex[:12]}", "name": fc.name, "input": dict(fc.args or {})}
                )
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    stop = "tool_use" if tool_calls else "end_turn"
    return LLMResponse(
        final_text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=stop,
        tokens_used=in_tok + out_tok,
        cost_usd=_cost(model, in_tok, out_tok),
    )
