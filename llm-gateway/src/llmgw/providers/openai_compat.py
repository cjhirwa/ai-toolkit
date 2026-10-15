"""OpenAI Chat Completions, plus OpenAI-compatible servers (Ollama)."""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Iterator

from ..types import Request, Response, ToolCall, UnsupportedFeature, Usage
from .base import Provider

# Reasoning models reject a custom temperature.
_NO_TEMPERATURE = ("o1", "o3", "o4", "gpt-5")
_warned_temperature: set[str] = set()
log = logging.getLogger("llmgw")


def _usage(u) -> Usage:
    details = getattr(u, "prompt_tokens_details", None) if u else None
    return Usage(
        input_tokens=u.prompt_tokens if u else 0,
        output_tokens=u.completion_tokens if u else 0,
        cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
    )


def _strict_schema(schema: dict) -> dict | None:
    """A copy of ``schema`` usable with OpenAI's strict mode, or None when it cannot be.

    Strict mode guarantees the reply matches the schema, but only accepts schemas where
    every object lists all its properties as required and forbids extra keys. Pydantic
    models with optional fields do not meet that, and sending them strict is a 400, so
    those keep the non-strict format (validated and retried by llmgw)."""
    import copy

    out = copy.deepcopy(schema)

    def visit(node) -> bool:
        if isinstance(node, list):
            return all(visit(n) for n in node)
        if not isinstance(node, dict):
            return True
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties")
            if not props or set(node.get("required", [])) != set(props):
                return False
            if node.get("additionalProperties") not in (None, False):
                return False
            node["additionalProperties"] = False
        return all(visit(v) for k, v in node.items() if k not in ("required", "enum", "const", "default", "examples"))

    return out if visit(out) else None


class OpenAIProvider(Provider):
    name = "openai"
    max_tokens_param = "max_completion_tokens"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, client=None, timeout: float | None = None):
        from openai import OpenAI

        # max_retries=0: llmgw retries itself; the SDK's own retries would multiply ours.
        self.client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=self._timeout(timeout), max_retries=0)

    def _messages(self, req: Request) -> list[dict]:
        out: list[dict] = []
        if req.system:
            out.append({"role": "system", "content": req.system})
        for m in req.messages:
            if m.role == "system":
                out.append({"role": "system", "content": m.content})
            elif m.role == "user":
                if not m.attachments:
                    out.append({"role": "user", "content": m.content})
                    continue
                parts: list[dict] = [{"type": "text", "text": m.content}] if m.content else []
                for a in m.attachments:
                    if not a.mime.startswith("image/"):
                        raise UnsupportedFeature(f"{self.name} chat input does not accept {a.mime}")
                    url = f"data:{a.mime};base64,{base64.b64encode(a.data).decode()}"
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                out.append({"role": "user", "content": parts})
            elif m.role == "assistant":
                msg: dict = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ]
                out.append(msg)
            else:
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        return out

    def _params(self, req: Request) -> dict:
        p: dict = {"model": req.model, "messages": self._messages(req)}
        if req.temperature is not None and not req.model.startswith(_NO_TEMPERATURE):
            p["temperature"] = req.temperature
        elif req.temperature is not None and req.model not in _warned_temperature:
            # These models reject the parameter; say so rather than silently ignoring it.
            _warned_temperature.add(req.model)
            log.warning("%s does not accept temperature; ignoring temperature=%s (results are not deterministic)",
                        req.model, req.temperature)
        if req.max_tokens:
            p[self.max_tokens_param] = req.max_tokens
        if req.tools:
            p["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in req.tools
            ]
            if req.tool_choice in ("auto", "required", "none"):
                p["tool_choice"] = req.tool_choice
            elif req.tool_choice:
                p["tool_choice"] = {"type": "function", "function": {"name": req.tool_choice}}
        if req.json_schema:
            strict = _strict_schema(req.json_schema)
            p["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": req.schema_name, "schema": strict or req.json_schema,
                                **({"strict": True} if strict else {})},
            }
        p.update(req.extra)
        return p

    def complete(self, req: Request) -> Response:
        resp = self.client.chat.completions.create(**self._params(req))
        choice = resp.choices[0]
        calls = []
        for c in choice.message.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c.function.arguments}
            calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
        usage = _usage(resp.usage)
        return Response(
            text=choice.message.content or "",
            model=req.model,
            tool_calls=calls,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    def stream(self, req: Request) -> Iterator[str | Usage]:
        # include_usage makes the server send token counts in a final chunk without choices.
        for chunk in self.client.chat.completions.create(**self._params(req), stream=True,
                                                         stream_options={"include_usage": True}):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):
                yield _usage(chunk.usage)

    def embed(self, texts: list[str], model: str, task: str | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            r = self.client.embeddings.create(model=model, input=texts[i : i + 256])
            out.extend(d.embedding for d in r.data)
        return out


class OllamaProvider(OpenAIProvider):
    name = "ollama"
    max_tokens_param = "max_tokens"
    default_timeout = 600.0  # local models on CPU can take minutes for a long answer

    def __init__(self, base_url: str | None = None, client=None, timeout: float | None = None):
        super().__init__(
            api_key="ollama",
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            client=client,
            timeout=timeout,
        )

    # Embedding models trained with task prefixes; without them retrieval quality drops.
    _TASK_PREFIXES = {
        "nomic-embed": {"query": "search_query: ", "document": "search_document: "},
        "mxbai-embed": {"query": "Represent this sentence for searching relevant passages: ", "document": ""},
    }

    def embed(self, texts: list[str], model: str, task: str | None = None) -> list[list[float]]:
        prefixes = next((p for k, p in self._TASK_PREFIXES.items() if model.startswith(k)), None)
        if prefixes and task in prefixes:
            texts = [prefixes[task] + t for t in texts]
        return super().embed(texts, model, task)
