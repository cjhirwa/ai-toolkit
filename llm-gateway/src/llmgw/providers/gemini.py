from __future__ import annotations

from collections.abc import Iterator

from ..types import Message, Request, Response, ToolCall, Usage
from .base import Provider

_TASK_TYPES = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}
_TOOL_MODES = {"auto": "AUTO", "required": "ANY", "none": "NONE"}


def _usage(um) -> Usage:
    """Thinking tokens are billed as output, so they count toward output_tokens."""
    return Usage(
        input_tokens=(um.prompt_token_count or 0) if um else 0,
        output_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0,
        cached_tokens=(um.cached_content_token_count or 0) if um else 0,
    )


class GeminiProvider(Provider):
    """Google Gemini via ``google-genai``. Reads GOOGLE_API_KEY / GEMINI_API_KEY
    (or the Vertex env vars) when no key is passed."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, client=None, timeout: float | None = None):
        from google import genai
        from google.genai import types as gt

        http_options = gt.HttpOptions(timeout=int(self._timeout(timeout) * 1000))  # milliseconds
        self.client = client or genai.Client(api_key=api_key, http_options=http_options)  # None: read from env

    def _contents(self, messages: list[Message]):
        from google.genai import types as gt

        contents = []
        tool_batch: list = []

        def flush_tools():
            if tool_batch:
                contents.append(gt.Content(role="user", parts=list(tool_batch)))
                tool_batch.clear()

        for m in messages:
            if m.role == "tool":
                tool_batch.append(
                    gt.Part.from_function_response(name=m.name or "tool", response={"result": m.content})
                )
                continue
            flush_tools()
            if m.role == "system":
                continue
            if m.role == "assistant":
                if isinstance(m.raw, gt.Content):
                    contents.append(m.raw)
                    continue
                parts = [gt.Part.from_text(text=m.content)] if m.content else []
                parts += [gt.Part.from_function_call(name=c.name, args=c.arguments) for c in m.tool_calls]
                contents.append(gt.Content(role="model", parts=parts))
            else:
                parts = [gt.Part.from_bytes(data=a.data, mime_type=a.mime) for a in m.attachments]
                if m.content:
                    parts.append(gt.Part.from_text(text=m.content))
                contents.append(gt.Content(role="user", parts=parts))
        flush_tools()
        return contents

    def _config(self, req: Request):
        from google.genai import types as gt

        system = "\n\n".join(s for s in [req.system, *(m.content for m in req.messages if m.role == "system")] if s)
        cfg: dict = {
            # Tool calls are returned to the caller, never executed by the SDK.
            "automatic_function_calling": gt.AutomaticFunctionCallingConfig(disable=True),
            "system_instruction": system or None,
            "temperature": req.temperature,
            "max_output_tokens": req.max_tokens,
        }
        if req.tools:
            cfg["tools"] = [
                gt.Tool(
                    function_declarations=[
                        gt.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                        for t in req.tools
                    ]
                )
            ]
            if req.tool_choice:
                mode = _TOOL_MODES.get(req.tool_choice, "ANY")
                allowed = None if req.tool_choice in _TOOL_MODES else [req.tool_choice]
                cfg["tool_config"] = gt.ToolConfig(
                    function_calling_config=gt.FunctionCallingConfig(mode=gt.FunctionCallingConfigMode(mode), allowed_function_names=allowed)
                )
        if req.json_schema:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = req.json_schema
        cfg.update(req.extra)
        return gt.GenerateContentConfig(**{k: v for k, v in cfg.items() if v is not None})

    def complete(self, req: Request) -> Response:
        resp = self.client.models.generate_content(
            model=req.model, contents=self._contents(req.messages), config=self._config(req)
        )
        cand = resp.candidates[0] if resp.candidates else None
        texts, calls = [], []
        if cand and cand.content and cand.content.parts:
            for i, p in enumerate(cand.content.parts):
                if p.function_call:
                    fc = p.function_call
                    calls.append(ToolCall(id=fc.id or f"call_{i}", name=fc.name, arguments=dict(fc.args or {})))
                elif p.text and not p.thought:
                    texts.append(p.text)
        usage = _usage(resp.usage_metadata)
        finish = str(cand.finish_reason) if cand and cand.finish_reason else None
        if cand is None and resp.prompt_feedback:
            finish = f"blocked: {resp.prompt_feedback.block_reason}"
        return Response(
            text="".join(texts),
            model=req.model,
            tool_calls=calls,
            usage=usage,
            finish_reason=finish,
            raw=cand.content if cand else None,
        )

    def stream(self, req: Request) -> Iterator[str | Usage]:
        last_usage = None
        for chunk in self.client.models.generate_content_stream(
            model=req.model, contents=self._contents(req.messages), config=self._config(req)
        ):
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for p in chunk.candidates[0].content.parts:
                    if p.text and not p.thought:
                        yield p.text
            last_usage = chunk.usage_metadata or last_usage  # cumulative; the last one is the total
        if last_usage:
            yield _usage(last_usage)

    def embed(self, texts: list[str], model: str, task: str | None = None) -> list[list[float]]:
        from google.genai import types as gt

        cfg = gt.EmbedContentConfig(task_type=_TASK_TYPES[task]) if task in _TASK_TYPES else None
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            # One Content per text: multimodal embedding models treat a bare list of
            # strings as parts of a single input and return one combined vector.
            batch = [gt.Content(parts=[gt.Part.from_text(text=t)]) for t in texts[i : i + 100]]
            r = self.client.models.embed_content(model=model, contents=batch, config=cfg)  # type: ignore[arg-type]
            embeddings = r.embeddings or []
            if len(embeddings) != len(batch) or any(e.values is None for e in embeddings):
                raise RuntimeError(f"{model} returned {len(embeddings)} embeddings for {len(batch)} texts")
            out.extend(list(e.values or []) for e in embeddings)
        return out
