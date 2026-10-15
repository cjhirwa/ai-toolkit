from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from . import pricing, telemetry
from .budget import BudgetExceeded, daily_limit, daily_spend
from .limits import limit_for
from .providers.base import Provider
from .storage import CallLogger, EmbeddingCache, ResponseCache, request_key
from .tools import Tool, as_tool
from .types import Attachment, Message, Request, Response, StructuredOutputError, ToolSpec, Usage, normalize_finish

log = logging.getLogger("llmgw")
T = TypeVar("T", bound=BaseModel)

_dotenv_loaded = False


_EMBED_CACHES: dict[str, EmbeddingCache] = {}
_embed_cache_lock = threading.Lock()


def _embed_cache(setting: bool | str | Path | None) -> EmbeddingCache | None:
    """None follows LLMGW_EMBED_CACHE (a path, default .llmgw/embed-cache.sqlite, or 0 to
    turn it off); True uses that path; False turns it off; a path uses that file."""
    default = ".llmgw/embed-cache.sqlite"
    if setting is False:
        return None
    if setting is None or setting is True:
        env = os.getenv("LLMGW_EMBED_CACHE", default).strip()
        disabled = env.lower() in ("0", "off", "false", "")
        if disabled and setting is None:
            return None
        setting = default if disabled else env
    path = str(Path(setting).resolve())
    with _embed_cache_lock:
        if path not in _EMBED_CACHES:
            _EMBED_CACHES[path] = EmbeddingCache(path)
        return _EMBED_CACHES[path]


def find_env_file(start: Path | None = None) -> Path | None:
    """The ``.env`` to load: ``LLMGW_DOTENV`` if set (``0`` means none), otherwise the
    nearest ``.env`` from ``start`` upward, stopping at the project root (a folder with .git,
    or a pyproject.toml that is not a member of a uv workspace). Searching further would let a library pick up an
    unrelated project's settings from a parent folder."""
    explicit = os.getenv("LLMGW_DOTENV", "").strip()
    if explicit:
        return None if explicit.lower() in ("0", "off", "false") else Path(explicit)
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / ".env").is_file():
            return folder / ".env"
        if (folder / ".git").exists():
            return None
        # A package inside a uv workspace (like this toolkit's own packages) is not the
        # project root; keep going up to the workspace's pyproject.toml.
        if (folder / "pyproject.toml").is_file() and not _inside_workspace(folder):
            return None
    return None


def _inside_workspace(folder: Path) -> bool:
    for parent in folder.parents:
        p = parent / "pyproject.toml"
        if p.is_file() and "[tool.uv.workspace]" in p.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def _load_dotenv_once() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        from dotenv import load_dotenv

        if path := find_env_file():
            load_dotenv(path)
        _dotenv_loaded = True


def load_env() -> None:
    """Load the project's ``.env`` once (see :func:`find_env_file`: the nearest one up to the
    project root, or ``LLMGW_DOTENV``). Variables already set in the environment win.
    ``LLM()`` does this itself; CLIs that read settings before creating an ``LLM`` call it first."""
    _load_dotenv_once()


def default_model() -> str:
    return os.getenv("LLMGW_DEFAULT_MODEL", "gemini:gemini-3.6-flash")


def default_embed_model() -> str:
    return os.getenv("LLMGW_EMBED_MODEL", "gemini:gemini-embedding-2-preview")


# ── provider registry ────────────────────────────────────────────────

def _gemini():
    from .providers.gemini import GeminiProvider
    return GeminiProvider()


def _openai():
    from .providers.openai_compat import OpenAIProvider
    return OpenAIProvider()


def _ollama():
    from .providers.openai_compat import OllamaProvider
    return OllamaProvider()


def _fake():
    from .providers.fake import FakeProvider
    return FakeProvider()


_FACTORIES: dict[str, Callable[[], Provider]] = {
    "gemini": _gemini,
    "google": _gemini,
    "openai": _openai,
    "ollama": _ollama,
    "fake": _fake,
}
_INSTANCES: dict[str, Provider] = {}
_registry_lock = threading.Lock()


def register_provider(name: str, provider: Provider | Callable[[], Provider]) -> None:
    """Add or replace a provider, e.g. ``register_provider("fake", FakeProvider([...]))``."""
    with _registry_lock:
        if isinstance(provider, Provider):
            _INSTANCES[name] = provider
        else:
            _FACTORIES[name] = provider
            _INSTANCES.pop(name, None)


def get_provider(name: str) -> Provider:
    with _registry_lock:
        if name not in _INSTANCES:
            if name not in _FACTORIES:
                raise ValueError(f"Unknown provider {name!r}. Known: {sorted(_FACTORIES)}")
            _INSTANCES[name] = _FACTORIES[name]()
        return _INSTANCES[name]


def parse_model(spec: str) -> tuple[str, str]:
    """``"openai:gpt-4o-mini"`` -> ("openai", "gpt-4o-mini"). The prefix may be
    omitted for well-known model families."""
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider, model
    s = spec.lower()
    if s.startswith(("gemini", "gemma", "models/")):
        return "gemini", spec
    if s.startswith(("gpt", "o1", "o3", "o4", "text-embedding", "chatgpt")):
        return "openai", spec
    raise ValueError(f"Cannot infer provider for {spec!r}; use 'provider:model'.")


# ── helpers ──────────────────────────────────────────────────────────

_LEADING_STATUS = re.compile(r"^\s*(\d{3})\b")


def _status(e: BaseException) -> int | None:
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int):
        return status
    m = _LEADING_STATUS.match(str(e))  # e.g. google-genai's "503 UNAVAILABLE. {...}"
    return int(m.group(1)) if m else None


def is_retryable(e: BaseException) -> bool:
    """Transient failures worth retrying (and falling back on): throttling, server errors,
    timeouts and dropped connections. Decided by status code or exception type, never by
    words in the message, so a 400 that mentions "timeout" is not retried."""
    status = _status(e)
    if status is not None:
        # 501 Not Implemented / 505 are permanent; other 5xx and throttling are transient.
        return status in (408, 409, 429) or (status >= 500 and status not in (501, 505))
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    return any(("Timeout" in c.__name__ or "Connect" in c.__name__) for c in type(e).__mro__)


def retry_after(e: BaseException) -> float | None:
    """Seconds the provider asked us to wait (``Retry-After`` header), if any."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return min(float(value), 120.0) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _to_messages(prompt: Any, attachments: list[Attachment] | None = None) -> list[Message]:
    if isinstance(prompt, str):
        return [Message.user(prompt, attachments)]
    msgs = []
    for m in prompt:
        if isinstance(m, Message):
            msgs.append(m)
        elif isinstance(m, dict):
            msgs.append(Message(m["role"], m.get("content") or ""))
        else:
            raise TypeError(f"Cannot convert {type(m).__name__} to Message")
    if attachments:
        raise ValueError("Pass attachments on the Message when giving a message list.")
    return msgs


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _clean_json(text: str) -> str:
    text = _FENCE.sub("", text.strip())
    if not text.startswith(("{", "[")):
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start >= 0:
            text = text[start:]
    return text


_UNSET = object()


# ── client ───────────────────────────────────────────────────────────

class LLM:
    """A model handle with defaults, retries, fallbacks, logging, caching and usage totals.

        llm = LLM("gemini:gemini-3.6-flash", temperature=0.2)
        llm.text("Say hi")
        llm.structured("Extract...", MySchema)
        llm.embed(["a", "b"])
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        fallbacks: Iterable[str] = (),
        embed_model: str | None = None,
        max_retries: int = 3,
        log_dir: str | Path | None | bool = None,
        cache: bool | str | Path = False,
        provider: Provider | None = None,
        log_content: bool | None = None,
        embed_cache: bool | str | Path | None = None,
        max_cost_usd: float | None = None,
    ):
        _load_dotenv_once()
        self.model = model or default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system
        self.fallbacks = list(fallbacks)
        self.embed_model = embed_model or default_embed_model()
        self.max_retries = max_retries
        self.max_cost_usd = max_cost_usd
        self._provider_override = provider
        if log_dir is False or os.getenv("LLMGW_LOG", "1") == "0":
            self.logger = None
        else:
            path = log_dir if isinstance(log_dir, (str, Path)) else os.getenv("LLMGW_LOG_DIR", ".llmgw/logs")
            if log_content is None:
                # Off unless asked for: prompts and documents may hold personal or confidential data.
                log_content = os.getenv("LLMGW_LOG_CONTENT", "0") == "1"
            self.logger = CallLogger(path, content=log_content)
        if cache:
            path = cache if isinstance(cache, (str, Path)) else os.getenv("LLMGW_CACHE", ".llmgw/cache.sqlite")
            self.cache: ResponseCache | None = ResponseCache(path)
        else:
            self.cache = None
        self.embed_cache = embed_cache
        self.usage = Usage()
        self.calls = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"LLM({self.model!r})"

    def with_options(self, **changes: Any) -> LLM:
        """Copy with some defaults changed (shares logger and cache, fresh usage totals)."""
        clone = copy.copy(self)
        for k, v in changes.items():
            if not hasattr(clone, k):
                raise AttributeError(k)
            setattr(clone, k, v)
        clone.usage, clone.calls, clone._lock = Usage(), 0, threading.Lock()
        return clone

    def _provider(self, name: str) -> Provider:
        return self._provider_override or get_provider(name)

    def _check_budget(self) -> None:
        """Refuse the call if this client's or today's spending cap is reached (see llmgw.budget)."""
        spent = self.usage.cost_usd or 0.0
        if self.max_cost_usd is not None and spent >= self.max_cost_usd:
            raise BudgetExceeded(f"this LLM has spent ${spent:.4f}, its max_cost_usd is ${self.max_cost_usd}")
        limit = daily_limit()
        if limit is None:
            return
        if self.logger is None:
            raise BudgetExceeded("LLMGW_DAILY_BUDGET_USD is set but call logging is off, so today's spend is unknown")
        today = daily_spend(self.logger.dir).total()
        if today >= limit:
            raise BudgetExceeded(f"${today:.4f} spent today (UTC) in {self.logger.dir}; LLMGW_DAILY_BUDGET_USD is ${limit}")

    def _record(self, usage: Usage) -> None:
        with self._lock:
            self.usage = self.usage + usage
            self.calls += 1

    def _build(self, spec: str, messages: list[Message], **kw: Any) -> tuple[str, Request]:
        provider_name, model = parse_model(spec)
        tools = kw.pop("tools", None)
        specs = [t if isinstance(t, ToolSpec) else as_tool(t).spec for t in tools] if tools else None
        temperature = kw.pop("temperature", _UNSET)
        max_tokens = kw.pop("max_tokens", _UNSET)
        system = kw.pop("system", _UNSET)
        req = Request(
            model=model,
            messages=messages,
            system=self.system if system is _UNSET else system,
            temperature=self.temperature if temperature is _UNSET else temperature,
            max_tokens=self.max_tokens if max_tokens is _UNSET else max_tokens,
            tools=specs,
            tool_choice=kw.pop("tool_choice", None),
            json_schema=kw.pop("json_schema", None),
            schema_name=kw.pop("schema_name", "response"),
            extra=kw,
        )
        return provider_name, req

    def complete(
        self,
        prompt: str | list[Message] | list[dict],
        *,
        attachments: list[Attachment] | None = None,
        tools: list[Tool | Callable | ToolSpec] | None = None,
        **kw: Any,
    ) -> Response:
        """Single completion. Keyword overrides: system, temperature, max_tokens,
        tool_choice, json_schema, schema_name; anything else goes to the provider."""
        messages = _to_messages(prompt, attachments)
        specs = [self.model, *self.fallbacks]
        last: BaseException | None = None
        for i, spec in enumerate(specs):
            provider_name, req = self._build(spec, messages, tools=tools, **kw)
            try:
                return self._call(provider_name, req)
            except Exception as e:  # noqa: BLE001
                # Only transient failures move on to a fallback model. A bad request, auth
                # error or bug would fail there too, and silently switching models hides it.
                if i + 1 >= len(specs) or not is_retryable(e):
                    raise
                last = e
                log.warning("%s failed (%s); falling back to %s", spec, e, specs[i + 1])
        assert last is not None
        raise last

    def _call(self, provider_name: str, req: Request) -> Response:
        with telemetry.span("llm.complete", provider_name, req.model, **{
                "gen_ai.request.temperature": req.temperature, "gen_ai.request.max_tokens": req.max_tokens}) as s:
            provider = self._provider(provider_name)
            key = request_key(provider_name, req) if self.cache and not req.tools else None
            if key and self.cache and (hit := self.cache.get(key)):
                self._record(Usage())
                telemetry.record_usage(s, Usage(), hit, **{"llmgw.cached": True})
                return hit
            self._check_budget()
            attempt = 0
            while True:
                t0 = time.perf_counter()
                try:
                    with limit_for(provider_name).acquire():
                        resp = provider.complete(req)
                except Exception as e:
                    latency = (time.perf_counter() - t0) * 1000
                    if attempt < self.max_retries and is_retryable(e):
                        delay = retry_after(e) or min(30.0, 2**attempt) * (0.5 + random.random())
                        log.info("retrying %s in %.1fs after %s", req.model, delay, e)
                        attempt += 1
                        time.sleep(delay)
                        continue
                    if self.logger:
                        self.logger.write(provider_name, req, None, latency, error=e)
                    raise
                resp.latency_ms = (time.perf_counter() - t0) * 1000
                resp.model = f"{provider_name}:{req.model}"
                resp.finish_reason = normalize_finish(resp.finish_reason)
                if resp.usage.cost_usd is None:
                    resp.usage.cost_usd = 0.0 if provider_name == "ollama" else pricing.cost(req.model, resp.usage)  # local: free
                self._record(resp.usage)
                if self.logger:
                    self.logger.write(provider_name, req, resp, resp.latency_ms)
                if key and self.cache and not resp.tool_calls:
                    self.cache.put(key, resp)
                telemetry.record_usage(s, resp.usage, resp, **{"llmgw.retries": attempt})
                return resp

    def text(self, prompt: str | list[Message], **kw: Any) -> str:
        return self.complete(prompt, **kw).text

    def structured(self, prompt: str | list[Message], schema: type[T], *, max_attempts: int = 3, **kw: Any) -> T:
        """Return a validated ``schema`` instance. On invalid output the error is
        fed back to the model and it gets another try."""
        messages = _to_messages(prompt, kw.pop("attachments", None))
        json_schema = schema.model_json_schema()
        text = ""
        for _ in range(max_attempts):
            resp = self.complete(messages, json_schema=json_schema, schema_name=schema.__name__, **kw)
            text = resp.text
            if resp.finish_reason == "length":
                # Asking the model to "correct" JSON that was cut off just repeats the cut.
                raise StructuredOutputError(
                    f"{schema.__name__} output was cut off at max_tokens ({kw.get('max_tokens') or self.max_tokens or 'provider default'}); "
                    "raise max_tokens", text)
            try:
                return schema.model_validate_json(_clean_json(text))
            except (ValidationError, json.JSONDecodeError) as e:
                messages = [
                    *messages,
                    Message.assistant(text),
                    Message.user(f"That output failed validation:\n{e}\nReply with only the corrected JSON."),
                ]
        raise StructuredOutputError(f"No valid {schema.__name__} after {max_attempts} attempts", text)

    def stream(self, prompt: str | list[Message], **kw: Any) -> Iterator[str]:
        """Stream text chunks (no retries or fallbacks). Logged and costed once complete,
        using the token counts the provider reports at the end of the stream."""
        messages = _to_messages(prompt, kw.pop("attachments", None))
        provider_name, req = self._build(self.model, messages, **kw)
        t0 = time.perf_counter()
        parts: list[str] = []
        # Not rate-limited: a slot held for a whole stream would deadlock code that calls the
        # same provider while iterating (Ollama's limit is 1). Limits target batch pools.
        usage = Usage()
        for chunk in self._provider(provider_name).stream(req):
            if isinstance(chunk, Usage):  # token counts reported at the end, not text
                usage = chunk
                continue
            parts.append(chunk)
            yield chunk
        if usage.cost_usd is None and (usage.input_tokens or usage.output_tokens):
            usage.cost_usd = 0.0 if provider_name == "ollama" else pricing.cost(req.model, usage)
        resp = Response(text="".join(parts), model=f"{provider_name}:{req.model}", usage=usage)
        self._record(usage)
        if self.logger:
            self.logger.write(provider_name, req, resp, (time.perf_counter() - t0) * 1000)

    def embed(self, texts: list[str], *, model: str | None = None, task: str | None = None) -> list[list[float]]:
        """Embed ``texts``. Vectors are cached by model, task and text hash (see
        ``LLMGW_EMBED_CACHE``), so only texts not seen before reach the provider."""
        provider_name, embed_model = parse_model(model or self.embed_model)
        texts = list(texts)
        cache = _embed_cache(self.embed_cache)
        if cache is None:
            return self._embed_uncached(provider_name, embed_model, texts, task)
        keys = [EmbeddingCache.key(f"{provider_name}:{embed_model}", task, t) for t in texts]
        found = cache.get_many(list(dict.fromkeys(keys)))
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            new = self._embed_uncached(provider_name, embed_model, [texts[i] for i in missing], task, n_cached=len(texts) - len(missing))
            fresh = {keys[i]: v for i, v in zip(missing, new)}
            cache.put_many(fresh)
            found.update(fresh)
        return [found[k] for k in keys]

    def _embed_uncached(self, provider_name: str, embed_model: str, texts: list[str], task: str | None,
                        n_cached: int = 0) -> list[list[float]]:
        with telemetry.span("llm.embed", provider_name, embed_model, **{
                "llmgw.texts": len(texts), "llmgw.texts_cached": n_cached}) as s:
            provider = self._provider(provider_name)
            self._check_budget()
            attempt = 0
            t0 = time.perf_counter()
            while True:
                try:
                    with limit_for(provider_name).acquire():
                        vectors = provider.embed(list(texts), embed_model, task=task)
                    if len(vectors) != len(texts):
                        raise RuntimeError(f"{provider_name}:{embed_model} returned {len(vectors)} embeddings for {len(texts)} texts")
                except Exception as e:
                    if attempt < self.max_retries and is_retryable(e):
                        time.sleep(retry_after(e) or min(30.0, 2**attempt) * (0.5 + random.random()))
                        attempt += 1
                        continue
                    if self.logger:
                        self.logger.write_embed(provider_name, embed_model, len(texts), (time.perf_counter() - t0) * 1000, error=e)
                    raise
                # Providers do not report tokens for embeddings; ~4 characters per token is the
                # usual estimate, so building an index shows up in usage and cost.
                usage = Usage(input_tokens=-(-sum(len(t) for t in texts) // 4))
                usage.cost_usd = 0.0 if provider_name == "ollama" else pricing.cost(embed_model, usage)
                self._record(usage)
                if self.logger:
                    self.logger.write_embed(provider_name, embed_model, len(texts), (time.perf_counter() - t0) * 1000, usage, n_cached=n_cached)
                telemetry.record_usage(s, usage)
                return vectors

    def batch(self, items: Iterable[Any], fn: Callable[[Any], Any] | None = None, concurrency: int = 8) -> list[Any]:
        """Run ``fn(item)`` (default: ``self.text(item)``) over items in parallel, keeping order."""
        fn = fn or self.text
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(fn, items))

    async def acomplete(self, prompt: str | list[Message], **kw: Any) -> Response:
        return await asyncio.to_thread(self.complete, prompt, **kw)

    async def astructured(self, prompt: str | list[Message], schema: type[T], **kw: Any) -> T:
        return await asyncio.to_thread(lambda: self.structured(prompt, schema, **kw))
