import json

import pytest
from pydantic import BaseModel

from llmgw import LLM, FakeProvider, Message, Response, StructuredOutputError, ToolCall, parse_model, tool
from llmgw.client import is_retryable


class Person(BaseModel):
    name: str
    age: int


def make(responses, **kw):
    return LLM("fake:test", provider=FakeProvider(responses), log_dir=False, **kw)


def test_parse_model():
    assert parse_model("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert parse_model("gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")
    assert parse_model("ollama:llama3.2:3b") == ("ollama", "llama3.2:3b")
    with pytest.raises(ValueError):
        parse_model("mystery-model")


def test_text_and_usage():
    llm = make(["hello there"])
    assert llm.text("hi") == "hello there"
    assert llm.calls == 1 and llm.usage.output_tokens == 2


def test_structured_retries_on_invalid_output():
    llm = make(['{"name": "Ana"}', '```json\n{"name": "Ana", "age": 31}\n```'])
    person = llm.structured("extract", Person)
    assert person == Person(name="Ana", age=31)
    assert llm.calls == 2
    retry_prompt = llm._provider_override.requests[1].messages[-1].content
    assert "failed validation" in retry_prompt


def test_retry_on_transient_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 Service Unavailable")
        return "ok"

    assert make(flaky).text("x") == "ok"
    assert calls["n"] == 3


def test_fallback_model(monkeypatch):
    from llmgw import register_provider

    monkeypatch.setattr("time.sleep", lambda s: None)

    class Overloaded(Exception):
        status_code = 503

    def broken(req):
        raise Overloaded("overloaded")

    def bad_request(req):
        raise ValueError("bad request")

    register_provider("broken", FakeProvider(broken))
    register_provider("bad", FakeProvider(bad_request))
    register_provider("fake", FakeProvider(["from fallback"]))
    resp = LLM("broken:x", fallbacks=["fake:y"], log_dir=False).complete("hi")
    assert resp.text == "from fallback" and resp.model == "fake:y"
    with pytest.raises(ValueError):  # a permanent error must not silently switch models
        LLM("bad:x", fallbacks=["fake:y"], log_dir=False).complete("hi")


def test_tool_spec_from_function():
    @tool
    def search(query: str, k: int = 5) -> list:
        """Search the knowledge base.

        Args:
            query: What to look for.
            k: How many results.
        """
        return [query] * k

    spec = search.spec
    assert spec.description == "Search the knowledge base."
    assert spec.parameters["required"] == ["query"]
    assert spec.parameters["properties"]["k"]["description"] == "How many results."
    assert search.run({"query": "x", "k": "2"}) == ["x", "x"]


def test_tool_call_roundtrip():
    call = ToolCall("c1", "search", {"query": "x"})
    llm = make([Response(text="", model="m", tool_calls=[call]), "done"])
    r1 = llm.complete("find x", tools=[lambda query: query])
    assert r1.tool_calls[0].name == "search"
    r2 = llm.complete([Message.user("find x"), r1.to_message(), Message.tool_result(call, "result")])
    assert r2.text == "done"


def test_logging_and_cache(tmp_path):
    llm = LLM("fake:t", provider=FakeProvider(["a", "b"]), log_dir=tmp_path / "logs", cache=tmp_path / "c.sqlite", log_content=True)
    assert llm.text("same") == "a"
    assert llm.text("same") == "a"  # served from cache
    records = [json.loads(l) for f in (tmp_path / "logs").glob("*.jsonl") for l in f.open()]
    assert len(records) == 1 and records[0]["text"] == "a"


def test_embed_similarity():
    llm = make([])
    a, b, c = llm.embed(["thermostat reset button", "reset button on the thermostat", "office public holidays"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))
    assert dot(a, b) > dot(a, c)


def test_ollama_nomic_task_prefixes():
    from llmgw.providers.openai_compat import OllamaProvider

    sent = []

    class Emb:
        def create(self, model, input):
            sent.extend(input)

            class R:
                data = [type("D", (), {"embedding": [1.0]})() for _ in input]

            return R()

    client = type("C", (), {"embeddings": Emb()})()
    p = OllamaProvider(client=client)
    p.embed(["q"], "nomic-embed-text", task="query")
    p.embed(["d"], "nomic-embed-text:latest", task="document")
    p.embed(["x"], "llama3.1", task="query")
    assert sent == ["search_query: q", "search_document: d", "x"]


def test_is_retryable():
    class E(Exception):
        status_code = 429

    class NotImpl(Exception):
        status_code = 501

    class BadRequest(Exception):
        status_code = 400

    class APIConnectionError(Exception):
        pass

    assert is_retryable(E())
    assert not is_retryable(NotImpl("server does not support embeddings, connection details"))
    assert not is_retryable(BadRequest("timeout parameter invalid"))  # status code beats message text
    assert not is_retryable(ValueError("invalid argument"))
    assert not is_retryable(RuntimeError("internal error: connection pool misconfigured"))  # words are not enough
    assert is_retryable(RuntimeError("503 UNAVAILABLE. {'error': ...}"))  # google-genai's format
    assert not is_retryable(RuntimeError("400 INVALID_ARGUMENT"))
    assert is_retryable(APIConnectionError()) and is_retryable(TimeoutError())


def test_provider_timeouts(monkeypatch):
    from llmgw.providers.gemini import GeminiProvider
    from llmgw.providers.openai_compat import OllamaProvider, OpenAIProvider

    monkeypatch.delenv("LLMGW_TIMEOUT", raising=False)
    assert OpenAIProvider(api_key="x").client.timeout == 120.0
    assert OllamaProvider().client.timeout == 600.0
    assert OpenAIProvider(api_key="x", timeout=5).client.timeout == 5
    assert GeminiProvider(api_key="x").client._api_client._http_options.timeout == 120_000
    monkeypatch.setenv("LLMGW_TIMEOUT", "30")
    assert OpenAIProvider(api_key="x").client.timeout == 30.0
    assert OllamaProvider().client.timeout == 30.0


def test_cli_logging_quiets_http_libraries():
    import logging

    from llmgw.console import QuietHTTPFilter, cli_logging

    cli_logging()
    filters = [f for h in logging.getLogger().handlers for f in h.filters if isinstance(f, QuietHTTPFilter)]
    assert filters
    record = lambda name, level: logging.LogRecord(name, level, __file__, 1, "HTTP Request: POST", None, None)
    for name in ("httpx", "httpx2", "httpcore.connection", "google_genai.models", "openai._base_client"):
        assert not filters[0].filter(record(name, logging.INFO))
    assert filters[0].filter(record("httpx2", logging.WARNING))
    assert filters[0].filter(record("docingest", logging.INFO))


def test_metadata_only_logging(tmp_path, monkeypatch, capsys):
    from llmgw.cli import main as cli

    secret = "Patient Jane Doe, account 4421"
    monkeypatch.delenv("LLMGW_LOG_CONTENT", raising=False)  # the default must be metadata only
    llm = LLM("fake:t", provider=FakeProvider([f"Reply about {secret}"]), log_dir=tmp_path / "logs")
    llm.text(secret, system=f"Context: {secret}")
    def boom(req):
        raise ValueError(f"model returned invalid JSON: {secret}")

    failing = LLM("fake:t", provider=FakeProvider(boom), log_dir=tmp_path / "logs", max_retries=0)
    with pytest.raises(ValueError, match="invalid JSON"):
        failing.text(secret)
    raw = "".join(f.read_text(encoding="utf-8") for f in (tmp_path / "logs").glob("*.jsonl"))
    assert "Jane Doe" not in raw and "4421" not in raw
    ok, err = [json.loads(line) for line in raw.splitlines()]
    assert ok["model"] == "t" and ok["usage"]["output_tokens"] > 0 and "latency_ms" in ok
    assert not {"messages", "system", "text", "tool_calls"} & ok.keys()
    assert ":" not in err["error"]  # error type only; messages can quote content
    cli(["stats", "--log-dir", str(tmp_path / "logs")])
    assert "fake:t" in capsys.readouterr().out


def test_log_content_argument_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMGW_LOG_CONTENT", "0")
    llm = LLM("fake:t", provider=FakeProvider(["hello"]), log_dir=tmp_path / "logs", log_content=True)
    monkeypatch.setenv("LLMGW_LOG_CONTENT", "1")
    assert LLM("fake:t", provider=FakeProvider(["x"]), log_dir=tmp_path / "l2").logger.content
    llm.text("hi")
    record = json.loads(next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["text"] == "hello"


def test_retry_after_header_is_honoured(monkeypatch):
    waits = []
    monkeypatch.setattr("time.sleep", waits.append)

    class Throttled(Exception):
        status_code = 429

        class response:
            headers = {"retry-after": "7"}

    calls = {"n": 0}

    def throttled_once(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Throttled()
        return "ok"

    assert make(throttled_once).text("x") == "ok"
    assert waits == [7.0]


def test_openai_sdk_does_not_retry_on_its_own():
    from llmgw.providers.openai_compat import OpenAIProvider

    assert OpenAIProvider(api_key="x").client.max_retries == 0


def test_finish_reason_is_normalised_and_truncation_is_an_error():
    from pydantic import BaseModel

    from llmgw import Response
    from llmgw.types import normalize_finish

    assert [normalize_finish(r) for r in ("FinishReason.MAX_TOKENS", "length", "FinishReason.STOP", "end_turn",
                                         "blocked: BlockedReason.SAFETY", None)] == \
        ["length", "length", "stop", "stop", "content_filter", None]

    class Refund(BaseModel):
        order_id: str

    llm = make([Response(text='{"order_id": "A-10', model="t", finish_reason="FinishReason.MAX_TOKENS")])
    with pytest.raises(StructuredOutputError, match="cut off at max_tokens"):
        llm.structured("Refund for A-1042?", Refund)
    assert llm.calls == 1  # no pointless "please fix this JSON" retry


def test_embeddings_are_logged_and_costed(tmp_path, monkeypatch, capsys):
    from llmgw.cli import main as cli

    monkeypatch.setenv("LLMGW_PRICING", "")
    llm = LLM("fake:t", provider=FakeProvider(), embed_model="fake:text-embedding-3-small", log_dir=tmp_path / "logs")
    llm.embed(["a" * 400, "b" * 400])
    record = json.loads(next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["kind"] == "embed" and record["n_texts"] == 2 and record["tokens_estimated"]
    assert record["usage"]["input_tokens"] == 200
    assert record["usage"]["cost_usd"] == pytest.approx(200 * 0.02 / 1_000_000)
    assert llm.usage.input_tokens == 200
    cli(["stats", "--log-dir", str(tmp_path / "logs")])
    assert "fake:text-embedding-3-small" in capsys.readouterr().out


def test_embedding_cache_sends_only_new_texts(tmp_path):
    seen = []

    class Counting(FakeProvider):
        def embed(self, texts, model, task=None):
            seen.append(list(texts))
            return super().embed(texts, model, task)

    llm = LLM("fake:t", provider=Counting(), embed_model="fake:e", log_dir=tmp_path / "logs", embed_cache=tmp_path / "e.sqlite")
    first = llm.embed(["alpha", "beta"])
    again = llm.embed(["beta", "gamma", "alpha"])
    assert seen == [["alpha", "beta"], ["gamma"]]  # only the new text reached the provider
    assert again[0] == first[1] and again[2] == first[0]
    assert llm.embed(["alpha"], task="query") and seen[-1] == ["alpha"]  # task is part of the key
    records = [json.loads(l) for l in next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert records[1]["n_texts"] == 1 and records[1]["n_cached"] == 2
    assert LLM("fake:t", provider=Counting(), embed_cache=False).embed(["alpha"]) and seen[-1] == ["alpha"]


def test_provider_limits_are_shared_across_clients(monkeypatch):
    import threading
    import time as _time

    from llmgw.limits import limit_for, reset_limits

    monkeypatch.setenv("LLMGW_CONCURRENCY_LIMITED", "2")
    reset_limits()
    in_flight, peak, lock = [0], [0], threading.Lock()

    def slow(req):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        _time.sleep(0.05)
        with lock:
            in_flight[0] -= 1
        return "ok"

    from llmgw import register_provider

    register_provider("limited", FakeProvider(slow))
    clients = [LLM("limited:x", log_dir=False) for _ in range(3)]  # separate clients, one budget
    threads = [threading.Thread(target=c.text, args=("hi",)) for c in clients for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 2

    monkeypatch.setenv("LLMGW_RPM_THROTTLED", "120")  # 2 per second, burst of 10
    reset_limits()
    bucket = limit_for("throttled").bucket
    bucket.tokens = 0
    start = _time.monotonic()
    with limit_for("throttled").acquire():
        pass
    assert 0.4 < _time.monotonic() - start < 1.0  # waited for a token instead of firing
    reset_limits()
    assert limit_for("ollama").slots is not None and limit_for("gemini").slots is None


def test_pricing_cached_tokens_and_local_models():
    from llmgw import Response, Usage, pricing

    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=400_000)
    assert pricing.cost("gemini-3.6-flash", usage) == pytest.approx(0.6 * 0.75 + 0.4 * 0.075 + 3.75)
    assert pricing.cost("text-embedding-3-small", Usage(input_tokens=1_000_000)) == pytest.approx(0.02)  # no cached rate
    assert pricing.cost("mystery-model", usage) is None

    from llmgw import register_provider

    register_provider("ollama", FakeProvider([Response(text="hi", model="m", usage=Usage(input_tokens=10, output_tokens=5))]))
    try:
        assert LLM("ollama:llama3.1", log_dir=False).complete("hi").usage.cost_usd == 0.0
    finally:
        import llmgw.client

        llmgw.client._INSTANCES.pop("ollama", None)


def test_env_file_search_stops_at_the_project_root(tmp_path, monkeypatch):
    from llmgw.client import find_env_file

    monkeypatch.delenv("LLMGW_DOTENV", raising=False)
    (tmp_path / ".env").write_text("SECRET=from-an-unrelated-parent\n", encoding="utf-8")
    project = tmp_path / "other-project"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert find_env_file(project / "src") is None  # the parent's .env is outside this project
    (project / ".env").write_text("A=1\n", encoding="utf-8")
    assert find_env_file(project / "src") == (project / ".env").resolve()
    monkeypatch.setenv("LLMGW_DOTENV", "0")
    assert find_env_file(project / "src") is None
    monkeypatch.setenv("LLMGW_DOTENV", str(tmp_path / ".env"))
    assert find_env_file(project) == tmp_path / ".env"


def test_env_file_found_from_a_workspace_member(tmp_path, monkeypatch):
    from llmgw.client import find_env_file

    monkeypatch.delenv("LLMGW_DOTENV", raising=False)
    (tmp_path / "pyproject.toml").write_text("[tool.uv.workspace]\nmembers = ['pkg']\n", encoding="utf-8")
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "pyproject.toml").write_text("[project]\nname='pkg'\n", encoding="utf-8")
    assert find_env_file(tmp_path / "pkg") == (tmp_path / ".env").resolve()


def test_config_command_shows_sources_and_hides_keys(tmp_path, monkeypatch, capsys):
    from llmgw.cli import main as cli

    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text("LLMGW_EMBED_MODEL=ollama:nomic-embed-text\nOPENAI_API_KEY=sk-secret-value\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    import os

    env = {k: v for k, v in os.environ.items()
           if k not in ("LLMGW_DOTENV", "LLMGW_EMBED_MODEL", "OPENAI_API_KEY", "EVALH_JUDGE_MODEL")}
    env["LLMGW_DEFAULT_MODEL"] = "fake:chat"
    monkeypatch.setattr(os, "environ", env)  # loading .env writes here, and is undone after the test
    import llmgw.client

    monkeypatch.setattr(llmgw.client, "_dotenv_loaded", False)
    cli(["config"])
    out = capsys.readouterr().out
    assert "ollama:nomic-embed-text" in out and "[.env]" in out
    assert "fake:chat" in out and "[environment]" in out
    assert "judge is the same model" in out  # judge falls back to the default model
    assert "sk-secret-value" not in out and "OPENAI_API_KEY               set" in out


def test_spending_caps(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from llmgw import BudgetExceeded, Response, Usage

    def paid(req):
        return Response(text="ok", model="t", usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.01))

    llm = LLM("fake:t", provider=FakeProvider(paid), log_dir=False, max_cost_usd=0.025)
    for _ in range(3):
        llm.text("hi")  # 0.00, 0.01, 0.02 spent before each call: allowed
    with pytest.raises(BudgetExceeded, match="max_cost_usd"):
        llm.text("hi")

    monkeypatch.setenv("LLMGW_DAILY_BUDGET_USD", "0.03")
    logs = tmp_path / "logs"
    a = LLM("fake:t", provider=FakeProvider(paid), log_dir=logs)
    b = LLM("fake:t", provider=FakeProvider(paid), log_dir=logs)  # another client, same log folder
    a.text("1")
    b.text("2")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with (logs / f"calls-{day}.jsonl").open("a", encoding="utf-8") as f:  # spend logged by another process
        f.write(json.dumps({"usage": {"cost_usd": 0.01}}) + "\n")
    with pytest.raises(BudgetExceeded, match="spent today"):
        a.text("3")

    cached = LLM("fake:t", provider=FakeProvider(paid), log_dir=tmp_path / "logs2", cache=tmp_path / "c.sqlite")
    monkeypatch.setenv("LLMGW_DAILY_BUDGET_USD", "1")
    cached.text("same")
    monkeypatch.setenv("LLMGW_DAILY_BUDGET_USD", "0")
    assert cached.text("same") == "ok"  # a cache hit costs nothing, so it is not blocked
    with pytest.raises(BudgetExceeded, match="logging is off"):
        LLM("fake:t", provider=FakeProvider(paid), log_dir=False).text("x")


def test_calls_emit_opentelemetry_spans(tmp_path, monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    from llmgw import Response, Usage

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    monkeypatch.setattr("time.sleep", lambda s: None)

    attempts = {"n": 0}

    class Busy(Exception):
        status_code = 503

    def flaky(req):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Busy()
        return Response(text="ok", model="t", finish_reason="stop", usage=Usage(input_tokens=12, output_tokens=3, cost_usd=0.002))

    llm = LLM("fake:t", provider=FakeProvider(flaky), log_dir=False, cache=tmp_path / "c.sqlite")
    llm.text("hi")
    llm.text("hi")  # served from the cache
    llm.embed(["a", "b"])

    def boom(req):
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        LLM("fake:t", provider=FakeProvider(boom), log_dir=False).text("x")

    spans = exporter.get_finished_spans()
    first, cached, embed, failed = spans[-4:]
    assert first.name == "llm.complete" and first.attributes["gen_ai.system"] == "fake"
    assert first.attributes["gen_ai.usage.input_tokens"] == 12 and first.attributes["llmgw.cost_usd"] == 0.002
    assert first.attributes["llmgw.retries"] == 1 and first.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert cached.attributes["llmgw.cached"] is True
    assert embed.name == "llm.embed" and embed.attributes["llmgw.texts"] == 2
    assert failed.status.status_code == StatusCode.ERROR and failed.events[0].name == "exception"
