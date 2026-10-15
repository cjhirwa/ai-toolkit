"""Real calls against each configured provider: text, structured output, a tool call, embeddings.

Providers are included when their key is set (Gemini, OpenAI) or, for Ollama, when a local
server answers. Each check runs on its own, so one unsupported feature does not hide the rest.

    uv run python scripts/smoke_live.py                       # every configured provider
    uv run python scripts/smoke_live.py --providers ollama    # only these (comma-separated)
    uv run python scripts/smoke_live.py openai:gpt-4o-mini    # specific model(s), no embeddings

For Ollama, set SMOKE_OLLAMA_MODEL / SMOKE_OLLAMA_EMBED_MODEL to choose models; otherwise the
first installed model that supports tools (and the first embedding model) is used.
"""

import json
import os
import sys
import urllib.request

from pydantic import BaseModel

from llmgw import LLM, Message, load_env, tool

KEYED = {
    "GOOGLE_API_KEY": ("gemini:gemini-3.6-flash", "gemini:gemini-embedding-2-preview"),
    "GEMINI_API_KEY": ("gemini:gemini-3.6-flash", "gemini:gemini-embedding-2-preview"),
    "OPENAI_API_KEY": ("openai:gpt-4o-mini", "openai:text-embedding-3-small"),
}


class Capital(BaseModel):
    country: str
    capital: str


@tool
def get_weather(city: str) -> str:
    """Current weather for a city.

    Args:
        city: City name.
    """
    return f"22°C and sunny in {city}"


def _ollama_api(path: str, payload: dict | None = None, timeout: float = 3) -> dict:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/").removesuffix("/v1")
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(base + path, data=data), timeout=timeout) as r:
        return json.load(r)


def ollama_target() -> tuple[str, str | None] | None:
    """(chat model, embedding model) from the local Ollama server, or None if it is not running."""
    try:
        names = [m["name"] for m in _ollama_api("/api/tags")["models"]]
    except OSError:
        return None
    caps = {n: _ollama_api("/api/show", {"model": n}).get("capabilities", []) for n in names}
    chat = os.getenv("SMOKE_OLLAMA_MODEL") or next(
        (n for n in names if "tools" in caps[n]), next((n for n in names if "completion" in caps[n]), None))
    embed = os.getenv("SMOKE_OLLAMA_EMBED_MODEL") or next((n for n in names if "embedding" in caps[n]), None)
    return (f"ollama:{chat}", f"ollama:{embed}" if embed else None) if chat else None


def check(model: str, embed_model: str | None) -> int:
    """Run each check independently; return the number that failed."""
    llm = LLM(model, temperature=0, max_tokens=300)
    print(f"\n== {model}")
    failures = 0

    def step(name, fn):
        nonlocal failures
        try:
            print(f"  {name:<11} {fn()}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  {name:<11} FAILED: {type(e).__name__}: {str(e)[:200]}")

    def text():
        r = llm.complete("Reply with exactly: pong")
        return f"{r.text.strip()!r}  ({r.latency_ms:.0f} ms, cost {r.usage.cost_usd})"

    def tool_call():
        q = "What's the weather in Kigali? Use the tool."
        r = llm.complete(q, tools=[get_weather])
        if not r.tool_calls:
            raise RuntimeError(f"no tool call; replied {r.text.strip()[:80]!r}")
        msgs = [Message.user(q), r.to_message(), Message.tool_result(r.tool_calls[0], get_weather.run(r.tool_calls[0].arguments))]
        after = llm.complete(msgs, tools=[get_weather]).text.strip()[:80]
        return f"{[(c.name, c.arguments) for c in r.tool_calls]} -> {after!r}"

    def embeddings():
        texts = ["how do I reset my thermostat", "restoring the thermostat to factory settings", "our office is closed on public holidays"]
        v = llm.embed(texts, model=embed_model, task="document")
        cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
        related, unrelated = cos(v[0], v[1]), cos(v[0], v[2])
        if related <= unrelated:
            raise RuntimeError(f"related {related:.3f} <= unrelated {unrelated:.3f}")
        return f"{len(v)} x {len(v[0])}  sim(related)={related:.3f} sim(unrelated)={unrelated:.3f}"

    step("text", text)
    step("structured", lambda: llm.structured("Capital of Rwanda?", Capital))
    step("tool call", tool_call)
    if embed_model:
        step("embeddings", embeddings)
    return failures


def main(argv: list[str]) -> int:
    from llmgw.console import utf8_stdio

    utf8_stdio()
    load_env()
    only = None
    if argv[:1] == ["--providers"]:
        only, argv = set(argv[1].split(",")), argv[2:]
    if argv:
        targets = [(m, None) for m in argv]
    else:
        targets = list(dict.fromkeys(v for k, v in KEYED.items() if os.getenv(k)))
        if (only is None or "ollama" in only) and (local := ollama_target()):
            targets.append(local)
    if only is not None:
        targets = [t for t in targets if t[0].split(":", 1)[0] in only]
    if not targets:
        print("No providers to test: set a key in .env, start Ollama, or check --providers.")
        return 1
    failed = sum(check(model, embed) for model, embed in targets)
    print(f"\n{failed} check(s) failed" if failed else "\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
