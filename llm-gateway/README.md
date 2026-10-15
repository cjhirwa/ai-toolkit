# llm-gateway (`llmgw`)

One client for Gemini, OpenAI and Ollama. It handles retries with backoff, fallback models, structured output with Pydantic, tool calling, images, audio and PDFs as attachments, embeddings, cost tracking, JSONL call logs and an optional response cache.

```python
from llmgw import LLM, tool
from pydantic import BaseModel

llm = LLM("gemini:gemini-3.6-flash", temperature=0.2, fallbacks=["openai:gpt-4o-mini"])

llm.text("Explain what a smart thermostat does in one sentence.")

class Refund(BaseModel):
    order_id: str
    amount: float

llm.structured("Order A-1042 was refunded 49.90 EUR.", Refund)   # validated, retried on bad JSON

for chunk in llm.stream("Write a haiku"):
    print(chunk, end="")

vectors = llm.embed(["text a", "text b"], task="document")
print(llm.usage, llm.calls)                                       # running totals
```

## Model specs

Write a model as `provider:model`. You can leave out the provider prefix for `gemini*`, `gemma*`, `gpt*`, and `o1`/`o3`/`o4`.

| provider | env vars | notes |
|---|---|---|
| `gemini` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` (or Vertex vars) | native JSON schema, embeddings with task types |
| `openai` | `OPENAI_API_KEY` | Chat Completions, `json_schema` response format |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | OpenAI-compatible endpoint |
| `fake` | – | offline, scripted responses, for tests |

## Tools

```python
@tool
def order_status(order_id: str, include_items: bool = False) -> str:
    """Look up an order.

    Args:
        order_id: Order number, e.g. A-1042.
        include_items: Also list what was ordered.
    """
    ...

resp = llm.complete("Where is my order A-1042?", tools=[order_status])
resp.tool_calls   # [ToolCall(id, name, arguments)]
```

`agent-loop` builds the full tool-use loop on top of this.

## Attachments

```python
from llmgw import Attachment
llm.text("Transcribe this page", attachments=[Attachment.from_path("scan.png")])
```

Gemini accepts images, audio and PDFs. OpenAI accepts images.

## Logs, cost and cache

- Every call goes to `.llmgw/logs/calls-YYYY-MM-DD.jsonl`. Change the folder with `LLMGW_LOG_DIR`, turn logging off with `LLMGW_LOG=0`, or pass `log_dir=False`.
- By default each record holds only metadata: model, token counts, cost, latency, finish reason and the error type, never prompts, replies or document text, because those may contain personal or confidential data. Set `LLMGW_LOG_CONTENT=1` (or pass `log_content=True`) to also log prompts and replies while debugging. `llmgw stats` works the same either way. The optional response cache (`cache=True`) stores replies separately, so leave it off for such data.
- `llmgw stats --days 7` prints calls, tokens, latency and cost per model.
- Cost comes from `pricing.py` defaults (Gemini and current OpenAI prices from the providers' pricing pages, 2026-10-08) plus your own `LLMGW_PRICING=pricing.json` (`{"model": [in_per_M, out_per_M, cached_in_per_M]}`; the cached rate is optional). Cached input tokens are charged at the cached rate when one is known. Ollama calls cost 0. Models without a price report `None` rather than a guessed number. Note that `gemini-3.6-flash` is at an introductory price that doubles on 2027-01-01.
- `LLM(..., cache=True)` caches identical requests in SQLite (`.llmgw/cache.sqlite`). Use it for eval reruns.
- Embeddings are cached by default in `.llmgw/embed-cache.sqlite`, keyed by model, task and a hash of the text, so rebuilding an index only embeds passages that changed (a 2,156-passage rebuild went from 58 s to 1 s). The file holds hashes and vectors, no text. Set `LLMGW_EMBED_CACHE` to another path, or to `0` to turn it off; `LLM(embed_cache=False)` turns it off for one client.

## Monitoring

Install `llm-gateway[otel]` and every completion and embedding becomes an OpenTelemetry span (`llm.complete`, `llm.embed`) with the standard GenAI attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`) plus `llmgw.cost_usd`, `llmgw.retries` and `llmgw.cached`. Failed calls are marked as errors with the exception attached. Configure the OpenTelemetry SDK and an exporter in your app (OTLP to Grafana, Datadog, Honeycomb, Jaeger and so on) and alert on error rate, latency or cost; without an SDK the spans are no-ops.

## Spending caps

- `LLM(..., max_cost_usd=0.50)` stops one client after it has spent that much.
- `LLMGW_DAILY_BUDGET_USD=5` stops every client that logs to the same folder once today's (UTC) spend reaches $5. The total is read from the call log as it grows, so separate processes share the budget; it requires call logging.

A capped call raises `BudgetExceeded` instead of reaching the provider. The check runs before each call, so the call that crosses the line still completes; leave some headroom. Calls to models without a known price count as 0, so add prices for the models you use (see Logs, cost and cache). Cached answers are free and never blocked.

## Rate limits

Every `LLM` in a process shares one budget per provider, so the toolkit's thread pools (eval, OCR, synthetic data, `batch`) cannot overrun a provider between them:

- `LLMGW_RPM_<PROVIDER>`: requests per minute, e.g. `LLMGW_RPM_GEMINI=60`. Calls wait for their turn instead of collecting 429s.
- `LLMGW_CONCURRENCY_<PROVIDER>`: requests in flight at once. Ollama defaults to 1, because a local server answers one request at a time; other providers are unlimited unless set.

Completions and embeddings are limited; streams are not.

## Defaults

- `LLMGW_DEFAULT_MODEL` (default `gemini:gemini-3.6-flash`)
- `LLMGW_EMBED_MODEL` (default `gemini:gemini-embedding-2-preview`)
- `LLMGW_TIMEOUT`: seconds before a request is abandoned and retried (default 120, or 600 for Ollama). Providers also take a `timeout=` argument.

The project's `.env` is loaded automatically: the nearest one from the current directory up to the project root (`.git`, or a `pyproject.toml` that is not a uv workspace member). It never reads beyond the project, so using llmgw as a library cannot pick up another project's settings. `LLMGW_DOTENV=path` uses a specific file; `LLMGW_DOTENV=0` turns loading off.

## CLI

```
llmgw ask "Summarise our returns policy in 3 bullets" -m openai:gpt-4o-mini
echo "long text" | llmgw ask - --stream
llmgw stats
llmgw config      # which model each role uses, and whether it came from the environment, .env or a default
```
