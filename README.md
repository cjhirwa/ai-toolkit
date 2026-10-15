# ai-toolkit

Reusable building blocks for LLM projects. Each tool lives in its own folder and is its own package, but they share one uv workspace so they can import each other.

| folder | package / CLI | what it's for |
|---|---|---|
| [llm-gateway](llm-gateway) | `llmgw` | one client over Gemini, OpenAI and Ollama: retries, fallbacks, structured output, tools, attachments, embeddings, cost, logs, cache |
| [eval-harness](eval-harness) | `evalh` | score any pipeline on a dataset with rule-based metrics and LLM judges; runs stored in SQLite; compare runs; HTML reports |
| [prompt-registry](prompt-registry) | `promptreg` | versioned YAML + Jinja prompts, with diffs |
| [doc-ingest](doc-ingest) | `docingest` | PDF, DOCX, HTML or scans → chunks with page and heading metadata; pluggable chunkers; OCR for scanned pages (off by default, see its README) |
| [retrieval-kit](retrieval-kit) | `retrievekit` | BM25, dense and hybrid (RRF) retrieval, MultiQuery/HyDE, LLM and cross-encoder rerankers, benchmark |
| [structured-extract](structured-extract) | `sextract` | documents → validated Pydantic records; merging for long documents; batch to JSONL/CSV |
| [mcp-servers](mcp-servers) | `mcp-retrieval`, `mcp-evals`, `mcp-prompts` | make the toolkit available to Claude Code, Claude Desktop and other agents |
| [agent-loop](agent-loop) | `agentloop` | a small, readable tool-calling loop with JSONL traces and an HTML trace viewer |
| [guardrails-router](guardrails-router) | `qroute` | route queries through rules, then embeddings, then an LLM (scope, safety, intent, language), with evaluation |
| [playground](playground) | `playground` | Streamlit app to compare models and prompts side by side, on one input or a dataset |
| [synth-data](synth-data) | `synthdata` | grounded synthetic Q&A eval sets and retrieval qrels from your own chunks |
| [embedding-viz](embedding-viz) | `embviz` | corpus map, clusters, near-duplicates, query coverage gaps |
| [speech-tools](speech-tools) | `speechkit` | speech-to-text (Gemini, OpenAI, Whisper) and glossary-aware translation with back-translation checks |

**Core and experimental.** `llm-gateway`, `doc-ingest`, `retrieval-kit`, `eval-harness` and `guardrails-router` are the core: they get the reliability work and contract tests. The other packages (`prompt-registry`, `structured-extract`, `mcp-servers`, `agent-loop`, `playground`, `synth-data`, `embedding-viz`, `speech-tools`) are experimental: tested and usable, but their interfaces may change between versions.

## How they fit together

```
docs ──docingest──▶ chunks.jsonl ──retrievekit build──▶ index/ ──▶ mcp-retrieval / agentloop tool / your RAG
                         │                                 │
                         └──synthdata──▶ eval.jsonl, qrels ─┴──▶ retrievekit bench  (is retrieval good?)
                                               └──────────────▶ evalh run          (are answers good?)
prompts/ (promptreg) ──▶ playground (eyeball)  ──▶ evalh compare (measure)
user query ──qroute──▶ in_scope → RAG │ urgent → hand off to a human │ out_of_scope → polite refusal
```

Everything that calls a model goes through `llmgw`, so changing provider or model is a one-string change, and every completion and embedding call is logged with its cost (embedding tokens are estimated).

## Setup

```
cd ai-toolkit
uv sync --all-packages --extra pdf   # every package (editable) into .venv, plus PyMuPDF for OCR and PDF headings
cp .env.example .env    # add the keys you have
uv run pytest           # offline tests (fake provider, no API calls)
uv run python scripts/smoke_live.py   # optional: real calls to each provider with a key, plus local Ollama if running
uv run python scripts/smoke_live.py --providers ollama   # only some providers (free when local)
# CI runs the Gemini smoke test daily (.github/workflows/live.yml) once a GOOGLE_API_KEY repository secret exists
```

## Configuration

All settings come from environment variables, and the project's `.env` is loaded automatically: the nearest one from the current folder up to the project root (a folder with `.git`, or a `pyproject.toml` outside a uv workspace), never from beyond it. Set `LLMGW_DOTENV` to a file to use that one, or to `0` to load none. `.env.example` lists every variable with a working default: provider keys, `DOCS_DIR` (your documents), the default, judge and OCR models, and where the index, eval database and prompts live. Nothing in the code depends on a particular machine, so after cloning you only edit `.env`. Run `uv run llmgw config` to see which model each role (chat, embeddings, judge, OCR) uses and where every setting comes from.

## Running each component

Every command below runs from the toolkit folder; put `uv run` in front of it (or activate `.venv`). A plain `uv sync` or `uv run` re-syncs the environment and removes optional extras such as PyMuPDF; use `uv run --all-packages --extra pdf <command>` when you need them.

### Run everything locally, with no API costs

With [Ollama](https://ollama.com) running, pull a chat model, an embedding model and a vision model, then point the toolkit at them in `.env`:

```
ollama pull llama3.1 && ollama pull nomic-embed-text && ollama pull gemma3:4b
```

```
LLMGW_DEFAULT_MODEL=ollama:llama3.1
LLMGW_EMBED_MODEL=ollama:nomic-embed-text
EVALH_JUDGE_MODEL=ollama:llama3.1
DOCINGEST_OCR_MODEL=ollama:gemma3:4b
LLMGW_TIMEOUT=600
```

A local server holds one model in memory at a time on most machines, so run one model-heavy job at a time: two jobs that need different models keep unloading each other and time out. A small local judge is fine for comparing runs but less reliable than a large hosted model for absolute scores.

### Commands

| component | try it |
|---|---|
| llm-gateway | `llmgw ask "Explain RAG in two lines"` · `llmgw stats` (calls, tokens, cost per model) |
| doc-ingest | `docingest run -o data/chunks.jsonl` (reads `DOCS_DIR`; add `--ocr llm` for scanned pages) · `docingest stats data/chunks.jsonl` |
| retrieval-kit | `retrievekit build data/chunks.jsonl -o data/index` · `retrievekit search data/index "how do I reset it?" --mode hybrid` |
| synth-data | `synthdata gen data/chunks.jsonl -o data/eval.jsonl --qrels data/qrels.jsonl --sample 20` |
| retrieval benchmark | `retrievekit bench data/index data/qrels.jsonl --modes bm25,dense,hybrid` |
| eval-harness | `evalh run --task examples/rag_task.py:answer --dataset data/eval.jsonl --metrics correctness,faithfulness --name local` · `evalh runs` · `evalh report local --open` |
| prompt-registry | `promptreg new support --template "Answer: {{ question }}"` · `promptreg render support --var question=hi` · `promptreg list` |
| guardrails-router | `qroute classify guardrails-router/examples/support_routes.yaml "the unit is smoking"` |
| structured-extract | `sextract run invoices/ --schema schemas.py:Invoice -o out.jsonl --csv out.csv` (you write the Pydantic schema) |
| embedding-viz | `embviz map data/index` · `embviz dupes data/index` · `embviz coverage data/index data/eval.jsonl` |
| agent-loop | a Python library ([agent-loop/README.md](agent-loop/README.md)); `agentloop ls` and `agentloop view` show saved traces |
| playground | `playground` opens a browser tab to compare models side by side |
| mcp-servers | `mcp-retrieval --http --port 8765`, or register it with an MCP client ([mcp-servers/README.md](mcp-servers/README.md)) |
| speech-tools | `speechkit transcribe audio.m4a --backend whisper` (local; needs `--extra whisper`) |

Each package's README has the full options. An index must be searched with the same embedding model it was built with, so rebuild it after changing `LLMGW_EMBED_MODEL`.

## Security and data handling

- **Documents are untrusted input.** Retrieved passages can contain text written to steer a model ("ignore your instructions…"). The RAG example and the LLM reranker fence passages in `<passages>` tags and tell the model they are reference material, not instructions; do the same in your own prompts, and give agents only the tools they need.
- **Logs** hold metadata only by default (`LLMGW_LOG_CONTENT=0`). The OCR cache (`.llmgw/ocr-cache.sqlite`) and `chunks.jsonl` contain document text, so store them as you would the documents; the embedding cache holds only hashes and vectors.
- **MCP over HTTP** needs `MCP_AUTH_TOKEN` to serve beyond localhost.
- **Dependencies** are checked for known vulnerabilities in CI (`pip-audit`). PyMuPDF, used for OCR and PDF headings, is AGPL-licensed and optional (`doc-ingest[pdf]`); check its licence before distributing it.
- **Keys** live in `.env`, which is git-ignored; never commit it.

## Using the toolkit from another project

Install the packages you need straight from GitHub. List `llm-gateway` alongside any other package, because they all depend on it:

```
uv add "llm-gateway[all] @ git+https://github.com/cjhirwa/ai-toolkit#subdirectory=llm-gateway" \
       "retrieval-kit @ git+https://github.com/cjhirwa/ai-toolkit#subdirectory=retrieval-kit"
# pip: pip install "llm-gateway[all] @ git+https://github.com/cjhirwa/ai-toolkit#subdirectory=llm-gateway" ...
```

If you are changing the toolkit at the same time, use a local clone instead: `uv add --editable /path/to/ai-toolkit/llm-gateway`.

## Quick start: from a folder of PDFs to a measured RAG pipeline

Put your documents in the folder named by `DOCS_DIR` in `.env` (default `docs/`), then:

```
uv run docingest run -o data/chunks.jsonl --chunker headings
uv run retrievekit build data/chunks.jsonl -o data/index
uv run synthdata gen data/chunks.jsonl -o data/eval.jsonl --qrels data/qrels.jsonl --sample 40 --unanswerable 5
uv run retrievekit bench data/index data/qrels.jsonl --modes bm25,dense,hybrid,hybrid+llmrerank
uv run evalh run --task examples/rag_task.py:answer --dataset data/eval.jsonl --metrics correctness,faithfulness --name baseline
uv run evalh report baseline --open
```

`examples/rag_task.py` is a 30-line RAG pipeline that ties these pieces together.
