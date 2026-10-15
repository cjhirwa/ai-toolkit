"""llmgw ask "prompt" -m openai:gpt-4o-mini
llmgw stats [--days 7] [--log-dir .llmgw/logs]
llmgw config"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .console import utf8_stdio


def _ask(args) -> None:
    from .client import LLM

    llm = LLM(args.model, temperature=args.temperature, system=args.system)
    prompt = args.prompt if args.prompt != "-" else sys.stdin.read()
    if args.stream:
        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        resp = llm.complete(prompt)
        print(resp.text)
        cost = f"${resp.usage.cost_usd:.5f}" if resp.usage.cost_usd is not None else "cost unknown"
        print(
            f"\n[{resp.model} · {resp.latency_ms:.0f} ms · {resp.usage.input_tokens}→{resp.usage.output_tokens} tok · {cost}]",
            file=sys.stderr,
        )


def _stats(args) -> None:
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    rows: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "in": 0, "out": 0, "cost": 0.0, "unpriced": 0, "ms": 0.0})
    for f in sorted(Path(args.log_dir).glob("calls-*.jsonl")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            if datetime.fromisoformat(r["ts"]) < since:
                continue
            row = rows[f'{r["provider"]}:{r["model"]}']
            row["calls"] += 1
            row["ms"] += r.get("latency_ms", 0)
            if "error" in r:
                row["errors"] += 1
                continue
            u = r.get("usage") or {}
            row["in"] += u.get("input_tokens", 0)
            row["out"] += u.get("output_tokens", 0)
            if u.get("cost_usd") is None:
                row["unpriced"] += 1
            else:
                row["cost"] += u["cost_usd"]
    if not rows:
        print(f"No calls logged in {args.log_dir} in the last {args.days} days.")
        return
    print(f"{'model':40} {'calls':>6} {'errors':>6} {'in tok':>10} {'out tok':>10} {'avg ms':>8} {'cost $':>9}")
    total = 0.0
    for model, r in sorted(rows.items(), key=lambda kv: -kv[1]["calls"]):
        cost = f"{r['cost']:.4f}" + ("*" if r["unpriced"] else "")
        total += r["cost"]
        print(f"{model:40} {r['calls']:>6} {r['errors']:>6} {r['in']:>10} {r['out']:>10} {r['ms'] / r['calls']:>8.0f} {cost:>9}")
    print(f"\nTotal priced cost: ${total:.4f}  (* = some calls had no price; add them to LLMGW_PRICING)")


def _config(args) -> None:
    """Which model each role uses and where every setting comes from, so the implicit
    defaults that components fall back on are visible in one place."""
    import os

    from .client import find_env_file, load_env

    before = set(os.environ)
    load_env()

    def show(name: str, default: str, note: str = "") -> str:
        value = os.getenv(name)
        source = "environment" if name in before else ".env" if value is not None else "default"
        shown = value if value is not None else default
        print(f"  {name:28} {shown:42} [{source}]{'  ' + note if note else ''}")
        return shown

    print(f".env file: {find_env_file() or 'none'}\n\nmodels")
    chat = show("LLMGW_DEFAULT_MODEL", "gemini:gemini-3.6-flash", "answers, generation, reranking, routing")
    show("LLMGW_EMBED_MODEL", "gemini:gemini-embedding-2-preview", "indexing and search")
    judge = show("EVALH_JUDGE_MODEL", chat, "LLM judges in evalh")
    show("DOCINGEST_OCR_MODEL", "gemini:gemini-3.6-flash", "--ocr llm")  # doc-ingest's own default
    if judge == chat:
        print("  warning: the judge is the same model as the default model; a model grading its own answers inflates scores")
    print("\nrequests")
    show("LLMGW_TIMEOUT", "120 (600 for ollama)", "seconds")
    show("LLMGW_DAILY_BUDGET_USD", "none", "cap on today's spend across all clients logging here")
    limits = sorted(k for k in os.environ if k.startswith(("LLMGW_RPM_", "LLMGW_CONCURRENCY_")))
    for k in limits:
        show(k, "")
    if "LLMGW_CONCURRENCY_OLLAMA" not in limits:
        print(f"  {'LLMGW_CONCURRENCY_OLLAMA':28} {'1':42} [default]")
    print("\nlogs and caches")
    show("LLMGW_LOG", "1")
    show("LLMGW_LOG_DIR", ".llmgw/logs")
    show("LLMGW_LOG_CONTENT", "0", "1 also logs prompts and replies")
    show("LLMGW_EMBED_CACHE", ".llmgw/embed-cache.sqlite")
    show("DOCINGEST_OCR_CACHE", ".llmgw/ocr-cache.sqlite")
    print("\nprovider keys")
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        print(f"  {k:28} {'set' if os.getenv(k) else 'not set'}")  # never print key values
    print(f"  {'OLLAMA_BASE_URL':28} {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')}")


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="llmgw")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="send one prompt ('-' reads stdin)")
    a.add_argument("prompt")
    a.add_argument("-m", "--model")
    a.add_argument("-t", "--temperature", type=float)
    a.add_argument("-s", "--system")
    a.add_argument("--stream", action="store_true")
    s = sub.add_parser("stats", help="summarise logged calls")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--log-dir", default=".llmgw/logs")
    sub.add_parser("config", help="show which models and settings are in effect, and where each comes from")
    args = p.parse_args(argv)
    {"ask": _ask, "stats": _stats, "config": _config}[args.cmd](args)


if __name__ == "__main__":
    main()
