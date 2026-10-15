"""Render a JSONL trace as a single self-contained HTML page."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .tracing import read_trace

_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--card:#fff;--llm:#2563eb;--tool:#7c3aed;--err:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe8;--muted:#9c9b96;--line:#2c2b29;--card:#1e1e1c;--llm:#60a5fa;--tool:#a78bfa;--err:#f97066}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}h1{font-size:18px;margin:0}.muted{color:var(--muted);font-size:12px}
.ev{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:6px;margin:10px 0;padding:8px 12px}
.ev.llm{border-left-color:var(--llm)}.ev.tool{border-left-color:var(--tool)}.ev.err{border-left-color:var(--err)}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.kind{font-weight:600}.err .kind{color:var(--err)}
pre{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font:12px/1.45 ui-monospace,Consolas,monospace}
details summary{cursor:pointer;color:var(--muted);font-size:12px}
"""


def _pre(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)
    return f"<pre>{html.escape(s)}</pre>"


def render(path: str | Path) -> str:
    events = read_trace(path)
    start = next((e for e in events if e["kind"] == "run_start"), {})
    end = next((e for e in events if e["kind"] == "run_end"), {})
    parts = [
        f"<h1>{html.escape(start.get('name', 'agent'))}</h1>",
        f"<div class='muted'>{html.escape(start.get('run_id', ''))} · {html.escape(str(start.get('model', '')))} · "
        f"tools: {html.escape(', '.join(start.get('tools', [])))}</div>",
        f"<div class='ev'><div class='kind'>Task</div>{_pre(start.get('task', ''))}</div>",
    ]
    for e in events:
        if e["kind"] == "llm":
            u = e.get("usage") or {}
            calls = ", ".join(c["name"] for c in e.get("tool_calls", [])) or "final answer"
            parts.append(
                f"<div class='ev llm'><div class='head'><span class='kind'>LLM · step {e['step']}</span>"
                f"<span class='muted'>{e.get('latency_ms', 0):.0f} ms · {u.get('input_tokens', 0)}→{u.get('output_tokens', 0)} tok · {html.escape(calls)}</span></div>"
                + (_pre(e["text"]) if e.get("text") else "")
                + "</div>"
            )
        elif e["kind"] == "tool":
            cls = "tool err" if e.get("error") else "tool"
            parts.append(
                f"<div class='ev {cls}'><div class='head'><span class='kind'>{html.escape(e['name'])}</span>"
                f"<span class='muted'>{e.get('latency_ms', 0):.0f} ms</span></div>{_pre(e.get('arguments', {}))}"
                f"<details><summary>result ({len(e.get('result') or '')} chars)</summary>{_pre(e.get('result') or '')}</details></div>"
            )
    if end:
        u = end.get("usage") or {}
        cost = u.get("cost_usd")
        parts.append(
            f"<div class='ev'><div class='head'><span class='kind'>Done · {html.escape(end.get('stopped', ''))}</span>"
            f"<span class='muted'>{end.get('steps')} steps · {u.get('input_tokens', 0)}→{u.get('output_tokens', 0)} tok"
            f"{f' · ${cost:.4f}' if cost is not None else ''}</span></div>{_pre(end.get('output', ''))}</div>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Agent trace</title><style>{_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
    )
