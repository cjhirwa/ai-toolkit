"""Self-contained HTML reports for a run or a comparison of two runs."""

from __future__ import annotations

import html
import json

from .compare import Comparison
from .runner import RunResult

_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--card:#fff;--good:#1a7f4b;--bad:#b42318;--mid:#a15c07}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe8;--muted:#9c9b96;--line:#2c2b29;--card:#1e1e1c;--good:#4cc38a;--bad:#f97066;--mid:#f5b041}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}
.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{font-weight:600;font-size:12px;color:var(--muted)}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.good{color:var(--good)}.bad{color:var(--bad)}.mid{color:var(--mid)}
details{background:var(--card);border:1px solid var(--line);border-radius:6px;margin:6px 0;padding:6px 10px}summary{cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0 8px;font:12px/1.45 ui-monospace,Consolas,monospace}
.chips span{display:inline-block;margin-right:8px;font-variant-numeric:tabular-nums}
"""


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _cls(v) -> str:
    if not isinstance(v, (int, float)):
        return ""
    return "good" if v >= 0.8 else "mid" if v >= 0.5 else "bad"


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>{body}</main></body></html>"
    )


def _e(v) -> str:
    return html.escape(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1))


def run_report(run: RunResult) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num {_cls(v) if not k.startswith(('latency', 'error')) else ''}'>{_fmt(v)}</td></tr>"
        for k, v in run.summary.items()
    )
    metric_names = [k for k in run.summary if not k.startswith(("latency", "error", "n_"))]

    def worst_first(r):
        vals = [s.value for s in r.scores.values() if s.value is not None]
        return (0 if r.output.error else 1, min(vals) if vals else 1.0)

    items = []
    for r in sorted(run.results, key=worst_first):
        chips = "".join(
            f"<span class='{_cls(r.scores[m].value)}'>{m}: {_fmt(r.scores[m].value)}</span>" for m in metric_names if m in r.scores
        )
        reasons = "".join(
            f"<b>{html.escape(m)}</b><pre>{_e(s.reason)}</pre>" for m, s in r.scores.items() if s.reason
        )
        ctx = r.output.context or r.example.context
        items.append(
            f"<details><summary><code>{html.escape(r.example.id)}</code> <span class='chips'>{chips}</span>"
            f"{' <span class=bad>error</span>' if r.output.error else ''}</summary>"
            f"<b>Input</b><pre>{_e(r.example.input)}</pre>"
            + (f"<b>Expected</b><pre>{_e(r.example.expected)}</pre>" if r.example.expected is not None else "")
            + f"<b>Output</b><pre>{_e(r.output.error or r.output.text)}</pre>"
            + (f"<b>Context</b><pre>{_e(chr(10).join(ctx))}</pre>" if ctx else "")
            + reasons
            + "</details>"
        )
    body = (
        f"<h1>{html.escape(run.name)}</h1><div class='muted'>{run.run_id} · {run.created_at} · {len(run.results)} examples</div>"
        f"<h2>Summary</h2><table><tr><th>metric</th><th>value</th></tr>{rows}</table>"
        f"<h2>Config</h2><pre>{_e(run.config)}</pre>"
        f"<h2>Examples (worst first)</h2>{''.join(items)}"
    )
    return _page(f"Eval {run.name}", body)


def comparison_report(c: Comparison) -> str:
    rows = []
    for k, (x, y) in c.metric_deltas.items():
        d = None if x is None or y is None else y - x
        higher_better = not k.startswith(("latency", "error"))
        cls = "" if d is None or d == 0 else ("good" if (d > 0) == higher_better else "bad")
        rows.append(
            f"<tr><td>{html.escape(k)}</td><td class='num'>{_fmt(x)}</td><td class='num'>{_fmt(y)}</td>"
            f"<td class='num {cls}'>{'' if d is None else f'{d:+.3f}'}</td></tr>"
        )
    a_by = {r.example.id: r for r in c.a.results}
    b_by = {r.example.id: r for r in c.b.results}

    def block(title, deltas):
        out = [f"<h2>{title} ({len(deltas)})</h2>"]
        for d in deltas[:100]:
            ra, rb = a_by[d.example_id], b_by[d.example_id]
            out.append(
                f"<details><summary><code>{html.escape(d.example_id)}</code> {html.escape(d.metric)}: "
                f"{_fmt(d.before)} → <span class='{_cls(d.after)}'>{_fmt(d.after)}</span></summary>"
                f"<b>Input</b><pre>{_e(rb.example.input)}</pre>"
                f"<b>A output</b><pre>{_e(ra.output.text)}</pre><b>B output</b><pre>{_e(rb.output.text)}</pre>"
                f"<b>B reason</b><pre>{_e(rb.scores[d.metric].reason)}</pre></details>"
            )
        return "".join(out)

    body = (
        f"<h1>{html.escape(c.a.name)} → {html.escape(c.b.name)}</h1>"
        f"<div class='muted'>A {c.a.run_id} · B {c.b.run_id}</div>"
        f"<h2>Metrics</h2><table><tr><th>metric</th><th>A</th><th>B</th><th>Δ</th></tr>{''.join(rows)}</table>"
        + block("Regressions", c.regressions)
        + block("Improvements", c.improvements)
    )
    return _page(f"Compare {c.a.name} vs {c.b.name}", body)
