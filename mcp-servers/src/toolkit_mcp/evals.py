"""Read eval-harness results so an assistant can answer "did my change help?".

Env: EVALH_DB (default ./evals.db)."""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from ._run import run

mcp = MCPServer("evals")


def _store():
    from evalh import Store

    return Store(os.getenv("EVALH_DB", "evals.db"))


@mcp.tool()
def list_runs(limit: int = 20) -> list[dict]:
    """Recent eval runs with their summary metrics, newest first."""
    return _store().list_runs(limit)


@mcp.tool()
def run_summary(run: str) -> dict:
    """Summary metrics and config for one run (id, id prefix, or run name for its latest run)."""
    r = _store().load(run)
    return {"id": r.run_id, "name": r.name, "created_at": r.created_at, "examples": len(r.results),
            "summary": r.summary, "config": r.config}


@mcp.tool()
def compare_runs(a: str, b: str, threshold: float = 0.25, max_examples: int = 10) -> dict:
    """Compare run B against run A: metric deltas plus the biggest per-example regressions and improvements."""
    from evalh import compare

    store = _store()
    c = compare(store.load(a), store.load(b), threshold)
    by_id = {r.example.id: r for r in c.b.results}

    def rows(deltas):
        return [{"example": d.example_id, "metric": d.metric, "before": d.before, "after": d.after,
                 "input": str(by_id[d.example_id].example.input)[:300],
                 "reason": by_id[d.example_id].scores[d.metric].reason[:400]} for d in deltas[:max_examples]]

    return {
        "metrics": {k: {"a": x, "b": y} for k, (x, y) in c.metric_deltas.items()},
        "regressions": rows(c.regressions),
        "improvements": rows(c.improvements),
    }


@mcp.tool()
def failing_examples(run: str, metric: str, below: float = 0.5, limit: int = 10) -> list[dict]:
    """Examples in a run that scored below a threshold on a metric, with output and judge reasoning."""
    r = _store().load(run)
    return [
        {"id": x.example.id, "input": x.example.input, "expected": x.example.expected,
         "output": x.output.text[:1500], "score": x.scores[metric].value, "reason": x.scores[metric].reason}
        for x in r.failures(metric, below)[:limit]
    ]


def main() -> None:
    run(mcp)


if __name__ == "__main__":
    main()
