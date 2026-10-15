from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .dataset import Example
from .metrics import Output, Score
from .runner import ExampleResult, RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, name TEXT, created_at TEXT, config TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT, example_id TEXT, input TEXT, expected TEXT, context TEXT, metadata TEXT,
    output TEXT, output_context TEXT, error TEXT, latency_ms REAL, scores TEXT,
    PRIMARY KEY (run_id, example_id)
);
"""


def _j(v) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


class Store:
    def __init__(self, path: str | Path = "evals.db"):
        self.path = Path(path)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def save(self, run: RunResult) -> None:
        """Write a whole run: header with summary, then every result."""
        self.save_header(run)
        for r in run.results:
            self.add_result(run.run_id, r)

    def save_header(self, run: RunResult) -> None:
        """Create or update the run row. Called at the start of a run (so results written
        as they finish have a parent) and again at the end with the summary."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (run.run_id, run.name, run.created_at, _j(run.config), _j(run.summary)),
            )

    def add_result(self, run_id: str, r: ExampleResult) -> None:
        """Save one scored example immediately, so a crash loses at most the examples in flight."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, r.example.id, _j(r.example.input), _j(r.example.expected),
                    _j(r.example.context), _j(r.example.metadata), r.output.text,
                    _j(r.output.context), r.output.error, r.output.latency_ms,
                    _j({k: [s.value, s.reason] for k, s in r.scores.items()}),
                ),
            )

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, name, created_at, summary, config FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2], "summary": json.loads(r[3]),
                 "status": json.loads(r[4] or "{}").get("status", "complete")} for r in rows]

    def resolve(self, ref: str) -> str:
        """Run id, unique id prefix, or a run name (latest run with that name)."""
        if self._db.execute("SELECT 1 FROM runs WHERE id = ?", (ref,)).fetchone():
            return ref
        row = self._db.execute(
            "SELECT id FROM runs WHERE name = ? ORDER BY created_at DESC LIMIT 1", (ref,)
        ).fetchone()
        if row:
            return row[0]
        rows = self._db.execute("SELECT id FROM runs WHERE id LIKE ?", (ref + "%",)).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        raise KeyError(f"No unique run matches {ref!r}")

    def load(self, ref: str) -> RunResult:
        run_id = self.resolve(ref)
        rid, name, created, config, summary = self._db.execute(
            "SELECT id, name, created_at, config, summary FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        results = []
        for row in self._db.execute(
            "SELECT example_id, input, expected, context, metadata, output, output_context, error, latency_ms, scores "
            "FROM results WHERE run_id = ? ORDER BY example_id",
            (run_id,),
        ):
            ex = Example(row[0], json.loads(row[1]), json.loads(row[2]), json.loads(row[3]), json.loads(row[4]))
            out = Output(row[5], json.loads(row[6]), row[8] or 0.0, row[7])
            scores = {k: Score(k, v[0], v[1]) for k, v in json.loads(row[9]).items()}
            results.append(ExampleResult(ex, out, scores))
        return RunResult(rid, name, created, results, json.loads(summary), json.loads(config))
