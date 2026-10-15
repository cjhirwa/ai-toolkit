from __future__ import annotations

import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import Example
from .metrics import LLMJudge, Metric, Output, Score, resolve_metrics

Task = Callable[[Example], Any]


@dataclass
class ExampleResult:
    example: Example
    output: Output
    scores: dict[str, Score]


@dataclass
class RunResult:
    run_id: str
    name: str
    created_at: str
    results: list[ExampleResult]
    summary: dict[str, float | None]
    config: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Run {self.name} ({self.run_id}) — {len(self.results)} examples"]
        for k, v in self.summary.items():
            lines.append(f"  {k:24} {'n/a' if v is None else f'{v:.3f}'}")
        return "\n".join(lines)

    def failures(self, metric: str, below: float = 0.5) -> list[ExampleResult]:
        return [
            r for r in self.results
            if (s := r.scores.get(metric)) and s.value is not None and s.value < below
        ]


def _to_output(raw: Any) -> Output:
    if isinstance(raw, Output):
        return raw
    if isinstance(raw, dict):
        return Output(
            text=str(raw.get("output", raw.get("text", ""))),
            context=raw.get("context"),
            extra={k: v for k, v in raw.items() if k not in ("output", "text", "context")},
        )
    return Output(text="" if raw is None else str(raw))


def summarize(results: list[ExampleResult]) -> dict[str, float | None]:
    names = list(dict.fromkeys(n for r in results for n in r.scores))
    summary: dict[str, float | None] = {}
    for n in names:
        vals = [v for r in results if n in r.scores and (v := r.scores[n].value) is not None]
        summary[n] = sum(vals) / len(vals) if vals else None
        # How many examples the mean is over: failed judge calls are left out, so two runs
        # can average over different sets. compare() warns when these differ.
        summary[f"n_{n}"] = float(len(vals))
    lat = sorted(r.output.latency_ms for r in results)
    summary["error_rate"] = sum(1 for r in results if r.output.error) / len(results) if results else None
    summary["latency_ms_p50"] = statistics.median(lat) if lat else None
    summary["latency_ms_p95"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None
    return summary


def _run_metadata(judge_llm, uses_judge: bool) -> dict[str, Any]:
    """What a run needs to be explained later: the models involved and the code version."""
    from llmgw import default_embed_model, default_model

    meta: dict[str, Any] = {"default_model": default_model(), "embed_model": default_embed_model()}
    if uses_judge:
        judge = judge_llm.model if judge_llm is not None else (os.getenv("EVALH_JUDGE_MODEL") or default_model())
        meta["judge_model"] = judge
        meta["same_model_judge"] = judge == meta["default_model"]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            meta["git_commit"] = sha.stdout.strip()
            meta["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return meta


def run_eval(
    task: Task,
    dataset: list[Example],
    metrics: list[str | Metric],
    *,
    name: str | None = None,
    concurrency: int = 4,
    store: Any = "evals.db",
    config: dict[str, Any] | None = None,
    judge_llm=None,
    progress: bool = True,
    resume: str | None = None,
) -> RunResult:
    """Run ``task`` on every example and score it.

    ``task(example)`` may return a string, ``{"output": ..., "context": [...]}``
    (context feeds the faithfulness judge), or an :class:`Output`.
    ``store`` is a path to a SQLite file, a :class:`Store`, or None to skip saving.
    Each result is saved as soon as it is scored. ``resume`` (a run id or name in the
    store) continues that run: examples already scored are kept, and missing ones or ones
    whose task or metric failed are run again under the same run id.
    """
    from .store import Store

    db = None if store is None else (store if isinstance(store, Store) else Store(Path(store)))
    resolved = resolve_metrics(metrics, judge_llm)
    meta = _run_metadata(judge_llm, any(isinstance(m, LLMJudge) for m in resolved))
    if meta.get("same_model_judge"):
        warnings.warn(
            f"The judge ({meta['judge_model']}) is the same model as the default model the task uses. "
            "A model grading its own answers inflates scores; set EVALH_JUDGE_MODEL to a different, "
            "preferably stronger, model.", stacklevel=2)
    kept: dict[str, ExampleResult] = {}
    if resume is not None:
        if db is None:
            raise ValueError("resume needs a store")
        previous = db.load(resume)
        run_id, created_at, run_name = previous.run_id, previous.created_at, name or previous.name
        kept = {r.example.id: r for r in previous.results
                if not r.output.error and all(s.value is not None for s in r.scores.values())}
    else:
        run_id, created_at = uuid.uuid4().hex[:10], datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_name = name or str(getattr(task, "__name__", "run"))
    todo = [ex for ex in dataset if ex.id not in kept]
    run_config = {"metrics": [m.name for m in resolved], **meta, **(config or {})}
    if db is not None:
        db.save_header(RunResult(run_id, run_name, created_at, [], {}, {**run_config, "status": "running"}))
    done = 0
    lock = threading.Lock()

    def one(ex: Example) -> ExampleResult:
        nonlocal done
        t0 = time.perf_counter()
        try:
            out = _to_output(task(ex))
        except Exception as e:  # noqa: BLE001
            out = Output("", error=f"{type(e).__name__}: {e}")
        out.latency_ms = out.latency_ms or (time.perf_counter() - t0) * 1000
        scores: dict[str, Score] = {}
        for m in resolved:
            if out.error:
                scores[m.name] = Score(m.name, 0.0, "task error")
                continue
            try:
                scores[m.name] = m(ex, out)
            except Exception as e:  # noqa: BLE001
                scores[m.name] = Score(m.name, None, f"metric error: {type(e).__name__}: {e}")
        result = ExampleResult(ex, out, scores)
        if db is not None:
            db.add_result(run_id, result)
        with lock:
            done += 1
            if progress:
                print(f"\r[{done + len(kept)}/{len(dataset)}]", end="", file=sys.stderr, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        fresh = {r.example.id: r for r in pool.map(one, todo)}
    if progress:
        print(file=sys.stderr)

    results = [kept.get(ex.id) or fresh[ex.id] for ex in dataset]
    run = RunResult(run_id, run_name, created_at, results, summarize(results),
                    {**run_config, "status": "complete", **({"resumed": True} if resume else {})})
    if db is not None:
        db.save_header(run)
    return run
