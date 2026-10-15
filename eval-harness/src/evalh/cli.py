from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import sys
import webbrowser
from pathlib import Path

from llmgw.console import utf8_stdio

from .compare import compare
from .dataset import load_dataset
from .report import comparison_report, run_report
from .runner import run_eval
from .store import Store


def load_callable(ref: str):
    """``package.module:fn`` or ``path/to/file.py:fn``."""
    target, _, attr = ref.rpartition(":")
    if not target:
        raise ValueError(f"Expected 'module:function', got {ref!r}")
    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(target).stem, target)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import {target!r}: not a Python file that can be loaded")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(target).parent.resolve()))
        spec.loader.exec_module(mod)
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(target)
    return getattr(mod, attr)


def _write(html_text: str, path: str, open_it: bool) -> None:
    Path(path).write_text(html_text, encoding="utf-8")
    print(f"wrote {path}")
    if open_it:
        webbrowser.open(Path(path).resolve().as_uri())


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="evalh")
    p.add_argument("--db", default=os.getenv("EVALH_DB", "evals.db"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a task over a dataset")
    r.add_argument("--task", required=True, help="module:function or file.py:function")
    r.add_argument("--dataset", required=True)
    r.add_argument("--metrics", default="answer_relevance", help="comma-separated built-ins, or module:obj")
    r.add_argument("--name")
    r.add_argument("--limit", type=int)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--resume", metavar="RUN", help="continue an interrupted run (id or name): only missing or failed examples run")

    sub.add_parser("runs", help="list recent runs")

    s = sub.add_parser("show", help="print a run summary")
    s.add_argument("ref")

    c = sub.add_parser("compare", help="compare two runs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--threshold", type=float, default=0.25)
    c.add_argument("--html")
    c.add_argument("--open", action="store_true")

    rep = sub.add_parser("report", help="HTML report for a run")
    rep.add_argument("ref")
    rep.add_argument("-o", "--out")
    rep.add_argument("--open", action="store_true")

    args = p.parse_args(argv)
    store = Store(args.db)

    if args.cmd == "run":
        metrics = [m if ":" not in m else load_callable(m) for m in args.metrics.split(",") if m]
        run = run_eval(
            load_callable(args.task),
            load_dataset(args.dataset, limit=args.limit),
            metrics,
            name=args.name,
            concurrency=args.concurrency,
            store=store,
            resume=args.resume,
            config={"task": args.task, "dataset": args.dataset,
                    "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()},
        )
        print(run)
    elif args.cmd == "runs":
        for row in store.list_runs():
            main_scores = ", ".join(
                f"{k}={v:.2f}" for k, v in row["summary"].items() if v is not None and not k.startswith(("latency", "error", "n_"))
            )
            status = "" if row["status"] == "complete" else f"  [{row['status']}: resume with --resume {row['id']}]"
            print(f"{row['id']}  {row['created_at']}  {row['name']:24} {main_scores}{status}")
    elif args.cmd == "show":
        print(store.load(args.ref))
    elif args.cmd == "compare":
        cmp = compare(store.load(args.a), store.load(args.b), args.threshold)
        print(cmp)
        if args.html:
            _write(comparison_report(cmp), args.html, args.open)
    elif args.cmd == "report":
        run = store.load(args.ref)
        _write(run_report(run), args.out or f"eval-{run.name}-{run.run_id}.html", args.open)


if __name__ == "__main__":
    main()
