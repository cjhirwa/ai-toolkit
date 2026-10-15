from __future__ import annotations

import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .router import Decision, Router


def load_labeled(path: str | Path, text_col: str, label_col: str, limit: int | None = None) -> list[tuple[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = [(r[text_col], r[label_col]) for r in csv.DictReader(f) if r.get(text_col) and r.get(label_col)]
    return rows[:limit] if limit else rows


def evaluate(router: Router, rows: list[tuple[str, str]], concurrency: int = 8) -> dict:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        decisions: list[Decision] = list(pool.map(lambda r: router.classify(r[0]), rows))
    labels = sorted({g for _, g in rows} | {d.route for d in decisions})
    confusion: dict[str, Counter] = defaultdict(Counter)
    for (_, gold), d in zip(rows, decisions):
        confusion[gold][d.route] += 1
    per_label = {}
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in labels if g != lab)
        fn = sum(c for p, c in confusion[lab].items() if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "support": tp + fn}
    correct = sum(1 for (_, g), d in zip(rows, decisions) if g == d.route)
    errors = [(t, g, d) for (t, g), d in zip(rows, decisions) if g != d.route]
    return {
        "accuracy": correct / len(rows) if rows else 0.0,
        "per_label": per_label,
        "confusion": {g: dict(c) for g, c in confusion.items()},
        "stages": dict(Counter(d.stage for d in decisions)),
        "errors": errors,
    }


def format_report(r: dict, show_errors: int = 10) -> str:
    lines = [f"accuracy {r['accuracy']:.3f}   stages {r['stages']}", "", f"{'label':28}{'prec':>8}{'recall':>8}{'n':>6}"]
    for lab, m in r["per_label"].items():
        lines.append(f"{lab:28}{m['precision']:>8.2f}{m['recall']:>8.2f}{m['support']:>6}")
    if r["errors"] and show_errors:
        lines += ["", "errors:"]
        for text, gold, d in r["errors"][:show_errors]:
            lines.append(f"  gold={gold} got={d.route} ({d.stage}, {d.confidence:.2f}): {text[:100]}")
    return "\n".join(lines)
