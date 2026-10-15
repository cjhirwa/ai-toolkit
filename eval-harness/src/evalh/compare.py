from __future__ import annotations

import random
from dataclasses import dataclass

from .runner import RunResult


@dataclass
class ExampleDelta:
    example_id: str
    metric: str
    before: float | None
    after: float | None

    @property
    def delta(self) -> float:
        return (self.after or 0.0) - (self.before or 0.0)


@dataclass
class PairedDelta:
    mean: float
    low: float
    high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_deltas(a: RunResult, b: RunResult, iterations: int = 2000, seed: int = 0) -> dict[str, PairedDelta]:
    """Paired bootstrap of B - A per metric, over examples both runs scored. Pairing removes
    the variation between examples, so the interval reflects the change itself."""
    by_id = {r.example.id: r for r in a.results}
    diffs: dict[str, list[float]] = {}
    for rb in b.results:
        ra = by_id.get(rb.example.id)
        if not ra:
            continue
        for m, sb in rb.scores.items():
            sa = ra.scores.get(m)
            if sa and sa.value is not None and sb.value is not None:
                diffs.setdefault(m, []).append(sb.value - sa.value)
    rng = random.Random(seed)
    out = {}
    for m, d in diffs.items():
        n = len(d)
        means = sorted(sum(rng.choices(d, k=n)) / n for _ in range(iterations))
        out[m] = PairedDelta(sum(d) / n, means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1], n)
    return out


@dataclass
class Comparison:
    a: RunResult
    b: RunResult
    metric_deltas: dict[str, tuple[float | None, float | None]]
    regressions: list[ExampleDelta]
    improvements: list[ExampleDelta]

    @property
    def paired(self) -> dict[str, PairedDelta]:
        """Per metric, the mean change over examples scored in both runs, with a 95%
        bootstrap confidence interval. Only ``significant`` changes are more than noise."""
        return paired_deltas(self.a, self.b)

    @property
    def n_mismatch(self) -> dict[str, tuple[float, float]]:
        """Metrics whose means cover different numbers of examples in the two runs."""
        return {k[2:]: (x, y) for k, (x, y) in self.metric_deltas.items()
                if k.startswith("n_") and x is not None and y is not None and x != y}

    def __str__(self) -> str:
        lines = [f"{self.a.name} ({self.a.run_id})  →  {self.b.name} ({self.b.run_id})", ""]
        lines.append(f"{'metric':24} {'A':>8} {'B':>8} {'Δ':>8}")
        for k, (x, y) in self.metric_deltas.items():
            fx = "n/a" if x is None else f"{x:.3f}"
            fy = "n/a" if y is None else f"{y:.3f}"
            d = "" if x is None or y is None else f"{y - x:+.3f}"
            lines.append(f"{k:24} {fx:>8} {fy:>8} {d:>8}")
        if self.paired:
            lines.append(f"\n{'paired change (B - A)':24} {'mean':>8} {'95% CI':>17} {'n':>5}")
            for m, p in self.paired.items():
                flag = "  significant" if p.significant else ""
                lines.append(f"{m:24} {p.mean:>+8.3f} [{p.low:+.3f}, {p.high:+.3f}] {p.n:>5}{flag}")
        if self.n_mismatch:
            lines.append("\nWARNING: metrics scored over different numbers of examples: " + ", ".join(
                f"{m} ({a:.0f} vs {b:.0f})" for m, (a, b) in self.n_mismatch.items()) + ". The means are not directly comparable.")
        lines.append(f"\n{len(self.regressions)} regressions, {len(self.improvements)} improvements (per example × metric)")
        for reg in self.regressions[:10]:
            lines.append(f"  ↓ {reg.example_id:20} {reg.metric:20} {reg.before} → {reg.after}")
        return "\n".join(lines)


def compare(a: RunResult, b: RunResult, threshold: float = 0.25) -> Comparison:
    """Compare two runs on the examples they share. Per-example changes of at
    least ``threshold`` count as regressions or improvements."""
    keys = list(dict.fromkeys([*a.summary, *b.summary]))
    metric_deltas = {k: (a.summary.get(k), b.summary.get(k)) for k in keys}
    by_id = {r.example.id: r for r in a.results}
    regressions, improvements = [], []
    for rb in b.results:
        ra = by_id.get(rb.example.id)
        if not ra:
            continue
        for m, sb in rb.scores.items():
            sa = ra.scores.get(m)
            if not sa or sa.value is None or sb.value is None:
                continue
            d = ExampleDelta(rb.example.id, m, sa.value, sb.value)
            if d.delta <= -threshold:
                regressions.append(d)
            elif d.delta >= threshold:
                improvements.append(d)
    regressions.sort(key=lambda d: d.delta)
    improvements.sort(key=lambda d: -d.delta)
    return Comparison(a, b, metric_deltas, regressions, improvements)
