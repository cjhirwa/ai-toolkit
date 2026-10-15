"""Retrieval benchmarks: recall@k, hit@k, MRR and nDCG@k over labelled queries."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .core import Retriever


def load_qrels(path: str | Path) -> list[dict]:
    """Rows of ``{"query": ..., "relevant_ids": [...], "relevant_spans": [...]}``. Also accepts
    eval-harness / synth-data rows: ``{"input": ..., "metadata": {"chunk_ids": [...], "spans": [...]}}``.
    A span is ``{"source", "start", "end"}`` in the extracted document text."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            query = r.get("query") or r.get("input")
            meta = r.get("metadata") or {}
            rel = r.get("relevant_ids") or meta.get("chunk_ids") or r.get("chunk_ids") or []
            spans = r.get("relevant_spans") or meta.get("spans") or []
            if query and (rel or spans):
                rows.append({"query": query, "relevant_ids": list(rel), **({"relevant_spans": list(spans)} if spans else {})})
    return rows


def _overlaps(hit_meta: dict, span: dict) -> bool:
    """Same source, and the hit covers at least half of the shorter of the two ranges.
    A neighbouring chunk that only shares the overlap margin does not count."""
    if hit_meta.get("source") != span.get("source") or hit_meta.get("start") is None or hit_meta.get("end") is None:
        return False
    a0, a1, b0, b1 = hit_meta["start"], hit_meta["end"], span["start"], span["end"]
    shared = min(a1, b1) - max(a0, b0)
    return shared > 0 and shared >= 0.5 * min(a1 - a0, b1 - b0)


def evaluate(retriever: Retriever, qrels: list[dict], k: int = 10) -> dict[str, float]:
    recall = hit = mrr = ndcg = 0.0
    latency = 0.0
    for row in qrels:
        t0 = time.perf_counter()
        hits = retriever.retrieve(row["query"], k)
        latency += time.perf_counter() - t0
        spans = row.get("relevant_spans")
        if spans and all(h.metadata.get("start") is not None for h in hits):
            # Judge by position in the document, so any chunking can be scored.
            # A hit is relevant if it covers an answer span no earlier hit covered: with small
            # chunks several hits can cover one span, and each span may only count once.
            seen: set[int] = set()
            found = []
            for i, h in enumerate(hits):
                new = {j for j, s in enumerate(spans) if j not in seen and _overlaps(h.metadata, s)}
                if new:
                    found.append(i)
                    seen |= new
            rel = spans
            recall += len(seen) / len(spans)
        else:
            rel = set(row["relevant_ids"])
            ids = [h.id for h in hits]
            found = [i for i, d in enumerate(ids) if d in rel]
            recall += len({ids[i] for i in found}) / len(rel) if rel else 0.0
        hit += 1.0 if found else 0.0
        mrr += 1.0 / (found[0] + 1) if found else 0.0
        dcg = sum(1.0 / math.log2(i + 2) for i in found)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
        ndcg += dcg / idcg if idcg else 0.0
    n = max(len(qrels), 1)
    return {
        f"recall@{k}": recall / n,
        f"hit@{k}": hit / n,
        "mrr": mrr / n,
        f"ndcg@{k}": ndcg / n,
        "latency_ms": latency / n * 1000,
    }


def benchmark(retrievers: dict[str, Retriever], qrels: list[dict], k: int = 10) -> dict[str, dict[str, float]]:
    return {name: evaluate(r, qrels, k) for name, r in retrievers.items()}


def format_table(results: dict[str, dict[str, float]]) -> str:
    if not results:
        return ""
    cols = list(next(iter(results.values())))
    lines = [f"{'retriever':28}" + "".join(f"{c:>12}" for c in cols)]
    for name, m in results.items():
        lines.append(f"{name:28}" + "".join(f"{m[c]:>12.3f}" if c != "latency_ms" else f"{m[c]:>12.0f}" for c in cols))
    return "\n".join(lines)
