from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .core import Filter, Hit, Retriever


def rrf(rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion: score = Σ w / (k + rank). Robust to retrievers whose
    raw scores are on different scales (BM25 vs cosine)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, Hit] = {}
    for w, ranking in zip(weights, rankings):
        for rank, h in enumerate(ranking, start=1):
            cur = fused.get(h.id)
            if cur is None:
                cur = fused[h.id] = Hit(h.id, h.text, 0.0, h.metadata, {})
            cur.score += w / (k + rank)
            cur.scores.update(h.scores)
    for h in fused.values():
        h.scores["rrf"] = h.score
    return sorted(fused.values(), key=lambda h: -h.score)


class HybridRetriever(Retriever):
    """Run several retrievers and fuse their rankings with RRF."""

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None, rrf_k: int = 60, fetch_k: int = 50):
        self.retrievers = retrievers
        self.weights = weights
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def retrieve(self, query: str, k: int = 5, filters: Filter = None) -> list[Hit]:
        # Sub-retrievers are independent (e.g. BM25 and a dense search waiting on an
        # embedding call), so run them at the same time.
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as pool:
            rankings = list(pool.map(lambda r: r.retrieve(query, max(k, self.fetch_k), filters), self.retrievers))
        return rrf(rankings, self.rrf_k, self.weights)[:k]
