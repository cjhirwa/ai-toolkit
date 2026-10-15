from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from .core import Filter, Hit, Retriever
from .fusion import rrf
from .rerank import Reranker
from .transforms import QueryTransform


class Pipeline(Retriever):
    """query → (transforms) → retriever per query → RRF fuse → (rerank) → top k."""

    name = "pipeline"

    def __init__(
        self,
        retriever: Retriever,
        transforms: Sequence[QueryTransform] = (),
        reranker: Reranker | None = None,
        *,
        fetch_k: int = 30,
        rrf_k: int = 60,
        include_original: bool = True,
    ):
        self.retriever = retriever
        self.transforms = list(transforms)
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self.include_original = include_original

    def queries(self, query: str) -> list[str]:
        qs = [query] if self.include_original or not self.transforms else []
        for t in self.transforms:
            qs.extend(t(query))
        return list(dict.fromkeys(q for q in qs if q.strip()))

    def retrieve(self, query: str, k: int = 5, filters: Filter = None) -> list[Hit]:
        qs = self.queries(query)
        fetch = max(k, self.fetch_k)
        # Each rewritten query is an independent search; run them concurrently.
        with ThreadPoolExecutor(max_workers=max(1, len(qs))) as pool:
            rankings = list(pool.map(lambda q: self.retriever.retrieve(q, fetch, filters), qs))
        hits = rankings[0] if len(rankings) == 1 else rrf(rankings, self.rrf_k)
        if self.reranker is not None:
            return self.reranker.rerank(query, hits[:fetch], k)
        return hits[:k]
