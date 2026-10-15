from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]
    metadata: list[dict]
    vectors: np.ndarray  # L2-normalised, shape (n, d)


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def load_corpus(path: str | Path, llm=None, embed_model: str | None = None) -> Corpus:
    """A retrieval-kit index directory (reuses its embeddings) or a chunks.jsonl (embeds it)."""
    from retrievekit import DenseRetriever, Index, load_docs

    p = Path(path)
    if p.is_dir():
        dense = Index.load(p, llm=llm).dense
    else:
        dense = DenseRetriever.build(load_docs(p), llm=llm, embed_model=embed_model)
    docs = dense.docs
    return Corpus([d.id for d in docs], [d.text for d in docs], [d.metadata for d in docs], dense.embeddings)


def embed_queries(texts: list[str], llm=None, embed_model: str | None = None) -> np.ndarray:
    if llm is None:
        from llmgw import LLM

        llm = LLM()
    return _normalise(np.array(llm.embed(texts, model=embed_model, task="query")))


def load_queries(path: str | Path) -> list[str]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                q = r.get("input") or r.get("query") or r.get("question")
                if isinstance(q, str):
                    out.append(q)
    return out


def reduce(vectors: np.ndarray, method: str = "auto", seed: int = 0) -> np.ndarray:
    """2-D projection. ``auto`` uses UMAP when installed, else PCA."""
    if method in ("auto", "umap"):
        try:
            import umap

            n = len(vectors)
            return umap.UMAP(n_components=2, metric="cosine", random_state=seed, n_neighbors=min(15, max(2, n - 1))).fit_transform(vectors)
        except ImportError:
            if method == "umap":
                raise
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, metric="cosine", random_state=seed, perplexity=min(30, max(2, len(vectors) // 4))).fit_transform(vectors)
    centred = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def kmeans(vectors: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    """Spherical k-means (cosine), no sklearn needed."""
    rng = np.random.default_rng(seed)
    k = min(k, len(vectors))
    centres = vectors[rng.choice(len(vectors), k, replace=False)]
    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(iters):
        new = np.argmax(vectors @ centres.T, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            members = vectors[labels == j]
            if len(members):
                c = members.mean(axis=0)
                centres[j] = c / max(float(np.linalg.norm(c)), 1e-12)
    return labels


def near_duplicates(corpus: Corpus, threshold: float = 0.95, block: int = 2048) -> list[tuple[str, str, float]]:
    """Pairs of chunks with cosine similarity ≥ threshold, highest first."""
    v = corpus.vectors
    pairs = []
    for start in range(0, len(v), block):
        sims = v[start : start + block] @ v.T
        for i, j in zip(*np.where(sims >= threshold)):
            a, b = start + int(i), int(j)
            if a < b:
                pairs.append((corpus.ids[a], corpus.ids[b], float(sims[i, j])))
    return sorted(pairs, key=lambda p: -p[2])


@dataclass
class CoverageRow:
    query: str
    best_similarity: float
    nearest_id: str
    nearest_text: str


def coverage(corpus: Corpus, queries: list[str], qvecs: np.ndarray) -> list[CoverageRow]:
    """For each query, its closest chunk. Low best-similarity = likely gap in the corpus."""
    sims = qvecs @ corpus.vectors.T
    best = sims.argmax(axis=1)
    rows = [CoverageRow(q, float(sims[i, best[i]]), corpus.ids[best[i]], corpus.texts[best[i]][:200]) for i, q in enumerate(queries)]
    return sorted(rows, key=lambda r: r.best_similarity)
