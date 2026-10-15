"""An index directory: docs.jsonl + embeddings.npy + index.json.

    idx = Index.build("chunks.jsonl", "index/")
    idx = Index.load("index/")
    idx.retriever("hybrid").retrieve("how do I reset the thermostat", k=5)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import Doc, Retriever, load_docs
from .fusion import HybridRetriever
from .pipeline import Pipeline
from .retrievers import BM25Retriever, DenseRetriever


class Index:
    def __init__(self, dense: DenseRetriever):
        self.dense = dense
        self.docs = dense.docs
        self._bm25: BM25Retriever | None = None
        self._by_id = {d.id: d for d in self.docs}

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.docs)
        return self._bm25

    def get(self, doc_id: str) -> Doc | None:
        return self._by_id.get(doc_id)

    @classmethod
    def build(cls, chunks_path: str | Path, directory: str | Path, llm=None, embed_model: str | None = None) -> Index:
        dense = DenseRetriever.build(load_docs(chunks_path), llm=llm, embed_model=embed_model)
        dense.save(directory)
        return cls(dense)

    @classmethod
    def load(cls, directory: str | Path, llm=None) -> Index:
        d = Path(directory)
        meta = json.loads((d / "index.json").read_text())
        docs = load_docs(d / "docs.jsonl")
        vectors = np.load(d / "embeddings.npy")
        # A mismatch would make hits return the wrong passage text, so refuse to load.
        problems = []
        if vectors.ndim != 2 or len(vectors) != len(docs):
            problems.append(f"{len(docs)} passages but {vectors.shape[0] if vectors.ndim else 0} vectors")
        if "count" in meta and meta["count"] != len(docs):
            problems.append(f"index.json says {meta['count']} passages, docs.jsonl has {len(docs)}")
        if "dim" in meta and vectors.ndim == 2 and meta["dim"] != vectors.shape[1]:
            problems.append(f"index.json says dimension {meta['dim']}, vectors have {vectors.shape[1]}")
        if problems:
            raise ValueError(f"Index at {d} is inconsistent ({'; '.join(problems)}); rebuild it with retrievekit build")
        return cls(DenseRetriever(docs, vectors, llm=llm, embed_model=meta["embed_model"]))

    def retriever(self, mode: str = "hybrid", llm=None) -> Retriever:
        """Modes: bm25, dense, hybrid, and hybrid with suffixes ``+mq`` (multi-query),
        ``+hyde``, ``+llmrerank``, ``+cerank`` (cross-encoder), e.g. ``hybrid+mq+llmrerank``."""
        base, *extras = mode.split("+")
        r: Retriever = {"bm25": lambda: self.bm25, "dense": lambda: self.dense,
                        "hybrid": lambda: HybridRetriever([self.bm25, self.dense])}[base]()
        if not extras:
            return r
        from .rerank import CrossEncoderReranker, LLMReranker, Reranker
        from .transforms import HyDE, MultiQuery, QueryTransform

        transforms: list[QueryTransform] = []
        reranker: Reranker | None = None
        for e in extras:
            if e == "mq":
                transforms.append(MultiQuery(llm))
            elif e == "hyde":
                transforms.append(HyDE(llm))
            elif e == "llmrerank":
                reranker = LLMReranker(llm)
            elif e == "cerank":
                reranker = CrossEncoderReranker()
            else:
                raise ValueError(f"Unknown mode suffix {e!r}")
        return Pipeline(r, transforms, reranker)
