from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .core import Doc, Filter, Hit, Retriever, top_k

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever(Retriever):
    """BM25 with Lucene's always-positive IDF, over an inverted index."""

    name = "bm25"

    def __init__(self, docs: list[Doc], k1: float = 1.2, b: float = 0.75, stopwords: set[str] | None = None):
        self.docs = docs
        self.k1, self.b = k1, b
        self.stopwords = stopwords or set()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            toks = self._tok(d.text)
            lengths[i] = len(toks)
            for t, tf in Counter(toks).items():
                self.postings.setdefault(t, []).append((i, tf))
        self.norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()) if len(docs) else 1.0, 1e-9))

    def _tok(self, text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in self.stopwords]

    def scores(self, query: str) -> np.ndarray:
        n = len(self.docs)
        scores = np.zeros(n, dtype=np.float32)
        for t in set(self._tok(query)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            idx = np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.norm[idx])
        return scores

    def retrieve(self, query: str, k: int = 5, filters: Filter = None) -> list[Hit]:
        scores = self.scores(query)
        if filters is not None:
            scores = np.where(self._mask(filters), scores, -np.inf)
        top = top_k(scores, k)
        return [
            Hit(self.docs[i].id, self.docs[i].text, float(scores[i]), self.docs[i].metadata, {"bm25": float(scores[i])})
            for i in top
            if scores[i] > 0
        ]


class DenseRetriever(Retriever):
    """Cosine similarity over embeddings held in memory (fine up to a few hundred thousand chunks)."""

    name = "dense"

    def __init__(self, docs: list[Doc], embeddings: np.ndarray, llm=None, embed_model: str | None = None):
        if llm is None:
            from llmgw import LLM

            llm = LLM()
        self.docs = docs
        self.llm = llm
        self.embed_model = embed_model or llm.embed_model
        e = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)

    @classmethod
    def build(cls, docs: list[Doc], llm=None, embed_model: str | None = None, batch_size: int = 64) -> DenseRetriever:
        if llm is None:
            from llmgw import LLM

            llm = LLM()
        model = embed_model or llm.embed_model
        vecs: list[list[float]] = []
        for i in range(0, len(docs), batch_size):
            vecs.extend(llm.embed([d.text for d in docs[i : i + batch_size]], model=model, task="document"))
        return cls(docs, np.array(vecs), llm, model)

    def embed_query(self, query: str) -> np.ndarray:
        q = np.asarray(self.llm.embed([query], model=self.embed_model, task="query")[0], dtype=np.float32)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    def retrieve(self, query: str, k: int = 5, filters: Filter = None, *, vector: np.ndarray | None = None) -> list[Hit]:
        q = self.embed_query(query) if vector is None else vector
        sims = self.embeddings @ q
        if filters is not None:
            sims = np.where(self._mask(filters), sims, -np.inf)
        top = top_k(sims, k)
        return [
            Hit(self.docs[i].id, self.docs[i].text, float(sims[i]), self.docs[i].metadata, {"dense": float(sims[i])})
            for i in top
            if np.isfinite(sims[i])
        ]

    def save(self, directory: str | Path) -> None:
        """Write to a sibling temp folder and swap it in only when complete, so an
        interrupted build never leaves texts and vectors that disagree."""
        d = Path(directory)
        tmp = d.with_name(f".{d.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        np.save(tmp / "embeddings.npy", self.embeddings)
        with (tmp / "docs.jsonl").open("w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps({"id": doc.id, "text": doc.text, "metadata": doc.metadata}, ensure_ascii=False) + "\n")
        meta = {"embed_model": self.embed_model, "count": len(self.docs), "dim": int(self.embeddings.shape[1])}
        (tmp / "index.json").write_text(json.dumps(meta))
        old = d.with_name(f".{d.name}.old-{os.getpid()}")
        if d.exists():
            d.rename(old)
        tmp.rename(d)
        shutil.rmtree(old, ignore_errors=True)


class ChromaRetriever(Retriever):
    """Query an existing Chroma collection (e.g. one built by another app) through the same interface."""

    name = "chroma"

    def __init__(self, collection, llm=None, embed_model: str | None = None):
        self.collection = collection
        self.llm = llm
        self.embed_model = embed_model

    @classmethod
    def open(cls, path: str, collection: str, **kw) -> ChromaRetriever:
        import chromadb

        return cls(chromadb.PersistentClient(path=path).get_collection(collection), **kw)

    def retrieve(self, query: str, k: int = 5, filters: Filter = None) -> list[Hit]:
        if callable(filters):
            raise ValueError("ChromaRetriever supports dict filters only")
        kwargs: dict[str, Any] = {"n_results": k, "where": filters or None}
        if self.llm is not None:
            kwargs["query_embeddings"] = self.llm.embed([query], model=self.embed_model, task="query")
        else:
            kwargs["query_texts"] = [query]
        r = self.collection.query(**kwargs)
        hits = []
        for i, doc_id in enumerate(r["ids"][0]):
            sim = 1.0 - float(r["distances"][0][i])
            hits.append(Hit(doc_id, r["documents"][0][i], sim, r["metadatas"][0][i] or {}, {"chroma": sim}))
        return hits
