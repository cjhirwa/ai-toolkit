from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

Filter = Union[dict[str, Any], Callable[[dict[str, Any]], bool], None]


@dataclass
class Doc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)  # per-retriever / reranker scores

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": round(self.score, 5), "text": self.text, "metadata": self.metadata, "scores": self.scores}


def matches(meta: dict[str, Any], flt: Filter) -> bool:
    """dict filters: every key must equal the value, or be in it when the value is a list/set."""
    if flt is None:
        return True
    if callable(flt):
        return bool(flt(meta))
    for k, v in flt.items():
        got = meta.get(k)
        if isinstance(v, (list, tuple, set)):
            if got not in v:
                return False
        elif got != v:
            return False
    return True


class FilterMask:
    """Vectorised dict filters over a fixed list of documents. Each metadata field used in a
    filter becomes a cached array, so a filter costs a few NumPy comparisons instead of a
    Python call per document. Callable filters still run per document."""

    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self._columns: dict[str, np.ndarray] = {}

    def _column(self, key: str) -> np.ndarray:
        if key not in self._columns:
            col = np.empty(len(self.docs), dtype=object)
            col[:] = [d.metadata.get(key) for d in self.docs]
            self._columns[key] = col
        return self._columns[key]

    def __call__(self, flt: Filter) -> np.ndarray | None:
        """Boolean mask of documents passing ``flt``, or None for no filter."""
        if flt is None:
            return None
        if callable(flt):
            return np.fromiter((bool(flt(d.metadata)) for d in self.docs), dtype=bool, count=len(self.docs))
        mask = np.ones(len(self.docs), dtype=bool)
        for k, v in flt.items():
            col = self._column(k)
            if isinstance(v, (list, tuple, set)):
                allowed = set(v)
                if len(allowed) <= 8:  # a few vectorised comparisons beat a Python loop
                    hit = np.zeros(len(col), dtype=bool)
                    for a in allowed:
                        hit |= col == a
                else:
                    hit = np.fromiter((x in allowed for x in col), dtype=bool, count=len(col))
                mask &= hit
            else:
                mask &= col == v
        return mask


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first, without sorting everything."""
    if k >= len(scores):
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def load_docs(path: str | Path) -> list[Doc]:
    """Read chunks.jsonl as produced by doc-ingest ({id, text, metadata})."""
    docs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs.append(Doc(str(d["id"]), d["text"], d.get("metadata") or {}))
    return docs


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, filters: Filter = None) -> list[Hit]: ...

    def __call__(self, query: str, k: int = 5, filters: Filter = None) -> list[Hit]:
        return self.retrieve(query, k, filters)

    def _mask(self, filters: Filter) -> np.ndarray:
        """Filter mask over ``self.docs`` (all True for no filter), with field columns cached across queries."""
        fm = self.__dict__.get("_filter_mask")
        if fm is None:
            fm = self.__dict__["_filter_mask"] = FilterMask(self.docs)  # type: ignore[attr-defined]
        mask = fm(filters)
        return mask if mask is not None else np.ones(len(fm.docs), dtype=bool)
