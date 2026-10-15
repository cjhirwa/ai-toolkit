"""Query transforms turn one user query into several search queries."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class QueryTransform(Protocol):
    def __call__(self, query: str) -> list[str]: ...


class _Queries(BaseModel):
    queries: list[str]


class MultiQuery:
    """Paraphrase the query ``n`` ways (synonyms, expanded abbreviations, other phrasings)."""

    def __init__(self, llm=None, n: int = 3, domain: str | None = None):
        if llm is None:
            from llmgw import LLM

            llm = LLM(temperature=0.3)
        self.llm, self.n, self.domain = llm, n, domain

    def __call__(self, query: str) -> list[str]:
        ctx = f" The domain is {self.domain}." if self.domain else ""
        prompt = (
            f"Write {self.n} alternative search queries for the question below. Vary vocabulary: "
            f"use synonyms, expand abbreviations, and include the formal/technical term where relevant.{ctx}\n\n"
            f"Question: {query}"
        )
        return self.llm.structured(prompt, _Queries).queries[: self.n]


class HyDE:
    """Hypothetical Document Embeddings: write a plausible answer passage and search with it.
    Helps when user questions are phrased very differently from the documents."""

    def __init__(self, llm=None, domain: str | None = None, keep_original: bool = False):
        if llm is None:
            from llmgw import LLM

            llm = LLM(temperature=0)
        self.llm, self.domain, self.keep_original = llm, domain, keep_original

    def __call__(self, query: str) -> list[str]:
        ctx = f" from {self.domain} guidance" if self.domain else ""
        passage = self.llm.text(
            f"Write a short factual passage (3-5 sentences){ctx} that would answer this question. "
            f"Write it as reference material, not as a reply.\n\nQuestion: {query}"
        )
        return [passage]
