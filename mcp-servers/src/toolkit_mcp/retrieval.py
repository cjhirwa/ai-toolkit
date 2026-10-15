"""Search a retrieval-kit index.

Env: RETRIEVAL_INDEX_DIR (default ./index), RETRIEVAL_MODE (default hybrid), plus the
llmgw provider keys used to embed queries."""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from ._run import run

mcp = MCPServer("retrieval")


@lru_cache(maxsize=1)
def _index():
    from retrievekit import Index

    return Index.load(os.getenv("RETRIEVAL_INDEX_DIR", "data/index"))


@lru_cache(maxsize=8)
def _retriever(mode: str):
    return _index().retriever(mode)


def _hit(h, max_chars: int) -> dict:
    return {
        "id": h.id,
        "score": round(h.score, 4),
        "source": h.metadata.get("source"),
        "page": h.metadata.get("page"),
        "heading": h.metadata.get("heading_path"),
        "text": h.text[:max_chars],
    }


@mcp.tool()
def search(query: str, k: int = 5, source: str | None = None, mode: str | None = None, max_chars: int = 1500) -> list[dict]:
    """Search the document library and return the most relevant passages with source and page.

    Args:
        query: Natural-language search query.
        k: Number of passages to return (1-20).
        source: Only search this source file (exact path as shown by list_sources).
        mode: bm25, dense, hybrid (default), or hybrid+llmrerank for slower but sharper ranking.
        max_chars: Truncate each passage to this many characters.
    """
    retriever = _retriever(mode or os.getenv("RETRIEVAL_MODE", "hybrid"))
    hits = retriever.retrieve(query, max(1, min(k, 20)), {"source": source} if source else None)
    return [_hit(h, max_chars) for h in hits]


@mcp.tool()
def get_passage(id: str) -> dict:
    """Fetch one passage in full by id (as returned by search)."""
    d = _index().get(id)
    if d is None:
        return {"error": f"no passage {id!r}"}
    return {"id": d.id, "text": d.text, "metadata": d.metadata}


@mcp.tool()
def list_sources() -> list[dict]:
    """List the documents in the index with their passage counts."""
    counts = Counter(d.metadata.get("source") for d in _index().docs)
    return [{"source": s, "passages": n} for s, n in counts.most_common()]


def main() -> None:
    run(mcp)


if __name__ == "__main__":
    main()
