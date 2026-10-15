"""Chunkers return character spans ``(start, end, metadata)`` over the document text,
so every chunk can be traced back to its page and heading."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

Span = tuple[int, int, dict[str, Any]]

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


class Chunker(Protocol):
    def split(self, text: str) -> list[Span]: ...


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _atoms(text: str, start: int, end: int, seps: tuple[str, ...], size: int) -> list[tuple[int, int]]:
    if end - start <= size:
        return [(start, end)]
    if not seps or seps[0] == "":
        return [(i, min(i + size, end)) for i in range(start, end, size)]
    sep, rest = seps[0], seps[1:]
    pieces, i = [], start
    while i < end:
        j = text.find(sep, i, end)
        if j == -1:
            pieces.append((i, end))
            break
        pieces.append((i, j + len(sep)))  # separator stays with the left piece
        i = j + len(sep)
    if len(pieces) == 1:
        return _atoms(text, start, end, rest, size)
    out: list[tuple[int, int]] = []
    for a, b in pieces:
        out.extend([(a, b)] if b - a <= size else _atoms(text, a, b, rest, size))
    return out


def _merge(atoms: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    cur: list[tuple[int, int]] = []
    for a in atoms:
        if cur and a[1] - cur[0][0] > size:
            chunks.append((cur[0][0], cur[-1][1]))
            end = cur[-1][1]
            cur = [x for x in cur if end - x[0] <= overlap] if overlap else []
            while cur and a[1] - cur[0][0] > size:
                cur.pop(0)
        cur.append(a)
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


class RecursiveChunker:
    """Split on the coarsest separator that fits (paragraph, line, sentence, word),
    then pack pieces into chunks of at most ``size`` chars with ``overlap``."""

    def __init__(self, size: int = 1000, overlap: int = 150, separators: tuple[str, ...] = DEFAULT_SEPARATORS):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap, self.separators = size, overlap, separators

    def split_range(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans = _merge(_atoms(text, start, end, self.separators, self.size), self.size, self.overlap)
        out = []
        for a, b in spans:
            a, b = _strip(text, a, b)
            if b > a:
                out.append((a, b))
        return out

    def split(self, text: str) -> list[Span]:
        return [(a, b, {}) for a, b in self.split_range(text, 0, len(text))]


class FixedChunker(RecursiveChunker):
    """Fixed-size windows that only break between words."""

    def __init__(self, size: int = 1000, overlap: int = 150):
        super().__init__(size, overlap, separators=(" ", ""))


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


class HeadingChunker:
    """Section-aware: one chunk per Markdown section when it fits, recursive
    splitting inside long sections. Adds ``heading_path`` ("A > B > C") to metadata,
    and with ``prefix_headings`` prepends it to the chunk text for better retrieval."""

    def __init__(self, size: int = 1200, overlap: int = 150, max_level: int = 6, prefix_headings: bool = True):
        self.inner = RecursiveChunker(size, overlap)
        self.max_level = max_level
        self.prefix_headings = prefix_headings

    def split(self, text: str) -> list[Span]:
        heads = [m for m in _HEADING.finditer(text) if len(m.group(1)) <= self.max_level]
        bounds = [(0, None)] + [(m.start(), m) for m in heads]
        stack: list[tuple[int, str]] = []
        out: list[Span] = []
        for i, (start, m) in enumerate(bounds):
            end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
            if m is not None:
                level = len(m.group(1))
                stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2).strip())]
            path = " > ".join(t for _, t in stack)
            parent = " > ".join(t for _, t in stack[:-1])
            for a, b in self.inner.split_range(text, start, end):
                if m is not None and (a, b) == _strip(text, m.start(), m.end()):
                    continue  # a heading with no body
                meta: dict[str, Any] = {"heading_path": path} if path else {}
                # A chunk that opens with its own heading only needs the parent path.
                prefix = parent if m is not None and a == _strip(text, m.start(), m.end())[0] else path
                if self.prefix_headings and prefix:
                    meta["prefix"] = prefix
                out.append((a, b, meta))
        return out


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=\S)|\n\s*\n")


class SemanticChunker:
    """Break where adjacent sentences are least similar (embedding distance above
    the ``breakpoint_percentile``), keeping chunks between ``min_size`` and ``max_size`` chars."""

    def __init__(self, llm=None, max_size: int = 1500, min_size: int = 200, breakpoint_percentile: float = 90):
        if llm is None:
            from llmgw import LLM

            llm = LLM()
        self.llm = llm
        self.max_size, self.min_size, self.pct = max_size, min_size, breakpoint_percentile
        self._fallback = RecursiveChunker(max_size, 0)

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        spans, i = [], 0
        for m in _SENT_END.finditer(text):
            spans.append(_strip(text, i, m.start()))
            i = m.end()
        spans.append(_strip(text, i, len(text)))
        return [(a, b) for a, b in spans if b > a]

    def split(self, text: str) -> list[Span]:
        sents = self._sentences(text)
        if len(sents) < 3:
            return self._fallback.split(text)
        vecs = self.llm.embed([text[a:b] for a, b in sents], task="document")

        def cos(u, v):
            nu, nv = math.sqrt(sum(x * x for x in u)) or 1, math.sqrt(sum(x * x for x in v)) or 1
            return sum(x * y for x, y in zip(u, v)) / (nu * nv)

        dists = [1 - cos(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        ordered = sorted(dists)
        threshold = ordered[min(len(ordered) - 1, int(len(ordered) * self.pct / 100))]
        out: list[Span] = []
        start = sents[0][0]
        for i in range(len(sents) - 1):
            cur_end, nxt_end = sents[i][1], sents[i + 1][1]
            size_now = cur_end - start
            if (dists[i] >= threshold and size_now >= self.min_size) or nxt_end - start > self.max_size:
                out.extend((a, b, {}) for a, b in self._fallback.split_range(text, start, cur_end))
                start = sents[i + 1][0]
        out.extend((a, b, {}) for a, b in self._fallback.split_range(text, start, sents[-1][1]))
        return out


def get_chunker(name: str, size: int = 1000, overlap: int = 150, **kw) -> Chunker:
    if name == "recursive":
        return RecursiveChunker(size, overlap)
    if name == "fixed":
        return FixedChunker(size, overlap)
    if name == "headings":
        return HeadingChunker(size, overlap, **kw)
    if name == "semantic":
        return SemanticChunker(max_size=size, **kw)
    raise ValueError(f"Unknown chunker {name!r} (recursive | fixed | headings | semantic)")
