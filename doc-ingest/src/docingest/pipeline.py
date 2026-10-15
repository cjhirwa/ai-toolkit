from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .chunking import Chunker, RecursiveChunker
from .loaders import iter_files, load
from .models import Chunk, Document
from .ocr import OCRBackend

log = logging.getLogger("docingest")


def chunk_document(doc: Document, chunker: Chunker) -> list[Chunk]:
    chunks = []
    ocr_pages = set(doc.metadata.get("ocr_pages") or ())
    for i, (start, end, meta) in enumerate(chunker.split(doc.text)):
        meta = dict(meta)
        prefix = meta.pop("prefix", None)
        body = doc.text[start:end]
        page, page_end = doc.page_at(start), doc.page_at(max(start, end - 1))
        chunks.append(
            Chunk(
                id=f"{doc.id}-{i:04d}",
                doc_id=doc.id,
                text=f"{prefix}\n\n{body}" if prefix else body,
                metadata={
                    "source": doc.metadata.get("source", doc.source),
                    "title": doc.metadata.get("title"),
                    "chunk_index": i,
                    "start": start,
                    "end": end,
                    **({"page": page} if page is not None else {}),
                    **({"page_end": page_end} if page_end not in (None, page) else {}),
                    # Marks text read by OCR, so it can be reviewed or filtered.
                    **({"ocr": True} if page is not None and ocr_pages & set(range(page, (page_end or page) + 1)) else {}),
                    **meta,
                },
            )
        )
    return chunks


def ingest(
    paths: Iterable[str | Path],
    chunker: Chunker | None = None,
    ocr: OCRBackend | None = None,
    *,
    pattern: str = "**/*",
    dedupe: bool = True,
    skip_errors: bool = True,
) -> list[Chunk]:
    """Load every supported file under ``paths`` and chunk it.

    Each chunk's ``source`` is the file's path relative to the folder it was found in
    (``guides/manual.pdf``), or just its name when the file itself was passed, so
    citations do not depend on where the command was run from.
    """
    chunker = chunker or RecursiveChunker()
    out: list[Chunk] = []
    seen: set[str] = set()
    max_mb = float(os.getenv("DOCINGEST_MAX_FILE_MB", "200"))
    for root, f in _iter_with_roots(paths, pattern):
        try:
            # Guard against huge or hostile uploads before reading them into memory.
            size_mb = f.stat().st_size / 1_048_576
            if max_mb > 0 and size_mb > max_mb:
                raise ValueError(f"{size_mb:.0f} MB is over DOCINGEST_MAX_FILE_MB={max_mb:g}")
            doc = load(f, ocr)
            doc.metadata["source"] = f.relative_to(root).as_posix() if root.is_dir() else f.name
        except Exception as e:
            if not skip_errors:
                raise
            log.warning("skipping %s: %s", f, e)
            continue
        if not doc.text:
            log.warning("no text in %s", f)
            continue
        for c in chunk_document(doc, chunker):
            key = hashlib.sha1(" ".join(c.text.split()).lower().encode()).hexdigest()
            if dedupe and key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out


def _iter_with_roots(paths: Iterable[str | Path], pattern: str) -> Iterator[tuple[Path, Path]]:
    for root in map(Path, paths):
        for f in iter_files([root], pattern):
            yield root, f


def write_jsonl(chunks: Iterable[Chunk], path: str | Path) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Chunk.from_dict(json.loads(line))
