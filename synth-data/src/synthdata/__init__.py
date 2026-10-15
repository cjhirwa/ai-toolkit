"""synthdata — grounded synthetic eval sets from your own documents."""

from .generate import UNANSWERABLE_EXPECTED, SourceChunk, dedupe, generate, load_chunks, to_qrels, write_jsonl

__all__ = ["generate", "load_chunks", "SourceChunk", "to_qrels", "dedupe", "write_jsonl", "UNANSWERABLE_EXPECTED"]
