"""docingest — documents in, traceable chunks out."""

from .chunking import Chunker, FixedChunker, HeadingChunker, RecursiveChunker, SemanticChunker, get_chunker
from .loaders import iter_files, load
from .models import Chunk, Document, PageSpan
from .ocr import LLMOCR, OCRBackend, TesseractOCR, get_ocr
from .pipeline import chunk_document, ingest, read_jsonl, write_jsonl

__all__ = [
    "Chunk", "Document", "PageSpan", "load", "iter_files",
    "Chunker", "RecursiveChunker", "FixedChunker", "HeadingChunker", "SemanticChunker", "get_chunker",
    "OCRBackend", "TesseractOCR", "LLMOCR", "get_ocr",
    "ingest", "chunk_document", "write_jsonl", "read_jsonl",
]
