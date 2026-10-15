"""sextract — documents to validated Pydantic records."""

from .core import (
    ExtractionResult,
    extract,
    extract_files,
    extract_long,
    merge_partials,
    partial_model,
    write_csv,
    write_jsonl,
)

__all__ = [
    "extract", "extract_long", "extract_files", "ExtractionResult",
    "partial_model", "merge_partials", "write_jsonl", "write_csv",
]
