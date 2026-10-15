from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, get_origin

from pydantic import BaseModel, ValidationError, create_model

log = logging.getLogger("sextract")
T = TypeVar("T", bound=BaseModel)

_SYSTEM = (
    "You extract structured data from documents. Use only information stated in the text. "
    "If a field is not stated, leave it null (or an empty list) rather than guessing. "
    "Copy names, numbers and units exactly as written."
)


def _llm(llm):
    if llm is None:
        from llmgw import LLM

        return LLM(temperature=0)
    return llm


def _prompt(text: str, instructions: str | None) -> str:
    extra = f"\n\nAdditional instructions:\n{instructions}" if instructions else ""
    return f"Extract the data from this document.{extra}\n\n<document>\n{text}\n</document>"


def extract(text: str, schema: type[T], *, llm=None, instructions: str | None = None, max_attempts: int = 3) -> T:
    """One call; invalid JSON or schema violations are sent back to the model to fix."""
    return _llm(llm).structured(_prompt(text, instructions), schema, system=_SYSTEM, max_attempts=max_attempts)


def partial_model(schema: type[BaseModel]) -> type[BaseModel]:
    """Same fields, all optional — used for per-chunk extraction where any chunk may lack a field."""
    fields: dict[str, Any] = {}
    for name, f in schema.model_fields.items():
        ann = f.annotation
        if get_origin(ann) is list:
            fields[name] = (ann, [])
        else:
            fields[name] = (Optional[ann], None)
    return create_model(f"Partial{schema.__name__}", __doc__=schema.__doc__, **fields)


def _is_list_field(schema: type[BaseModel], name: str) -> bool:
    return get_origin(schema.model_fields[name].annotation) is list


def merge_partials(schema: type[BaseModel], partials: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list]]:
    """Lists are concatenated (deduplicated); scalars take the first non-null value.
    Returns the merged dict and any scalar conflicts found."""
    merged: dict[str, Any] = {}
    conflicts: dict[str, list] = {}
    for name in schema.model_fields:
        if _is_list_field(schema, name):
            seen, items = set(), []
            for p in partials:
                for item in p.get(name) or []:
                    key = json.dumps(item, sort_keys=True, default=str)
                    if key not in seen:
                        seen.add(key)
                        items.append(item)
            merged[name] = items
        else:
            values = [p.get(name) for p in partials if p.get(name) not in (None, "", [])]
            distinct = list({json.dumps(v, sort_keys=True, default=str): v for v in values}.values())
            if len(distinct) > 1:
                conflicts[name] = distinct
            merged[name] = distinct[0] if distinct else None
    return merged, conflicts


def extract_long(
    text: str | list[str],
    schema: type[T],
    *,
    llm=None,
    instructions: str | None = None,
    chunk_size: int = 12000,
    overlap: int = 500,
    resolve_conflicts: bool = True,
) -> T:
    """Extract from text longer than one comfortable prompt: per-chunk partial
    extraction, rule-based merge, then (optionally) an LLM pass to settle
    conflicting scalar values and fill required fields."""
    llm = _llm(llm)
    if isinstance(text, str):
        if len(text) <= chunk_size:
            return extract(text, schema, llm=llm, instructions=instructions)
        from docingest import RecursiveChunker

        chunks = [text[a:b] for a, b, _ in RecursiveChunker(chunk_size, overlap).split(text)]
    else:
        chunks = text
    Partial = partial_model(schema)
    partials = [
        extract(c, Partial, llm=llm, instructions=instructions).model_dump(mode="json")
        for c in chunks
    ]
    merged, conflicts = merge_partials(schema, partials)
    try:
        if conflicts and resolve_conflicts:
            raise ValueError("conflicts")
        return schema.model_validate(merged)
    except (ValidationError, ValueError):
        prompt = (
            "Several extractions from different parts of one document were merged. Produce the final "
            "record. Where values conflict, choose the one best supported (prefer the most specific); "
            f"keep all list items.\n\nMerged:\n{json.dumps(merged, ensure_ascii=False, indent=1)}\n\n"
            f"Conflicting values:\n{json.dumps(conflicts, ensure_ascii=False, indent=1)}"
        )
        return llm.structured(prompt, schema, system=_SYSTEM)


@dataclass
class ExtractionResult:
    source: str
    data: dict[str, Any] | None
    error: str | None = None


def extract_files(
    paths: Iterable[str | Path],
    schema: type[BaseModel],
    *,
    llm=None,
    instructions: str | None = None,
    ocr=None,
    concurrency: int = 4,
) -> Iterator[ExtractionResult]:
    """Load each file with doc-ingest and extract one record per file."""
    from concurrent.futures import ThreadPoolExecutor

    from docingest import iter_files, load

    llm = _llm(llm)

    def one(path: Path) -> ExtractionResult:
        try:
            doc = load(path, ocr)
            obj = extract_long(doc.text, schema, llm=llm, instructions=instructions)
            return ExtractionResult(str(path), obj.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", path, e)
            return ExtractionResult(str(path), None, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        yield from pool.map(one, iter_files(list(paths)))


def write_jsonl(results: Iterable[ExtractionResult], path: str | Path) -> list[ExtractionResult]:
    out = list(results)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({"source": r.source, "data": r.data, "error": r.error}, ensure_ascii=False) + "\n")
    return out


def write_csv(results: Iterable[ExtractionResult], path: str | Path) -> None:
    """One row per record; nested values are JSON-encoded in their cell."""
    rows = [r for r in results if r.data is not None]
    cols = list(dict.fromkeys(k for r in rows for k in r.data))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", *cols])
        for r in rows:
            w.writerow([r.source, *(
                json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for v in (r.data.get(c) for c in cols)
            )])
