from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Example:
    id: str
    input: Any  # a string, or a dict of task inputs
    expected: Any = None
    context: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maybe_json(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _content_id(input_value: Any, expected: Any) -> str:
    """An id derived from the example itself, so it survives reordering and inserted rows
    (row-number ids made compare match unrelated examples after any edit)."""
    payload = json.dumps([input_value, expected], sort_keys=True, ensure_ascii=False, default=str)
    return "ex-" + hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_dataset(
    path: str | Path,
    *,
    input_key: str = "input",
    expected_key: str = "expected",
    context_key: str = "context",
    id_key: str = "id",
    limit: int | None = None,
) -> list[Example]:
    """Load JSONL, JSON (a list) or CSV. Other columns go into ``metadata``.
    CSV cells that look like JSON lists/objects are parsed."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = [{k: _maybe_json(v) for k, v in r.items()} for r in csv.DictReader(f)]
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    used = {input_key, expected_key, context_key, id_key}
    examples = []
    for i, r in enumerate(rows[:limit] if limit else rows):
        if input_key not in r:
            raise KeyError(f"Row {i} has no {input_key!r} field (columns: {list(r)})")
        ctx = r.get(context_key)
        if isinstance(ctx, str):
            ctx = [ctx]
        meta = dict(r.get("metadata") or {}) if isinstance(r.get("metadata"), dict) else {}
        meta.update({k: v for k, v in r.items() if k not in used and k != "metadata"})
        examples.append(
            Example(
                id=str(r.get(id_key) or _content_id(r.get(input_key), r.get(expected_key))),
                input=r[input_key],
                expected=r.get(expected_key) if r.get(expected_key) not in ("", None) else None,
                context=ctx or None,
                metadata=meta,
            )
        )
    dupes = sorted(i for i, n in Counter(e.id for e in examples).items() if n > 1)
    if dupes:
        # The store keys results by (run, example id), so duplicates would overwrite each other.
        raise ValueError(f"{path}: duplicate example ids {dupes[:5]}; give each row a unique {id_key!r} or remove repeats")
    return examples


def save_dataset(examples: list[Example], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
