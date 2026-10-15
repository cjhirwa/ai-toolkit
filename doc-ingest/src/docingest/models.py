from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageSpan:
    page: int  # 1-based
    start: int  # char offset in Document.text
    end: int


@dataclass
class Document:
    id: str
    source: str
    text: str
    pages: list[PageSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def page_at(self, offset: int) -> int | None:
        if not self.pages:
            return None
        i = bisect.bisect_right([p.start for p in self.pages], offset) - 1
        return self.pages[max(i, 0)].page


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(d["id"], d.get("doc_id", ""), d["text"], d.get("metadata") or {})
