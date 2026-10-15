from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Tracer:
    """Append-only JSONL trace, one file per agent run."""

    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory or os.getenv("AGENTLOOP_TRACE_DIR", ".agentloop/traces"))
        self._lock = threading.Lock()
        self.path: Path | None = None
        self.run_id: str | None = None

    def start(self, name: str, **data: Any) -> str:
        self.run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.run_id}-{name}.jsonl"
        self.event("run_start", name=name, **data)
        return self.run_id

    def event(self, kind: str, **data: Any) -> None:
        if self.path is None:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "run_id": self.run_id, **data}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


class NullTracer(Tracer):
    def __init__(self):
        super().__init__(".")

    def start(self, name: str, **data: Any) -> str:
        self.run_id = uuid.uuid4().hex[:8]
        return self.run_id

    def event(self, kind: str, **data: Any) -> None:
        pass


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def latest_trace(directory: str | Path | None = None) -> Path | None:
    d = Path(directory or os.getenv("AGENTLOOP_TRACE_DIR", ".agentloop/traces"))
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return files[-1] if files else None
