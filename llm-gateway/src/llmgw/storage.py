"""Call logging (JSONL, one file per day), an optional SQLite response cache and an embedding cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from array import array
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Request, Response, ToolCall, Usage

_MAX_LOGGED_CHARS = 8000


def _clip(s: str | None) -> str | None:
    if s is None or len(s) <= _MAX_LOGGED_CHARS:
        return s
    return s[:_MAX_LOGGED_CHARS] + f"... [{len(s) - _MAX_LOGGED_CHARS} chars clipped]"


class CallLogger:
    """Appends one JSON line per call. With ``content=False`` only metadata is written
    (model, tokens, cost, latency, finish reason, error type): no prompts, replies or
    tool arguments, for data that must not be stored."""

    def __init__(self, directory: str | Path, content: bool = True):
        self.dir = Path(directory)
        self.content = content
        self._lock = threading.Lock()

    def write(self, provider: str, req: Request, resp: Response | None, latency_ms: float, error: BaseException | None = None):
        now = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "ts": now.isoformat(),
            "provider": provider,
            "model": req.model,
            "latency_ms": round(latency_ms, 1),
            "tools": [t.name for t in req.tools or []],
            "structured": bool(req.json_schema),
        }
        if self.content:
            record["system"] = _clip(req.system)
            record["messages"] = [
                {
                    "role": m.role,
                    "content": _clip(m.content),
                    **({"tool_calls": [asdict(c) for c in m.tool_calls]} if m.tool_calls else {}),
                    **({"attachments": [a.mime for a in m.attachments]} if m.attachments else {}),
                }
                for m in req.messages
            ]
        if resp is not None:
            record.update(usage=resp.usage.to_dict(), finish_reason=resp.finish_reason, cached=resp.cached)
            if self.content:
                record.update(text=_clip(resp.text), tool_calls=[asdict(c) for c in resp.tool_calls])
        if error is not None:
            # Error messages can quote model output, so metadata-only logs keep just the type.
            record["error"] = f"{type(error).__name__}: {error}" if self.content else type(error).__name__
        self._append(now, record)

    def write_embed(self, provider: str, model: str, n_texts: int, latency_ms: float,
                    usage: Usage | None = None, error: BaseException | None = None, n_cached: int = 0):
        """One record per embed() call. Token counts are estimated (providers do not
        return them for embeddings), so ``tokens_estimated`` is set."""
        now = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "ts": now.isoformat(), "provider": provider, "model": model, "kind": "embed",
            "latency_ms": round(latency_ms, 1), "n_texts": n_texts,
        }
        if n_cached:
            record["n_cached"] = n_cached  # served from the embedding cache, not sent or billed
        if usage is not None:
            record.update(usage=usage.to_dict(), tokens_estimated=True)
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}" if self.content else type(error).__name__
        self._append(now, record)

    def _append(self, now: datetime, record: dict[str, Any]) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f"calls-{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def request_key(provider: str, req: Request) -> str:
    payload = {
        "provider": provider,
        "model": req.model,
        "system": req.system,
        "messages": [
            [m.role, m.content, [asdict(c) for c in m.tool_calls], m.tool_call_id,
             [hashlib.sha256(a.data).hexdigest() for a in m.attachments]]
            for m in req.messages
        ],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "tools": [asdict(t) for t in req.tools or []],
        "tool_choice": req.tool_choice,
        "json_schema": req.json_schema,
        "extra": req.extra,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT, created TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> Response | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        return Response(
            text=d["text"],
            model=d["model"],
            tool_calls=[ToolCall(**c) for c in d["tool_calls"]],
            usage=Usage(**d["usage"]),
            finish_reason=d.get("finish_reason"),
            cached=True,
        )

    def put(self, key: str, resp: Response) -> None:
        value = json.dumps({
            "text": resp.text,
            "model": resp.model,
            "tool_calls": [asdict(c) for c in resp.tool_calls],
            "usage": resp.usage.to_dict(),
            "finish_reason": resp.finish_reason,
        })
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                (key, value, datetime.now(timezone.utc).isoformat()),
            )
            self._db.commit()


class EmbeddingCache:
    """Vectors keyed by (model, task, SHA-256 of the text). Stores no text, only hashes
    and vectors, so rebuilding an index re-embeds only passages that changed."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)")
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, task: str | None, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{task or ''}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                batch = keys[i : i + 500]
                rows = self._db.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(batch))})", batch
                ).fetchall()
                out.update({k: array("d", v).tolist() for k, v in rows})
        return out

    def put_many(self, items: dict[str, list[float]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                [(k, array("d", v).tobytes()) for k, v in items.items()],
            )
            self._db.commit()
