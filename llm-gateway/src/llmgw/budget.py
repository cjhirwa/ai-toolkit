"""Spending caps.

- ``LLM(max_cost_usd=...)`` caps what one client may spend.
- ``LLMGW_DAILY_BUDGET_USD`` caps what all clients writing to the same log folder spend
  per UTC day. Today's total is read from the call log (new lines only, as they are
  appended), so separate processes count against one budget. It needs logging on.

A call is refused with :class:`BudgetExceeded` once the cap is reached. The check happens
before each call, so the call that crosses the line still completes: set caps with some
headroom. Calls with no known price count as 0.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(Exception):
    """A spending cap was reached; no further calls are made."""


class DailySpend:
    """Running total of today's cost from ``calls-YYYY-MM-DD.jsonl`` in ``log_dir``."""

    def __init__(self, log_dir: str | Path):
        self.dir = Path(log_dir)
        self._lock = threading.Lock()
        self._day = ""
        self._offset = 0
        self._total = 0.0

    def total(self) -> float:
        with self._lock:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if day != self._day:
                self._day, self._offset, self._total = day, 0, 0.0
            path = self.dir / f"calls-{day}.jsonl"
            if path.exists():
                with path.open("rb") as f:
                    f.seek(self._offset)
                    data = f.read()
                complete = data[: data.rfind(b"\n") + 1]  # ignore a line still being written
                self._offset += len(complete)
                for line in complete.splitlines():
                    try:
                        self._total += (json.loads(line).get("usage") or {}).get("cost_usd") or 0.0
                    except ValueError:
                        continue
            return self._total


_SPENDS: dict[str, DailySpend] = {}
_lock = threading.Lock()


def daily_limit() -> float | None:
    value = os.getenv("LLMGW_DAILY_BUDGET_USD", "").strip()
    return float(value) if value else None


def daily_spend(log_dir: str | Path) -> DailySpend:
    key = str(Path(log_dir).resolve())
    with _lock:
        if key not in _SPENDS:
            _SPENDS[key] = DailySpend(log_dir)
        return _SPENDS[key]
