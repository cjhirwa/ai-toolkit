"""Per-provider limits shared by every LLM in the process.

Each thread pool in the toolkit (eval, OCR, synthetic data, batch) used to call providers
independently and only back off after a 429. These limits sit in front of every request
instead:

    LLMGW_RPM_<PROVIDER>=60          at most 60 requests per minute (token bucket)
    LLMGW_CONCURRENCY_<PROVIDER>=4   at most 4 requests in flight at once

<PROVIDER> is the provider name in upper case (GEMINI, OPENAI, OLLAMA). Ollama defaults to
one request at a time, since a local server answers them one by one and queued requests
only run into the timeout; other providers are unlimited unless configured.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_DEFAULT_CONCURRENCY = {"ollama": 1}


class _Bucket:
    def __init__(self, per_minute: float):
        self.rate = per_minute / 60.0
        self.capacity = max(1.0, per_minute / 60.0 * 5)  # allow a burst of ~5 seconds' worth
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


class ProviderLimit:
    def __init__(self, rpm: float | None, concurrency: int | None):
        self.bucket = _Bucket(rpm) if rpm else None
        self.slots = threading.BoundedSemaphore(concurrency) if concurrency else None

    @contextmanager
    def acquire(self) -> Iterator[None]:
        if self.slots:
            self.slots.acquire()
        try:
            if self.bucket:
                self.bucket.take()
            yield
        finally:
            if self.slots:
                self.slots.release()


_LIMITS: dict[str, ProviderLimit] = {}
_lock = threading.Lock()


def _env_number(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def limit_for(provider: str) -> ProviderLimit:
    """The shared limit for ``provider``, read from the environment on first use."""
    with _lock:
        if provider not in _LIMITS:
            key = provider.upper()
            rpm = _env_number(f"LLMGW_RPM_{key}")
            conc = _env_number(f"LLMGW_CONCURRENCY_{key}")
            conc = int(conc) if conc is not None else _DEFAULT_CONCURRENCY.get(provider)
            _LIMITS[provider] = ProviderLimit(rpm, conc if conc and conc > 0 else None)
        return _LIMITS[provider]


def reset_limits() -> None:
    """Forget configured limits (after changing the environment, or in tests)."""
    with _lock:
        _LIMITS.clear()
