from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..types import Request, Response, Usage


class Provider(ABC):
    name: str = ""
    # Seconds before a request is abandoned (and retried, if attempts remain).
    # LLMGW_TIMEOUT overrides it for every provider; a ``timeout=`` argument overrides both.
    default_timeout: float = 120.0

    def _timeout(self, timeout: float | None) -> float:
        if timeout is not None:
            return timeout
        env = os.getenv("LLMGW_TIMEOUT")
        return float(env) if env else self.default_timeout

    @abstractmethod
    def complete(self, req: Request) -> Response: ...

    def stream(self, req: Request) -> Iterator[str | Usage]:
        """Yield text chunks, then (optionally) one ``Usage`` with the call's token counts,
        which LLM.stream() uses for cost and logging instead of passing it to the caller."""
        resp = self.complete(req)
        yield resp.text
        yield resp.usage

    def embed(self, texts: list[str], model: str, task: str | None = None) -> list[list[float]]:
        """``task`` is "query" or "document" for providers that embed them differently."""
        raise NotImplementedError(f"{self.name} does not support embeddings")
