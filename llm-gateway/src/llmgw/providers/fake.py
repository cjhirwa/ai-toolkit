from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterator
from typing import Union

from ..types import Request, Response, Usage
from .base import Provider

Scripted = Union[str, Response, Callable[[Request], Union[str, Response]]]


class FakeProvider(Provider):
    """Deterministic offline provider for tests and dry runs.

    ``responses`` is either a list (consumed in order; once empty, the last user
    message is echoed back) or a function of the request. Embeddings are a
    hashed bag of words, so texts sharing words land close together.
    """

    name = "fake"

    def __init__(self, responses: list[Scripted] | Callable[[Request], str | Response] | None = None, dim: int = 256):
        self.responses = responses
        self.dim = dim
        self.requests: list[Request] = []

    def _next(self, req: Request) -> Scripted | None:
        if callable(self.responses):
            return self.responses
        if self.responses:
            return self.responses.pop(0)
        return None

    def complete(self, req: Request) -> Response:
        self.requests.append(req)
        r = self._next(req)
        if callable(r) and not isinstance(r, Response):
            r = r(req)
        if r is None:
            r = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if isinstance(r, str):
            r = Response(text=r, model=req.model)
        if not r.usage.input_tokens:
            words = sum(len(m.content.split()) for m in req.messages) + len((req.system or "").split())
            r.usage = Usage(input_tokens=words, output_tokens=len(r.text.split()))
        return r

    def stream(self, req: Request) -> Iterator[str | Usage]:
        resp = self.complete(req)
        for i, word in enumerate(resp.text.split(" ")):
            yield word if i == 0 else " " + word
        yield resp.usage

    def embed(self, texts: list[str], model: str, task: str | None = None) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
