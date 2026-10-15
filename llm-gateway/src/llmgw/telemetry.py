"""OpenTelemetry spans for model calls.

With ``opentelemetry-api`` installed (``llm-gateway[otel]``), every completion and embedding
call is a span, ``llm.complete`` or ``llm.embed``, carrying the OpenTelemetry GenAI
attributes (``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.usage.input_tokens`` and
``output_tokens``, ``gen_ai.response.finish_reasons``) plus ``llmgw.cost_usd``,
``llmgw.retries`` and ``llmgw.cached``. Failed calls are marked as errors with the exception
recorded. Spans go wherever your OpenTelemetry SDK exports them; with no SDK configured, the
API is a no-op and this costs nothing. Without the package, nothing is imported.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .types import Response, Usage

try:
    from opentelemetry import trace

    _tracer: Any = trace.get_tracer("llmgw")
except ImportError:  # optional dependency
    _tracer = None


@contextmanager
def span(name: str, provider: str, model: str, **attributes: Any) -> Iterator[Any]:
    """A span around one model call, or a stand-in with the same methods when tracing is off."""
    if _tracer is None:
        yield _NoSpan()
        return
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute("gen_ai.system", provider)
        s.set_attribute("gen_ai.request.model", model)
        for key, value in attributes.items():
            if value is not None:
                s.set_attribute(key, value)
        yield s


def record_usage(s: Any, usage: Usage, response: Response | None = None, **attributes: Any) -> None:
    s.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    s.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    if usage.cost_usd is not None:
        s.set_attribute("llmgw.cost_usd", usage.cost_usd)
    if response is not None and response.finish_reason:
        s.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
    for key, value in attributes.items():
        if value is not None:
            s.set_attribute(key, value)


class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass
