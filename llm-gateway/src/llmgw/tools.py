"""Turn plain Python functions into tools the model can call.

    @tool
    def search(query: str, k: int = 5) -> list[dict]:
        '''Search the knowledge base.

        Args:
            query: What to look for.
            k: How many passages to return.
        '''
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, Field, create_model

from .types import ToolSpec

_ARG_LINE = re.compile(r"^\s{2,}(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    doc = inspect.cleandoc(doc or "")
    summary = doc.split("\n\n")[0].strip()
    params: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        if line.strip() in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue
        if in_args:
            m = _ARG_LINE.match(line)
            if m:
                params[m.group(1)] = m.group(2).strip()
            elif line.strip() and not line.startswith(" "):
                in_args = False
    return summary, params


class Tool:
    def __init__(self, fn: Callable[..., Any], name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        summary, param_docs = _parse_docstring(fn.__doc__ or "")
        self.description = description or summary or self.name
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for pname, p in inspect.signature(fn).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[pname] = (hints.get(pname, str), Field(default, description=param_docs.get(pname)))
        self.args_model: type[BaseModel] = create_model(f"{self.name}_args", **fields)

    @property
    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return ToolSpec(self.name, self.description, schema)

    def run(self, arguments: dict[str, Any]) -> Any:
        validated = self.args_model.model_validate(arguments)
        return self.fn(**{k: getattr(validated, k) for k in type(validated).model_fields})

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator (bare or with arguments) that wraps a function as a :class:`Tool`."""
    if fn is None:
        return lambda f: Tool(f, name=name, description=description)
    return Tool(fn, name=name, description=description)


def as_tool(obj: Tool | Callable) -> Tool:
    return obj if isinstance(obj, Tool) else Tool(obj)
