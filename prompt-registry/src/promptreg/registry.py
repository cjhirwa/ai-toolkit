"""Prompts live as ``<root>/<name>/v<N>.yaml``, one file per version, so git diffs stay readable.

    description: Answer a customer question from retrieved passages
    model: gemini:gemini-3.6-flash        # optional hint
    params: {temperature: 0.2}            # optional hint
    system: |
      You are a support assistant for Acme customers...
    template: |
      Passages:
      {% for p in passages %}[{{ loop.index }}] {{ p }}
      {% endfor %}
      Question: {{ question }}
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, meta

_env = Environment(undefined=StrictUndefined, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
_VERSION_FILE = re.compile(r"^v(\d+)\.ya?ml$")


class _LiteralDumper(yaml.SafeDumper):
    pass


def _str_repr(dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_repr)


@dataclass
class RenderedPrompt:
    system: str | None
    user: str

    def messages(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + [{"role": "user", "content": self.user}]


@dataclass
class Prompt:
    name: str
    version: int
    template: str
    system: str | None = None
    description: str = ""
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    created: str | None = None
    path: Path | None = None

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    @property
    def variables(self) -> set[str]:
        names: set[str] = set()
        for src in (self.template, self.system or ""):
            names |= meta.find_undeclared_variables(_env.parse(src))
        return names

    def render(self, **variables: Any) -> RenderedPrompt:
        missing = self.variables - set(variables)
        if missing:
            raise KeyError(f"{self.id} is missing variables: {sorted(missing)}")
        system = _env.from_string(self.system).render(**variables).strip() if self.system else None
        return RenderedPrompt(system, _env.from_string(self.template).render(**variables).strip())

    def to_yaml(self) -> str:
        data: dict[str, Any] = {"description": self.description, "created": self.created}
        if self.model:
            data["model"] = self.model
        if self.params:
            data["params"] = self.params
        if self.system:
            data["system"] = self.system
        data["template"] = self.template
        return yaml.dump(data, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True, width=1000)


class Registry:
    def __init__(self, root: str | Path = "prompts"):
        self.root = Path(root)

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and self.versions(d.name))

    def versions(self, name: str) -> list[int]:
        d = self.root / name
        if not d.exists():
            return []
        return sorted(int(m.group(1)) for f in d.iterdir() if (m := _VERSION_FILE.match(f.name)))

    def get(self, name: str, version: int | str = "latest") -> Prompt:
        """``get("rag_answer")``, ``get("rag_answer", 2)`` or ``get("rag_answer@v2")``."""
        if "@" in name:
            name, version = name.split("@", 1)
        versions = self.versions(name)
        if not versions:
            raise KeyError(f"No prompt named {name!r} under {self.root}")
        v = versions[-1] if version in ("latest", None) else int(str(version).lstrip("v"))
        path = next((p for p in (self.root / name / f"v{v}.yaml", self.root / name / f"v{v}.yml") if p.exists()), None)
        if path is None:
            raise KeyError(f"{name} has no version {v} (have {versions})")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Prompt(
            name=name,
            version=v,
            template=data["template"],
            system=data.get("system"),
            description=data.get("description") or "",
            model=data.get("model"),
            params=data.get("params") or {},
            created=str(data["created"]) if data.get("created") else None,
            path=path,
        )

    def render(self, ref: str, **variables: Any) -> RenderedPrompt:
        return self.get(ref).render(**variables)

    def create(self, name: str, template: str, *, system: str | None = None, description: str = "",
               model: str | None = None, params: dict[str, Any] | None = None) -> Prompt:
        """Save as the next version of ``name``."""
        v = (self.versions(name) or [0])[-1] + 1
        prompt = Prompt(name, v, template, system, description, model, params or {}, date.today().isoformat())
        path = self.root / name / f"v{v}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt.to_yaml(), encoding="utf-8")
        prompt.path = path
        return prompt

    def new_version(self, name: str, **changes: Any) -> Prompt:
        """Copy the latest version into a new one, applying ``changes``."""
        latest = self.get(name)
        fields = {
            "template": latest.template, "system": latest.system, "description": latest.description,
            "model": latest.model, "params": latest.params,
        }
        fields.update(changes)
        return self.create(name, **fields)

    def diff(self, name: str, a: int | str, b: int | str = "latest") -> str:
        pa, pb = self.get(name, a), self.get(name, b)
        return "".join(
            difflib.unified_diff(
                pa.to_yaml().splitlines(keepends=True),
                pb.to_yaml().splitlines(keepends=True),
                fromfile=pa.id,
                tofile=pb.id,
            )
        )
