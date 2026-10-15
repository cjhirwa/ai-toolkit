"""Browse and render prompt-registry prompts. Env: PROMPTS_DIR (default ./prompts)."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from ._run import run

mcp = MCPServer("prompts")


def _reg():
    from promptreg import Registry

    return Registry(os.getenv("PROMPTS_DIR", "prompts"))


@mcp.tool()
def list_prompts() -> list[dict]:
    """All prompts with their latest version, description and variables."""
    reg = _reg()
    out = []
    for name in reg.names():
        p = reg.get(name)
        out.append({"name": name, "latest": p.version, "versions": reg.versions(name),
                    "description": p.description, "variables": sorted(p.variables)})
    return out


@mcp.tool()
def get_prompt(name: str, version: str = "latest") -> dict:
    """The raw template, system prompt and hints for one prompt version."""
    p = _reg().get(name, version)
    return {"id": p.id, "system": p.system, "template": p.template, "model": p.model,
            "params": p.params, "variables": sorted(p.variables), "path": str(p.path)}


@mcp.tool()
def render_prompt(name: str, variables: dict[str, Any], version: str = "latest") -> dict:
    """Render a prompt with variables; returns the final system and user text."""
    r = _reg().get(name, version).render(**variables)
    return {"system": r.system, "user": r.user}


@mcp.tool()
def diff_prompt(name: str, a: str, b: str = "latest") -> str:
    """Unified diff between two versions of a prompt."""
    return _reg().diff(name, a, b) or "(no differences)"


def main() -> None:
    run(mcp)


if __name__ == "__main__":
    main()
