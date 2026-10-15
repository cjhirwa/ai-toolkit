from __future__ import annotations

import argparse
import json
import os
import sys

from .registry import Registry


def _parse_vars(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="promptreg")
    p.add_argument("--root", default=os.getenv("PROMPTS_DIR", "prompts"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("ref", help="name or name@vN")
    d = sub.add_parser("diff")
    d.add_argument("name")
    d.add_argument("a")
    d.add_argument("b", nargs="?", default="latest")
    n = sub.add_parser("new", help="create a prompt, or copy the latest version to a new one")
    n.add_argument("name")
    n.add_argument("--template", help="template text for a brand-new prompt")
    n.add_argument("--system")
    n.add_argument("--description", default="")
    r = sub.add_parser("render")
    r.add_argument("ref")
    r.add_argument("--var", action="append", default=[], help="key=value (value may be JSON)")
    args = p.parse_args(argv)
    reg = Registry(args.root)

    if args.cmd == "list":
        for name in reg.names():
            latest = reg.get(name)
            print(f"{name:28} v{latest.version:<3} {latest.description}")
    elif args.cmd == "show":
        pr = reg.get(args.ref)
        print(f"# {pr.id}  ({pr.path})\n# variables: {sorted(pr.variables)}\n")
        print(pr.to_yaml())
    elif args.cmd == "diff":
        print(reg.diff(args.name, args.a, args.b) or "(no differences)")
    elif args.cmd == "new":
        if reg.versions(args.name):
            pr = reg.new_version(args.name)
        else:
            if not args.template:
                p.error("--template is required for a new prompt")
            pr = reg.create(args.name, args.template, system=args.system, description=args.description)
        print(f"created {pr.id} at {pr.path}")
    elif args.cmd == "render":
        rendered = reg.render(args.ref, **_parse_vars(args.var))
        if rendered.system:
            print(f"--- system ---\n{rendered.system}\n")
        print(f"--- user ---\n{rendered.user}")


if __name__ == "__main__":
    main()
