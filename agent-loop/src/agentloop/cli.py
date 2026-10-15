from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from llmgw.console import utf8_stdio

from .tracing import latest_trace, read_trace
from .viewer import render


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="agentloop")
    p.add_argument("--dir", default=None, help="trace directory (default .agentloop/traces)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls", help="list traces")
    v = sub.add_parser("view", help="render a trace to HTML")
    v.add_argument("trace", nargs="?", default="latest")
    v.add_argument("-o", "--out")
    v.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)
    d = Path(args.dir or ".agentloop/traces")

    if args.cmd == "ls":
        for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
            events = read_trace(f)
            end = next((e for e in events if e["kind"] == "run_end"), {})
            n_tools = sum(1 for e in events if e["kind"] == "tool")
            print(f"{f.name:60} steps={end.get('steps', '?'):<3} tools={n_tools:<3} {end.get('stopped', 'running?')}")
    else:
        path = latest_trace(d) if args.trace == "latest" else Path(args.trace)
        if path is None:
            raise SystemExit(f"No traces in {d}")
        out = Path(args.out or path.with_suffix(".html"))
        out.write_text(render(path), encoding="utf-8")
        print(f"wrote {out}")
        if not args.no_open:
            webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
