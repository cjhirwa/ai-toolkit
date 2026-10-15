from __future__ import annotations

import argparse
import json

from llmgw.console import utf8_stdio

from .evaluate import evaluate, format_report, load_labeled
from .router import Router


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="qroute")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("config", help="routes.yaml")
    c.add_argument("text")
    c.add_argument("--no-embedding", action="store_true")
    e = sub.add_parser("eval", help="accuracy against a labelled CSV")
    e.add_argument("config")
    e.add_argument("csv")
    e.add_argument("--text-col", default="text")
    e.add_argument("--label-col", default="label")
    e.add_argument("--limit", type=int)
    e.add_argument("--no-embedding", action="store_true")
    e.add_argument("--no-llm", action="store_true")
    args = p.parse_args(argv)

    router = Router.from_yaml(
        args.config,
        embedding=not args.no_embedding,
        llm_fallback=not getattr(args, "no_llm", False),
    )
    if args.cmd == "classify":
        print(json.dumps(router.classify(args.text).__dict__, ensure_ascii=False, indent=1))
    else:
        print(format_report(evaluate(router, load_labeled(args.csv, args.text_col, args.label_col, args.limit))))


if __name__ == "__main__":
    main()
