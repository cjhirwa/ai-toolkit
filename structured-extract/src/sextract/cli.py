"""sextract run --schema my_schemas.py:LabReport docs/ -o out.jsonl --csv out.csv"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

from llmgw.console import cli_logging, utf8_stdio

from .core import extract_files, write_csv, write_jsonl


def _load(ref: str):
    target, _, attr = ref.rpartition(":")
    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(target).stem, target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(target)
    return getattr(mod, attr)


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    cli_logging()
    p = argparse.ArgumentParser(prog="sextract")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("paths", nargs="+")
    r.add_argument("--schema", required=True, help="module:Model or file.py:Model")
    r.add_argument("-o", "--out", default="extracted.jsonl")
    r.add_argument("--csv")
    r.add_argument("--instructions")
    r.add_argument("-m", "--model")
    r.add_argument("--ocr", choices=["none", "tesseract", "llm"], default="none")
    r.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args(argv)

    from docingest import get_ocr
    from llmgw import LLM

    results = write_jsonl(
        extract_files(
            args.paths, _load(args.schema), llm=LLM(args.model, temperature=0),
            instructions=args.instructions, ocr=get_ocr(args.ocr), concurrency=args.concurrency,
        ),
        args.out,
    )
    ok = sum(1 for x in results if x.data is not None)
    print(f"{ok}/{len(results)} extracted -> {args.out}")
    if args.csv:
        write_csv(results, args.csv)
        print(f"csv -> {args.csv}")


if __name__ == "__main__":
    main()
