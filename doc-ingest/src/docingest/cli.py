from __future__ import annotations

import argparse
import os
import statistics
from collections import Counter

from llmgw import load_env
from llmgw.console import cli_logging, utf8_stdio

from .chunking import get_chunker
from .ocr import get_ocr
from .pipeline import ingest, read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    load_env()
    cli_logging()
    p = argparse.ArgumentParser(prog="docingest")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest files/folders into chunks.jsonl")
    r.add_argument("paths", nargs="*", help="files or folders (default: DOCS_DIR from the environment or .env)")
    r.add_argument("-o", "--out", default="chunks.jsonl")
    r.add_argument("--chunker", default="headings", choices=["recursive", "fixed", "headings", "semantic"])
    r.add_argument("--size", type=int, default=1000, help="max chars per chunk")
    r.add_argument("--overlap", type=int, default=150)
    r.add_argument("--ocr", choices=["none", "tesseract", "llm"], default="none")
    r.add_argument("--glob", default="**/*")
    s = sub.add_parser("stats", help="summarise a chunks.jsonl")
    s.add_argument("path")
    args = p.parse_args(argv)

    if args.cmd == "run":
        if not args.paths:
            if not os.getenv("DOCS_DIR"):
                p.error("pass files or folders to ingest, or set DOCS_DIR in .env")
            args.paths = [os.environ["DOCS_DIR"]]
        chunker = get_chunker(args.chunker, args.size, args.overlap)
        chunks = ingest(args.paths, chunker, get_ocr(args.ocr), pattern=args.glob)
        n = write_jsonl(chunks, args.out)
        print(f"wrote {n} chunks from {len({c.doc_id for c in chunks})} documents to {args.out}")
    else:
        chunks = list(read_jsonl(args.path))
        lens = [len(c.text) for c in chunks]
        by_source = Counter(c.metadata.get("source") for c in chunks)
        print(f"{len(chunks)} chunks, {len(by_source)} sources")
        if lens:
            print(f"chars: min {min(lens)}, median {statistics.median(lens):.0f}, max {max(lens)}")
        for src, n in by_source.most_common(15):
            print(f"  {n:5}  {src}")


if __name__ == "__main__":
    main()
