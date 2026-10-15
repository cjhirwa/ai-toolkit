from __future__ import annotations

import argparse
import json

from llmgw.console import utf8_stdio

from .bench import benchmark, format_table, load_qrels
from .index import Index


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="retrievekit")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed chunks.jsonl into an index directory")
    b.add_argument("chunks")
    b.add_argument("-o", "--out", default="index")
    b.add_argument("--embed-model")
    s = sub.add_parser("search")
    s.add_argument("index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", default="hybrid")
    s.add_argument("--filter", help='JSON metadata filter, e.g. {"source": "manual.pdf"}')
    s.add_argument("--json", action="store_true")
    be = sub.add_parser("bench", help="compare retrieval modes on labelled queries")
    be.add_argument("index")
    be.add_argument("qrels")
    be.add_argument("-k", type=int, default=10)
    be.add_argument("--modes", default="bm25,dense,hybrid")
    be.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    if args.cmd == "build":
        idx = Index.build(args.chunks, args.out, embed_model=args.embed_model)
        print(f"indexed {len(idx.docs)} chunks with {idx.dense.embed_model} into {args.out}/")
    elif args.cmd == "search":
        hits = Index.load(args.index).retriever(args.mode).retrieve(
            args.query, args.k, json.loads(args.filter) if args.filter else None
        )
        if args.json:
            print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=1))
            return
        for i, h in enumerate(hits, 1):
            loc = h.metadata.get("source", "")
            if "page" in h.metadata:
                loc += f" p.{h.metadata['page']}"
            print(f"{i}. [{h.score:.3f}] {h.id}  {loc}\n   {h.text[:300].replace(chr(10), ' ')}\n")
    elif args.cmd == "bench":
        idx = Index.load(args.index)
        qrels = load_qrels(args.qrels)[: args.limit]
        results = benchmark({m: idx.retriever(m) for m in args.modes.split(",")}, qrels, args.k)
        print(f"{len(qrels)} queries\n")
        print(format_table(results))


if __name__ == "__main__":
    main()
