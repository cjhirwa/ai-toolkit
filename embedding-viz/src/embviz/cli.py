from __future__ import annotations

import argparse
import csv
import webbrowser
from pathlib import Path

import numpy as np

from llmgw.console import utf8_stdio

from .core import coverage, embed_queries, kmeans, load_corpus, load_queries, near_duplicates, reduce
from .plot import corpus_map, save_html


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="embviz")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("map", help="interactive 2-D map of the corpus")
    m.add_argument("corpus", help="retrieval-kit index dir or chunks.jsonl")
    m.add_argument("-o", "--out", default="corpus_map.html")
    m.add_argument("--color", default="source", help="metadata key, or 'cluster'")
    m.add_argument("--clusters", type=int, default=12)
    m.add_argument("--method", default="auto", choices=["auto", "pca", "umap", "tsne"])
    m.add_argument("--queries", help="eval/qrels JSONL to overlay")
    m.add_argument("--no-open", action="store_true")

    d = sub.add_parser("dupes", help="near-duplicate chunk pairs")
    d.add_argument("corpus")
    d.add_argument("--threshold", type=float, default=0.95)
    d.add_argument("-o", "--out", help="CSV output")

    c = sub.add_parser("coverage", help="queries with no close chunk (likely corpus gaps)")
    c.add_argument("corpus")
    c.add_argument("queries")
    c.add_argument("--threshold", type=float, default=0.6)
    args = p.parse_args(argv)

    corpus = load_corpus(args.corpus)
    if args.cmd == "map":
        queries = load_queries(args.queries) if args.queries else None
        qv = embed_queries(queries) if queries else None
        stacked = np.vstack([corpus.vectors, qv]) if qv is not None else corpus.vectors
        coords = reduce(stacked, args.method)
        color = [f"cluster {c}" for c in kmeans(corpus.vectors, args.clusters)] if args.color == "cluster" else args.color
        n = len(corpus.ids)
        fig = corpus_map(corpus, coords[:n], color_by=color, queries=queries,
                         query_coords=coords[n:] if qv is not None else None,
                         title=f"{Path(args.corpus).name} · {n} chunks")
        save_html(fig, args.out)
        print(f"wrote {args.out}")
        if not args.no_open:
            webbrowser.open(Path(args.out).resolve().as_uri())
    elif args.cmd == "dupes":
        pairs = near_duplicates(corpus, args.threshold)
        print(f"{len(pairs)} pairs ≥ {args.threshold}")
        by_id = dict(zip(corpus.ids, corpus.texts))
        for a, b, s in pairs[:20]:
            print(f"{s:.3f}  {a}  {b}\n   {by_id[a][:90]!r}\n   {by_id[b][:90]!r}")
        if args.out:
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id_a", "id_b", "similarity"])
                w.writerows(pairs)
    else:
        queries = load_queries(args.queries)
        rows = coverage(corpus, queries, embed_queries(queries))
        gaps = [r for r in rows if r.best_similarity < args.threshold]
        print(f"{len(gaps)}/{len(rows)} queries below {args.threshold}\n")
        for r in gaps[:30]:
            print(f"{r.best_similarity:.3f}  {r.query[:100]}\n       nearest: {r.nearest_text[:100]!r}")


if __name__ == "__main__":
    main()
