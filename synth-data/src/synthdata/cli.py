from __future__ import annotations

import argparse
import sys
from collections import Counter

from llmgw.console import utf8_stdio

from .generate import format_report, generate, load_chunks, to_qrels, write_jsonl


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="synthdata")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate an eval set from chunks.jsonl")
    g.add_argument("chunks")
    g.add_argument("-o", "--out", default="synthetic_eval.jsonl")
    g.add_argument("--qrels", help="also write retrieval qrels here")
    g.add_argument("--sample", type=int, default=50, help="how many chunks to draw from")
    g.add_argument("--per-chunk", type=int, default=2)
    g.add_argument("--kinds", default="factual,reasoning")
    g.add_argument("--multi-hop", type=int, default=0)
    g.add_argument("--unanswerable", type=int, default=0)
    g.add_argument("--persona", help='e.g. "a first-time customer with no technical background"')
    g.add_argument("--language", help="e.g. French")
    g.add_argument("--min-score", type=int, default=4)
    g.add_argument("-m", "--model")
    g.add_argument("--critic-model")
    g.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from llmgw import LLM

    gen_llm = LLM(args.model, temperature=0.7)
    critic = LLM(args.critic_model or args.model, temperature=0)
    if critic.model == gen_llm.model:
        print(f"note: the critic is the generator ({gen_llm.model}); a model rarely rejects its own questions. "
              "Use --critic-model with a different, ideally stronger, model.", file=sys.stderr)
    report: dict = {}
    rows = generate(
        load_chunks(args.chunks),
        llm=gen_llm,
        critic=critic,
        report=report,
        per_chunk=args.per_chunk,
        kinds=tuple(args.kinds.split(",")),
        sample=args.sample,
        multi_hop=args.multi_hop,
        unanswerable=args.unanswerable,
        persona=args.persona,
        language=args.language,
        min_score=args.min_score,
        seed=args.seed,
    )
    write_jsonl(rows, args.out)
    print(f"{len(rows)} items -> {args.out}  {dict(Counter(r['metadata']['kind'] for r in rows))}")
    print(format_report(report))
    if args.qrels:
        q = to_qrels(rows)
        write_jsonl(q, args.qrels)
        print(f"{len(q)} qrels -> {args.qrels}")


if __name__ == "__main__":
    main()
