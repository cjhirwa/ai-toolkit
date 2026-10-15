from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from llmgw.console import utf8_stdio

from .stt import get_stt
from .translate import back_translate_check, load_glossary, translate


def main(argv: list[str] | None = None) -> None:
    utf8_stdio()
    p = argparse.ArgumentParser(prog="speechkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="audio file(s) or URL(s) -> text")
    t.add_argument("audio", nargs="+")
    t.add_argument("--backend", default="gemini", choices=["gemini", "openai", "whisper"])
    t.add_argument("-m", "--model", help="llmgw model for gemini backend / OpenAI model / whisper size")
    t.add_argument("--language", help="expected language, e.g. rw")
    t.add_argument("--prompt", help="domain context or vocabulary hints")
    t.add_argument("--translate", metavar="LANG", help="also translate the transcript")
    t.add_argument("--glossary")
    t.add_argument("-o", "--out", help="write JSONL")
    t.add_argument("--concurrency", type=int, default=4)

    tr = sub.add_parser("translate")
    tr.add_argument("text", help="text, or @file.txt")
    tr.add_argument("--to", required=True)
    tr.add_argument("--from", dest="source")
    tr.add_argument("--glossary")
    tr.add_argument("--domain")
    tr.add_argument("--check", action="store_true", help="back-translate and score meaning drift")

    bt = sub.add_parser("batch", help="transcribe a URL column of a CSV (e.g. audio_question_url)")
    bt.add_argument("csv")
    bt.add_argument("--url-col", required=True)
    bt.add_argument("--id-col")
    bt.add_argument("--limit", type=int)
    bt.add_argument("--backend", default="gemini", choices=["gemini", "openai", "whisper"])
    bt.add_argument("--language")
    bt.add_argument("-o", "--out", default="transcripts.jsonl")
    bt.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args(argv)

    def backend():
        if args.backend == "gemini":
            return get_stt("gemini", model=args.model if hasattr(args, "model") else None)
        if args.backend == "openai":
            return get_stt("openai", **({"model": args.model} if getattr(args, "model", None) else {}))
        return get_stt("whisper", **({"model_size": args.model} if getattr(args, "model", None) else {}))

    if args.cmd == "translate":
        text = open(args.text[1:], encoding="utf-8").read() if args.text.startswith("@") else args.text
        glossary = load_glossary(args.glossary) if args.glossary else None
        if args.check:
            r = back_translate_check(text, args.to, glossary=glossary, domain=args.domain)
            print(r.translation, f"\n\n[back] {r.back_translation}\n[meaning score {r.score}/5] {r.issues or ''}")
        else:
            print(translate(text, args.to, source=args.source, glossary=glossary, domain=args.domain))
        return

    if args.cmd == "transcribe":
        items = [(a, a) for a in args.audio]
        language, prompt = args.language, args.prompt
    else:
        with open(args.csv, encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get(args.url_col)]
        rows = rows[: args.limit] if args.limit else rows
        items = [(r[args.id_col] if args.id_col else r[args.url_col], r[args.url_col]) for r in rows]
        language, prompt = args.language, None

    stt = backend()
    glossary = load_glossary(args.glossary) if getattr(args, "glossary", None) else None
    target = getattr(args, "translate", None)

    def one(item):
        key, src = item
        try:
            tr = stt.transcribe(src, language=language, prompt=prompt)
            rec = {"id": key, "text": tr.text, "language": tr.language, "backend": tr.backend}
            if target:
                rec["translation"] = translate(tr.text, target, source=tr.language, glossary=glossary)
        except Exception as e:  # noqa: BLE001
            rec = {"id": key, "error": f"{type(e).__name__}: {e}"}
        return rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, items))
    out = getattr(args, "out", None)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = sum(1 for r in results if "error" not in r)
        print(f"{ok}/{len(results)} transcribed -> {out}", file=sys.stderr)
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
