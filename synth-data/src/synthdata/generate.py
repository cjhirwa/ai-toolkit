"""chunks → candidate Q&A → critic filter → dedupe → eval-harness dataset (+ qrels).

Question kinds:
- factual:      answer is stated in one chunk
- reasoning:    needs applying/combining facts within one chunk
- multi_hop:    needs two related chunks (same source, nearby)
- unanswerable: on-topic but NOT answered by the chunk; expected answer is a refusal
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import threading
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

UNANSWERABLE_EXPECTED = "The provided material does not contain this information."


@dataclass
class SourceChunk:
    id: str
    text: str
    metadata: dict[str, Any]


def load_chunks(path: str | Path) -> list[SourceChunk]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(SourceChunk(str(d["id"]), d["text"], d.get("metadata") or {}))
    return out


class QA(BaseModel):
    question: str = Field(description="A self-contained question; never refers to 'the text' or 'the passage'.")
    answer: str = Field(description="Concise reference answer, using only the material.")


class QAList(BaseModel):
    pairs: list[QA]


class Critique(BaseModel):
    reasoning: str
    grounded: int = Field(ge=1, le=5, description="Answer fully supported by the material (for unanswerable: question truly NOT answered by it).")
    standalone: int = Field(ge=1, le=5, description="Question understandable without seeing the material.")
    realistic: int = Field(ge=1, le=5, description="A real user in the target audience would plausibly ask this.")


_KIND_GUIDE = {
    "factual": "Each question's answer must be stated directly in the material.",
    "reasoning": "Each question must require applying or combining facts from the material (e.g. a short case scenario), not just looking one up.",
    "multi_hop": "Each question must need information from BOTH passages to answer.",
    "unanswerable": (
        "Each question must be on the same topic and sound answerable, but the material must NOT contain the answer. "
        f"Set every answer to exactly: {UNANSWERABLE_EXPECTED}"
    ),
}


def _gen_prompt(kind: str, texts: list[str], n: int, persona: str | None, language: str | None) -> str:
    material = "\n\n".join(f"<passage {i + 1}>\n{t}\n</passage {i + 1}>" for i, t in enumerate(texts))
    who = f" Write them as {persona} would ask them, in their own words." if persona else ""
    lang = f" Write questions and answers in {language}." if language else ""
    return (
        f"Write {n} {kind.replace('_', '-')} question(s) with reference answers from the material below. "
        f"{_KIND_GUIDE[kind]}{who}{lang} Questions must be self-contained: never mention 'the passage', "
        f"'the text' or 'the document'. Vary phrasing and difficulty.\n\n{material}"
    )


def _critic_prompt(kind: str, texts: list[str], qa: QA, persona: str | None) -> str:
    material = "\n\n".join(texts)
    aud = f"\nTarget audience: {persona}." if persona else ""
    note = " This is meant to be UNANSWERABLE from the material: score 'grounded' high only if the material does not answer it." if kind == "unanswerable" else ""
    return (
        f"Rate this generated evaluation item.{note}{aud}\n\nMaterial:\n{material}\n\n"
        f"Question: {qa.question}\nAnswer: {qa.answer}"
    )


def _norm_tokens(q: str) -> set[str]:
    return set(re.findall(r"\w+", q.lower()))


def dedupe(rows: list[dict], jaccard: float = 0.8) -> list[dict]:
    kept: list[dict] = []
    seen: list[set[str]] = []
    for r in rows:
        toks = _norm_tokens(r["input"])
        if any(len(toks & s) / max(len(toks | s), 1) >= jaccard for s in seen):
            continue
        seen.append(toks)
        kept.append(r)
    return kept


def _neighbours(chunks: list[SourceChunk]) -> list[tuple[SourceChunk, SourceChunk]]:
    by_source: dict[str, list[SourceChunk]] = {}
    for c in chunks:
        by_source.setdefault(str(c.metadata.get("source", "")), []).append(c)
    pairs = []
    for group in by_source.values():
        group.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        pairs += [(group[i], group[i + 2 if i + 2 < len(group) else i + 1]) for i in range(len(group) - 1)]
    return pairs


def generate(
    chunks: Sequence[SourceChunk],
    *,
    llm=None,
    critic=None,
    per_chunk: int = 2,
    kinds: Sequence[Literal["factual", "reasoning"]] = ("factual", "reasoning"),
    sample: int | None = None,
    multi_hop: int = 0,
    unanswerable: int = 0,
    persona: str | None = None,
    language: str | None = None,
    min_score: int = 4,
    min_chunk_chars: int = 200,
    seed: int = 0,
    concurrency: int = 6,
    progress: bool = True,
    report: dict | None = None,
) -> list[dict]:
    """Returns eval-harness rows: {id, input, expected, context, metadata{chunk_ids, kind, source, critique}}.

    Pass a dict as ``report`` to get, per question kind, how many items were requested,
    generated, rejected by the critic (and on which criterion), lost to failed calls, removed
    as duplicates and kept. Small models often drop most items, so check it."""
    stats: dict[str, dict[str, int]] = report if report is not None else {}
    stats_lock = threading.Lock()

    def count(kind: str, key: str, n: int = 1) -> None:
        with stats_lock:
            k = stats.setdefault(kind, {})
            k[key] = k.get(key, 0) + n

    from llmgw import LLM

    llm = llm or LLM(temperature=0.7)
    critic = critic or LLM(temperature=0)
    rng = random.Random(seed)
    usable = [c for c in chunks if len(c.text) >= min_chunk_chars]
    picked = rng.sample(usable, min(sample, len(usable))) if sample else list(usable)

    jobs: list[tuple[str, list[SourceChunk], int]] = []
    for i, c in enumerate(picked):
        jobs.append((kinds[i % len(kinds)], [c], per_chunk))
    pairs = _neighbours(list(usable))
    for a, b in rng.sample(pairs, min(multi_hop, len(pairs))):
        jobs.append(("multi_hop", [a, b], 1))
    for c in rng.sample(usable, min(unanswerable, len(usable))):
        jobs.append(("unanswerable", [c], 1))

    done = 0
    for kind, _, n in jobs:
        count(kind, "requested", n)

    def run(job: tuple[str, list[SourceChunk], int]) -> list[dict]:
        nonlocal done
        kind, cs, n = job
        texts = [c.text for c in cs]
        try:
            qas = llm.structured(_gen_prompt(kind, texts, n, persona, language), QAList).pairs[:n]
        except Exception as e:  # noqa: BLE001
            print(f"\ngeneration failed for {cs[0].id}: {e}", file=sys.stderr)
            count(kind, "generation_failed", n)
            qas = []
        count(kind, "generated", len(qas))
        rows = []
        for qa in qas:
            try:
                crit = critic.structured(_critic_prompt(kind, texts, qa, persona), Critique)
            except Exception as e:  # noqa: BLE001
                print(f"\ncritique failed: {e}", file=sys.stderr)
                count(kind, "critique_failed")
                continue
            low = [a for a in ("grounded", "standalone", "realistic") if getattr(crit, a) < min_score]
            if low:
                count(kind, "rejected")
                for a in low:
                    count(kind, f"rejected_{a}")
                continue
            rows.append({
                "input": qa.question,
                "expected": qa.answer,
                "context": texts,
                "metadata": {
                    "kind": kind,
                    "chunk_ids": [] if kind == "unanswerable" else [c.id for c in cs],
                    "source_chunk_ids": [c.id for c in cs],
                    # Where the answer passage sits in the extracted document text. Unlike chunk
                    # ids, this stays valid when the corpus is re-chunked differently.
                    "spans": [] if kind == "unanswerable" else [
                        {"source": c.metadata.get("source"), "start": c.metadata["start"], "end": c.metadata["end"]}
                        for c in cs if c.metadata.get("start") is not None and c.metadata.get("end") is not None],
                    "source": cs[0].metadata.get("source"),
                    "critique": crit.model_dump(),
                },
            })
        done += 1
        if progress:
            print(f"\r[{done}/{len(jobs)}] generating", end="", file=sys.stderr, flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = [r for batch in pool.map(run, jobs) for r in batch]
    if progress:
        print(file=sys.stderr)
    before = Counter(r["metadata"]["kind"] for r in rows)
    rows = dedupe(rows)
    after = Counter(r["metadata"]["kind"] for r in rows)
    for kind in set(before) | set(stats):
        count(kind, "duplicates", before[kind] - after[kind])
        count(kind, "kept", after[kind])
    seen: dict[str, int] = {}
    for r in rows:
        # From the question text, so regenerating or filtering a set keeps ids stable.
        base = "syn-" + hashlib.sha1(r["input"].encode()).hexdigest()[:12]
        seen[base] = seen.get(base, 0) + 1
        r["id"] = base if seen[base] == 1 else f"{base}-{seen[base]}"
    return [{"id": r["id"], **{k: v for k, v in r.items() if k != "id"}} for r in rows]


def format_report(report: dict[str, dict[str, int]]) -> str:
    """One line per question kind from generate(report=...), flagging kinds that came out empty."""
    lines = [f"{'kind':14}{'asked':>7}{'made':>6}{'rejected':>10}{'failed':>8}{'dupes':>7}{'kept':>6}  rejected for"]
    for kind, c in sorted(report.items()):
        failed = c.get("generation_failed", 0) + c.get("critique_failed", 0)
        why = ", ".join(f"{a} {c[f'rejected_{a}']}" for a in ("grounded", "standalone", "realistic") if c.get(f"rejected_{a}"))
        flag = "   <- none kept" if c.get("requested") and not c.get("kept") else ""
        lines.append(f"{kind:14}{c.get('requested', 0):>7}{c.get('generated', 0):>6}{c.get('rejected', 0):>10}"
                     f"{failed:>8}{c.get('duplicates', 0):>7}{c.get('kept', 0):>6}  {why}{flag}")
    return "\n".join(lines)


def to_qrels(rows: list[dict]) -> list[dict]:
    """Retrieval labels: chunk ids, plus answer spans (source + character offsets) when the
    chunks carried offsets, so ``retrievekit bench`` can score other chunkings too."""
    out = []
    for r in rows:
        m = r["metadata"]
        if m["chunk_ids"]:
            out.append({"query": r["input"], "relevant_ids": m["chunk_ids"],
                        **({"relevant_spans": m["spans"]} if m.get("spans") else {})})
    return out


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
