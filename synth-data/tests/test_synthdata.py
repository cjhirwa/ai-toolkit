import json

from llmgw import LLM, FakeProvider
from synthdata import SourceChunk, dedupe, generate, to_qrels

CHUNKS = [
    SourceChunk(f"c{i}", f"Passage {i}. " + "The thermostat warranty lasts two years. " * 8, {"source": "warranty.pdf", "chunk_index": i})
    for i in range(4)
]


def gen_llm():
    def respond(req):
        prompt = req.messages[-1].content
        if "unanswerable" in prompt:
            return json.dumps({"pairs": [{"question": "What does the Acme smart speaker cost?", "answer": "n/a"}]})
        if "multi-hop" in prompt:
            return json.dumps({"pairs": [{"question": "Combine passages on warranty length?", "answer": "2 years"}]})
        return json.dumps({"pairs": [
            {"question": "How long is the thermostat warranty?", "answer": "Two years."},
            {"question": "How long is the thermostat warranty?", "answer": "2 years"},  # duplicate
        ]})

    return LLM("fake:g", provider=FakeProvider(respond), log_dir=False)


def critic_llm(score=5):
    return LLM("fake:c", provider=FakeProvider(lambda req: json.dumps(
        {"reasoning": "ok", "grounded": score, "standalone": score, "realistic": score})), log_dir=False)


def test_generate_filters_dedupes_and_labels():
    rows = generate(CHUNKS, llm=gen_llm(), critic=critic_llm(), sample=2, multi_hop=1, unanswerable=1,
                    min_chunk_chars=10, progress=False)
    kinds = [r["metadata"]["kind"] for r in rows]
    assert kinds.count("factual") + kinds.count("reasoning") == 1  # duplicates collapsed
    assert "multi_hop" in kinds and "unanswerable" in kinds
    assert all(r["id"].startswith("syn-") for r in rows)
    unans = next(r for r in rows if r["metadata"]["kind"] == "unanswerable")
    assert unans["metadata"]["chunk_ids"] == []
    assert len(to_qrels(rows)) == len(rows) - 1


def test_low_scores_are_dropped():
    rows = generate(CHUNKS, llm=gen_llm(), critic=critic_llm(2), sample=2, min_chunk_chars=10, progress=False)
    assert rows == []


def test_dedupe_jaccard():
    rows = [{"input": "how do I reset the thermostat"}, {"input": "How do I reset the thermostat?"}, {"input": "what is the refund policy"}]
    assert len(dedupe(rows)) == 2


def test_ids_are_derived_from_the_question():
    import hashlib

    rows = generate(CHUNKS, llm=gen_llm(), critic=critic_llm(), sample=2, min_chunk_chars=10, progress=False)
    assert all(r["id"].split("-")[1] == hashlib.sha1(r["input"].encode()).hexdigest()[:12] for r in rows)


def test_qrels_carry_answer_spans():
    chunks = [SourceChunk(f"c{i}", "The thermostat warranty lasts two years. " * 8,
                          {"source": "warranty.pdf", "chunk_index": i, "start": i * 400, "end": i * 400 + 328}) for i in range(4)]
    rows = generate(chunks, llm=gen_llm(), critic=critic_llm(), sample=2, min_chunk_chars=10, progress=False)
    q = to_qrels(rows)[0]
    assert q["relevant_spans"][0]["source"] == "warranty.pdf" and q["relevant_spans"][0]["end"] - q["relevant_spans"][0]["start"] == 328


def test_report_explains_what_was_dropped():
    from synthdata.generate import format_report

    report: dict = {}
    generate(CHUNKS, llm=gen_llm(), critic=critic_llm(2), sample=2, unanswerable=1, min_chunk_chars=10,
             progress=False, report=report)
    assert report["factual"]["requested"] == 2 and report["factual"]["rejected_grounded"] >= 1
    assert all(k.get("kept", 0) == 0 for k in report.values())
    text = format_report(report)
    assert "none kept" in text and "grounded" in text

    report = {}
    rows = generate(CHUNKS, llm=gen_llm(), critic=critic_llm(), sample=2, min_chunk_chars=10, progress=False, report=report)
    kept = sum(k.get("kept", 0) for k in report.values())
    made = sum(k.get("generated", 0) for k in report.values())
    assert kept == len(rows) == 1 and sum(k.get("duplicates", 0) for k in report.values()) == made - kept
