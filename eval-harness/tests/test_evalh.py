import json

from evalh import (
    Contains,
    ExactMatch,
    Example,
    Store,
    TokenF1,
    compare,
    comparison_report,
    correctness,
    load_dataset,
    metric,
    run_eval,
    run_report,
)
from evalh.metrics import Output
from llmgw import LLM, FakeProvider

DATA = [
    Example("q1", "Capital of Rwanda?", "Kigali"),
    Example("q2", "2+2?", "4"),
    Example("q3", "Colour of the sky?", "blue"),
]


def test_rule_metrics():
    ex = Example("x", "q", "The Kigali")
    assert ExactMatch()(ex, Output("kigali.")).value == 1.0
    assert Contains()(Example("x", "q", ["receipt", "refund"]), Output("Bring your receipt only")).value == 0.5
    assert 0 < TokenF1()(ex, Output("Kigali city")).value < 1
    assert ExactMatch()(Example("x", "q"), Output("a")).value is None


def test_run_store_compare(tmp_path):
    db = tmp_path / "e.db"
    answers_a = {"q1": "Kigali", "q2": "5", "q3": "blue"}
    answers_b = {"q1": "Kigali", "q2": "4", "q3": "green"}

    @metric(name="short")
    def short(ex, out):
        return len(out.text) < 10

    a = run_eval(lambda ex: answers_a[ex.id], DATA, ["exact_match", short], name="a", store=db, progress=False)
    b = run_eval(lambda ex: answers_b[ex.id], DATA, ["exact_match", short], name="b", store=db, progress=False)
    assert round(a.summary["exact_match"], 3) == 0.667
    store = Store(db)
    assert {r["name"] for r in store.list_runs()} == {"a", "b"}
    loaded = store.load("a")
    assert loaded.run_id == a.run_id and len(loaded.results) == 3
    cmp = compare(store.load(a.run_id), store.load(b.run_id))
    assert {d.example_id for d in cmp.regressions} == {"q3"}
    assert {d.example_id for d in cmp.improvements} == {"q2"}
    assert "<html>" in comparison_report(cmp) and "Kigali" in run_report(loaded)


def test_task_errors_are_recorded():
    def boom(ex):
        raise RuntimeError("nope")

    run = run_eval(boom, DATA[:1], ["exact_match"], store=None, progress=False)
    assert run.summary["error_rate"] == 1.0
    assert run.results[0].scores["exact_match"].value == 0.0


def test_llm_judge_with_fake_model():
    judge_llm = LLM("fake:j", provider=FakeProvider(lambda req: json.dumps({"reasoning": "matches", "score": 5})), log_dir=False)
    s = correctness(judge_llm)(DATA[0], Output("Kigali is the capital"))
    assert s.value == 1.0 and s.reason == "matches"
    assert correctness(judge_llm)(Example("x", "q"), Output("a")).value is None  # no reference


def test_load_dataset_formats(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('id,input,expected,context,lang\n1,hi,hello,"[""a"",""b""]",en\n', encoding="utf-8")
    ex = load_dataset(p)[0]
    assert ex.context == ["a", "b"] and ex.metadata == {"lang": "en"}
    j = tmp_path / "d.jsonl"
    j.write_text('{"question": "q", "answer": "a"}\n', encoding="utf-8")
    ex = load_dataset(j, input_key="question", expected_key="answer")[0]
    assert ex.input == "q" and ex.expected == "a"


def test_example_ids_come_from_content_and_duplicates_are_rejected(tmp_path):
    import pytest

    rows = [{"input": "Capital of Rwanda?", "expected": "Kigali"}, {"input": "2+2?", "expected": "4"}]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    b.write_text("\n".join(json.dumps(r) for r in [{"input": "new row"}, *reversed(rows)]), encoding="utf-8")
    ids_a = {e.input: e.id for e in load_dataset(a)}
    ids_b = {e.input: e.id for e in load_dataset(b)}
    assert ids_a["2+2?"] == ids_b["2+2?"] and ids_a["Capital of Rwanda?"] == ids_b["Capital of Rwanda?"]  # survives reordering

    dup = tmp_path / "dup.jsonl"
    dup.write_text("\n".join(json.dumps(r) for r in [rows[0], rows[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate example ids"):
        load_dataset(dup)


def test_counts_metadata_and_same_model_judge_warning(tmp_path, monkeypatch):
    import pytest

    from evalh import compare

    monkeypatch.setenv("LLMGW_DEFAULT_MODEL", "fake:same")
    monkeypatch.delenv("EVALH_JUDGE_MODEL", raising=False)
    judge = LLM("fake:same", provider=FakeProvider(lambda req: json.dumps({"reasoning": "ok", "score": 5})), log_dir=False)
    with pytest.warns(UserWarning, match="same model"):
        run = run_eval(lambda ex: ex.expected, DATA, ["exact_match", correctness(judge)], store=None,
                       judge_llm=judge, progress=False)
    assert run.summary["n_exact_match"] == 3.0 and run.summary["n_correctness"] == 3.0
    assert run.config["judge_model"] == "fake:same" and run.config["same_model_judge"] is True
    assert "default_model" in run.config and "git_commit" in run.config

    flaky = {"n": 0}

    def sometimes_down(req):
        flaky["n"] += 1
        if flaky["n"] == 2:
            raise ValueError("judge returned garbage")
        return json.dumps({"reasoning": "ok", "score": 5})

    judge2 = LLM("fake:other", provider=FakeProvider(sometimes_down), log_dir=False, max_retries=0)
    run2 = run_eval(lambda ex: ex.expected, DATA, ["exact_match", correctness(judge2)], store=None,
                    judge_llm=judge2, progress=False, concurrency=1)
    assert run2.summary["n_correctness"] == 2.0
    report = str(compare(run, run2))
    assert "different numbers of examples" in report and "correctness (3 vs 2)" in report


def test_results_are_saved_as_they_finish_and_runs_resume(tmp_path):
    import pytest

    from evalh import Store

    store = Store(tmp_path / "e.db")
    calls = []

    def crashes_on_third(ex):
        calls.append(ex.id)
        if ex.id == DATA[2].id:
            raise KeyboardInterrupt  # simulate the process dying mid-run
        return ex.expected

    with pytest.raises(KeyboardInterrupt):
        run_eval(crashes_on_third, DATA, ["exact_match"], name="long", store=store, progress=False, concurrency=1)
    partial = store.load("long")
    assert len(partial.results) == 2 and partial.config["status"] == "running"
    assert store.list_runs()[0]["status"] == "running"

    calls.clear()
    run = run_eval(lambda ex: (calls.append(ex.id), ex.expected)[1], DATA, ["exact_match"], store=store,
                   progress=False, resume="long")
    assert calls == [DATA[2].id]  # only the missing example ran again
    assert run.run_id == partial.run_id and run.config["status"] == "complete" and run.config["resumed"]
    assert len(store.load("long").results) == 3 and run.summary["exact_match"] == 1.0


def test_compare_reports_paired_confidence_intervals():
    from evalh import Example, compare
    from evalh.compare import paired_deltas

    data = [Example(f"q{i}", f"question {i}", "yes") for i in range(40)]
    base = run_eval(lambda ex: "yes" if int(ex.id[1:]) % 2 else "no", data, ["exact_match"], store=None, progress=False)
    better = run_eval(lambda ex: "yes", data, ["exact_match"], store=None, progress=False)
    noisy = run_eval(lambda ex: "yes" if int(ex.id[1:]) % 2 == (int(ex.id[1:]) // 20) % 2 else "no", data,
                     ["exact_match"], store=None, progress=False)
    up = paired_deltas(base, better)["exact_match"]
    assert up.mean == 0.5 and up.significant and up.n == 40
    same = paired_deltas(base, noisy)["exact_match"]
    assert same.mean == 0.0 and not same.significant  # half the examples flip each way: noise
    assert "significant" in str(compare(base, better))
