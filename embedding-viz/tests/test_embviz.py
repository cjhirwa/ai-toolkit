import json

from embviz import corpus_map, coverage, embed_queries, kmeans, load_corpus, load_queries, near_duplicates, reduce, save_html
from llmgw import LLM, FakeProvider


def test_end_to_end(tmp_path):
    llm = LLM("fake:x", provider=FakeProvider(), log_dir=False)
    rows = [
        {"id": "a", "text": "thermostat reset power button", "metadata": {"source": "manual.pdf", "page": 1}},
        {"id": "b", "text": "thermostat reset power button", "metadata": {"source": "manual-copy.pdf"}},
        {"id": "c", "text": "refund return policy days", "metadata": {"source": "returns.pdf"}},
        {"id": "d", "text": "office closed public holidays", "metadata": {"source": "company.pdf"}},
    ]
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    corpus = load_corpus(p, llm=llm, embed_model="fake:e")
    assert [x[:2] for x in near_duplicates(corpus, 0.99)] == [("a", "b")]
    labels = kmeans(corpus.vectors, 2)
    assert labels[0] == labels[1]
    coords = reduce(corpus.vectors, "pca")
    assert coords.shape == (4, 2)
    qs = ["thermostat reset", "pizza delivery menu"]
    cov = coverage(corpus, qs, embed_queries(qs, llm=llm, embed_model="fake:e"))
    assert cov[0].query == "pizza delivery menu"  # least covered first
    fig = corpus_map(corpus, coords, queries=qs, query_coords=reduce(embed_queries(qs, llm=llm, embed_model="fake:e"), "pca"))
    save_html(fig, tmp_path / "m.html")
    assert "plotly" in (tmp_path / "m.html").read_text(encoding="utf-8").lower()


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_load_queries_accepts_common_field_names(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"input": "a"}\n\n{"query": "b"}\n{"question": "c"}\n{"input": 5}\n{"other": "x"}\n', encoding="utf-8")
    assert load_queries(p) == ["a", "b", "c"]


def test_kmeans_separates_clear_clusters():
    import numpy as np

    pts = np.array([[1, 0.05], [1, -0.05], [0.98, 0.02], [0.05, 1], [-0.05, 1], [0.02, 0.98]])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    labels = kmeans(pts, 2)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1 and labels[0] != labels[3]
    assert len(kmeans(pts, 50)) == 6  # k larger than the data is clamped


def test_near_duplicates_across_blocks_and_coverage_order(tmp_path):
    llm = LLM("fake:x", provider=FakeProvider(), log_dir=False)
    rows = [{"id": f"c{i}", "text": t} for i, t in enumerate(
        ["reset the thermostat", "office public holidays", "refund return policy", "reset the thermostat"])]
    corpus = load_corpus(_write(tmp_path / "c.jsonl", rows), llm=llm, embed_model="fake:e")
    assert [p[:2] for p in near_duplicates(corpus, 0.99, block=1)] == [("c0", "c3")]  # pair split over blocks
    qs = ["refund policy", "parking at the airport", "thermostat reset"]
    rows = coverage(corpus, qs, embed_queries(qs, llm=llm, embed_model="fake:e"))
    assert rows[0].query == "parking at the airport"
    assert [r.best_similarity for r in rows] == sorted(r.best_similarity for r in rows)


def test_cli_dupes_and_coverage(tmp_path, monkeypatch, capsys):
    from embviz.cli import main

    monkeypatch.setenv("LLMGW_DEFAULT_MODEL", "fake:x")
    monkeypatch.setenv("LLMGW_EMBED_MODEL", "fake:e")
    monkeypatch.setenv("LLMGW_LOG", "0")
    chunks = _write(tmp_path / "c.jsonl", [{"id": "a", "text": "reset the thermostat"},
                                           {"id": "b", "text": "reset the thermostat"},
                                           {"id": "c", "text": "office public holidays"}])
    main(["dupes", str(chunks), "-o", str(tmp_path / "d.csv")])
    assert "1 pairs" in capsys.readouterr().out
    assert (tmp_path / "d.csv").read_text(encoding="utf-8").splitlines()[1].startswith("a,b,")
    queries = _write(tmp_path / "q.jsonl", [{"input": "thermostat reset"}, {"input": "airport parking"}])
    main(["coverage", str(chunks), str(queries), "--threshold", "0.5"])
    out = capsys.readouterr().out
    assert "1/2 queries below 0.5" in out and "airport parking" in out
