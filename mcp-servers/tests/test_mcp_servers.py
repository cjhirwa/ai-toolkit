import asyncio
import json

from llmgw import FakeProvider, register_provider


def test_retrieval_server_tools(tmp_path, monkeypatch):
    register_provider("fake", FakeProvider())
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(d) for d in [
        {"id": "a", "text": "Hold the power button for ten seconds to reset.", "metadata": {"source": "manual.pdf", "page": 3}},
        {"id": "b", "text": "Our office is closed on public holidays.", "metadata": {"source": "company.pdf", "page": 1}},
    ]))
    from llmgw import LLM
    from retrievekit import Index

    Index.build(chunks, tmp_path / "idx", llm=LLM("fake:x", embed_model="fake:e", log_dir=False))
    monkeypatch.setenv("RETRIEVAL_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("LLMGW_LOG", "0")

    from toolkit_mcp import retrieval

    retrieval._index.cache_clear()
    retrieval._retriever.cache_clear()
    hits = retrieval.search("reset power button", k=1)
    assert hits[0]["id"] == "a" and hits[0]["page"] == 3
    assert retrieval.get_passage("b")["metadata"]["source"] == "company.pdf"
    assert {s["source"] for s in retrieval.list_sources()} == {"manual.pdf", "company.pdf"}

    tools = asyncio.run(retrieval.mcp.list_tools())
    assert {t.name for t in tools} == {"search", "get_passage", "list_sources"}
    search_schema = next(t for t in tools if t.name == "search").input_schema
    assert search_schema["required"] == ["query"]


def test_evals_and_prompts_servers(tmp_path, monkeypatch):
    from evalh import Example, run_eval
    from promptreg import Registry

    db = tmp_path / "e.db"
    data = [Example("1", "q", "a"), Example("2", "q2", "b")]
    run_eval(lambda ex: "a", data, ["exact_match"], name="base", store=db, progress=False)
    run_eval(lambda ex: "b", data, ["exact_match"], name="new", store=db, progress=False)
    monkeypatch.setenv("EVALH_DB", str(db))
    from toolkit_mcp import evals

    assert len(evals.list_runs()) == 2
    cmp = evals.compare_runs("base", "new")
    assert cmp["regressions"][0]["example"] == "1" and cmp["improvements"][0]["example"] == "2"
    assert evals.failing_examples("new", "exact_match")[0]["id"] == "1"

    Registry(tmp_path / "p").create("greet", "Hello {{ name }}")
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path / "p"))
    from toolkit_mcp import prompts

    assert prompts.list_prompts()[0]["variables"] == ["name"]
    assert prompts.render_prompt("greet", {"name": "Ana"})["user"] == "Hello Ana"


def test_http_requires_token_beyond_localhost_and_checks_it():
    import pytest
    from starlette.testclient import TestClient

    from toolkit_mcp import prompts
    from toolkit_mcp._run import http_app

    with pytest.raises(SystemExit, match="MCP_AUTH_TOKEN"):
        http_app(prompts.mcp, "0.0.0.0", None)
    http_app(prompts.mcp, "127.0.0.1", None)  # localhost without a token is allowed

    with TestClient(http_app(prompts.mcp, "0.0.0.0", "s3cret")) as client:  # runs the app's startup
        assert client.post("/mcp", json={}).status_code == 401
        assert client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/mcp", json={}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code != 401  # reaches the MCP app (which rejects the empty body itself)
