import pytest

from promptreg import Registry


def test_create_get_render_diff(tmp_path):
    reg = Registry(tmp_path)
    reg.create(
        "rag_answer",
        "Passages:\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}\nQ: {{ question }}",
        system="You help {{ audience }}.",
        description="answer from passages",
    )
    v2 = reg.new_version("rag_answer", system="You help {{ audience }}. Cite passages like [1].")
    assert reg.versions("rag_answer") == [1, 2] and v2.version == 2
    p = reg.get("rag_answer@v1")
    assert p.variables == {"passages", "question", "audience"}
    r = p.render(passages=["Hold the button for 10 seconds to reset"], question="How do I reset it?", audience="customers")
    assert r.system == "You help customers." and "[1] Hold the button for 10 seconds to reset" in r.user
    assert r.messages()[0]["role"] == "system"
    with pytest.raises(KeyError):
        p.render(question="x")
    diff = reg.diff("rag_answer", 1, 2)
    assert "+system: You help {{ audience }}. Cite passages like [1]." in diff
    assert reg.names() == ["rag_answer"]
    assert reg.get("rag_answer").version == 2


def test_version_lookup_forms_and_errors(tmp_path):
    reg = Registry(tmp_path)
    reg.create("support", "Q: {{ question }}")
    reg.new_version("support", template="Question: {{ question }}")
    assert reg.get("support", 1).template == "Q: {{ question }}"
    assert reg.get("support", "v1").version == reg.get("support@v1").version == 1
    assert reg.get("support").version == 2
    with pytest.raises(KeyError, match="No prompt named"):
        reg.get("missing")
    with pytest.raises(KeyError, match="no version 7"):
        reg.get("support@v7")
    (tmp_path / "empty").mkdir()
    assert reg.names() == ["support"]  # folders without versions are ignored


def test_new_version_keeps_unchanged_fields(tmp_path):
    reg = Registry(tmp_path)
    reg.create("support", "Q: {{ question }}", system="Be brief.", description="v1",
               model="ollama:llama3.1", params={"temperature": 0.2})
    v2 = reg.new_version("support", description="shorter answers")
    assert (v2.system, v2.model, v2.params, v2.template) == ("Be brief.", "ollama:llama3.1", {"temperature": 0.2}, "Q: {{ question }}")
    assert reg.get("support").description == "shorter answers"


def test_multiline_text_is_saved_as_readable_yaml(tmp_path):
    reg = Registry(tmp_path)
    p = reg.create("support", "Passages:\n{{ passages }}\n\nQuestion: {{ question }}")
    saved = p.path.read_text(encoding="utf-8")
    assert "template: |" in saved  # block style, so diffs show one line per line
    assert reg.get("support").template == p.template


def test_cli_new_list_render_diff(tmp_path, capsys):
    from promptreg.cli import main

    root = ["--root", str(tmp_path)]
    main(root + ["new", "triage", "--template", "Classify: {{ q }}", "--description", "router prompt"])
    main(root + ["list"])
    assert "triage" in capsys.readouterr().out
    main(root + ["render", "triage", "--var", "q=My unit is smoking"])
    assert "Classify: My unit is smoking" in capsys.readouterr().out
    main(root + ["new", "triage"])  # copies the latest version
    main(root + ["diff", "triage", "1", "2"])
    out = capsys.readouterr().out
    assert "created triage@v2" in out and "(no differences)" in out  # an unchanged copy diffs clean
    main(root + ["render", "triage@v1", "--var", 'q=["a", "b"]'])  # JSON values are parsed
    assert "Classify: ['a', 'b']" in capsys.readouterr().out
