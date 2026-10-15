from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src" / "playground" / "app.py")


def test_app_renders_and_runs(monkeypatch, tmp_path):
    from llmgw import FakeProvider, register_provider

    register_provider("fake", FakeProvider(lambda req: f"echo: {req.messages[-1].content}"))
    monkeypatch.setenv("PLAYGROUND_MODELS", "fake:a,fake:b")
    monkeypatch.setenv("PLAYGROUND_HISTORY", str(tmp_path / "h.jsonl"))
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("LLMGW_LOG", "0")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.text_area[0].set_value("hello").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("echo: hello" in m.value for m in at.markdown)
    assert (tmp_path / "h.jsonl").exists()


def _app(monkeypatch, tmp_path, models):
    from llmgw import FakeProvider, register_provider

    register_provider("fake", FakeProvider(lambda req: f"system={req.system!r} user={req.messages[-1].content}"))
    monkeypatch.setenv("PLAYGROUND_MODELS", models)
    monkeypatch.setenv("PLAYGROUND_HISTORY", str(tmp_path / "h.jsonl"))
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path / "prompts"))
    monkeypatch.setenv("LLMGW_LOG", "0")
    return AppTest.from_file(APP, default_timeout=30).run()


def test_registry_prompt_is_rendered(monkeypatch, tmp_path):
    from promptreg import Registry

    Registry(tmp_path / "prompts").create("support", "Customer asks: {{ question }}", system="Be brief.")
    at = _app(monkeypatch, tmp_path, "fake:a")
    at.sidebar.slider[0].set_value(1).run()
    at.selectbox(key="p0").set_value("support@v1").run()
    at.text_area[0].set_value("How do I reset it?").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("system='Be brief.' user=Customer asks: How do I reset it?" in m.value for m in at.markdown)


def test_bad_model_shows_error_without_breaking_other_columns(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, "fake:a,nope:b")
    at.text_area[0].set_value("hello").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Unknown provider" in e.value for e in at.error)
    assert any("user=hello" in m.value for m in at.markdown)


def test_history_tab_lists_previous_runs(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, "fake:a")
    at.text_area[0].set_value("first question").run()
    at.button[0].click().run()
    at = AppTest.from_file(APP, default_timeout=30).run()  # a fresh session reads the saved history
    assert any("first question" in e.label for e in at.expander)
