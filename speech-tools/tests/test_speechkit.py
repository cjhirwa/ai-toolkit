import json

from llmgw import LLM, FakeProvider
from speechkit import GeminiSTT, back_translate_check, load_glossary, translate


def test_gemini_stt_sends_audio(tmp_path):
    audio = tmp_path / "q.m4a"
    audio.write_bytes(b"\x00\x01fake")
    provider = FakeProvider([json.dumps({"language": "fr", "text": "Le thermostat ne s'allume pas"})])
    tr = GeminiSTT(LLM("fake:x", provider=provider, log_dir=False)).transcribe(audio, language="French")
    assert tr.text == "Le thermostat ne s'allume pas" and tr.language == "fr"
    att = provider.requests[0].messages[0].attachments[0]
    assert att.mime == "audio/mp4" and att.data == b"\x00\x01fake"
    assert "French" in provider.requests[0].messages[0].content


def test_translate_includes_glossary(tmp_path):
    g = tmp_path / "g.csv"
    g.write_text("source,target\nwall plate,plaque murale\n", encoding="utf-8")
    glossary = load_glossary(g)
    provider = FakeProvider([json.dumps({"translation": "Fixez la plaque murale"})])
    out = translate("Fit the wall plate", "French", glossary=glossary, llm=LLM("fake:x", provider=provider, log_dir=False))
    assert out == "Fixez la plaque murale"
    assert "wall plate → plaque murale" in provider.requests[0].messages[0].content


def test_back_translation_check():
    provider = FakeProvider([
        json.dumps({"translation": "Insérez 2 piles"}),
        json.dumps({"translation": "Insert 3 batteries"}),
        json.dumps({"reasoning": "number changed", "score": 1, "issues": ["2 became 3"]}),
    ])
    r = back_translate_check("Insert 2 batteries", "French", llm=LLM("fake:x", provider=provider, log_dir=False))
    assert r.score == 1 and r.issues == ["2 became 3"] and r.back_translation == "Insert 3 batteries"
