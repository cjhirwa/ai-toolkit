"""Contract tests for the provider adapters: what each one sends to its SDK, and how it
reads the SDK's reply. Replies are built from the SDKs' own response types, so these
exercise the same parsing as real calls, without network access."""

import json
from types import SimpleNamespace

import pytest

from llmgw import LLM, Attachment, Message, ToolCall, UnsupportedFeature, tool
from llmgw.types import Request


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: Order number.
    """
    return "shipped"


SPEC = order_status.spec
PNG = Attachment(b"\x89PNG fake", "image/png")


def _conversation():
    call = ToolCall("call_1", "order_status", {"order_id": "A-1042"})
    return [
        Message.system("Be brief."),
        Message.user("Where is this order?", [PNG]),
        Message(role="assistant", content="", tool_calls=[call]),
        Message.tool_result(call, "shipped"),
    ]


# ── OpenAI and OpenAI-compatible (Ollama) ─────────────────────────────


class _OpenAIStub:
    def __init__(self, reply: dict):
        from openai.types.chat import ChatCompletion

        self.sent: dict = {}
        self.reply = ChatCompletion.model_validate(reply)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.sent = kw
        return self.reply


def _openai_reply(message: dict, finish="stop", usage=None):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def test_openai_request_translation():
    from llmgw.providers.openai_compat import OpenAIProvider

    stub = _OpenAIStub(_openai_reply({"content": "It shipped."}))
    p = OpenAIProvider(client=stub)
    p.complete(Request(model="gpt-4o-mini", messages=_conversation(), tools=[SPEC], tool_choice="order_status",
                       temperature=0.2, max_tokens=50, json_schema={"type": "object"}, schema_name="Reply"))
    sent = stub.sent
    system, user, assistant, tool_msg = sent["messages"]
    assert system == {"role": "system", "content": "Be brief."}
    assert user["content"][0] == {"type": "text", "text": "Where is this order?"}
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert assistant["tool_calls"][0]["function"] == {"name": "order_status", "arguments": json.dumps({"order_id": "A-1042"})}
    assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "shipped"}
    assert sent["tools"][0]["function"]["name"] == "order_status"
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "order_status"}}
    assert sent["response_format"]["json_schema"]["name"] == "Reply"
    assert sent["temperature"] == 0.2 and sent["max_completion_tokens"] == 50

    p.complete(Request(model="gpt-5-mini", messages=[Message.user("hi")], temperature=0, tool_choice="auto"))
    assert "temperature" not in stub.sent  # reasoning models reject it


def test_openai_rejects_non_image_attachments():
    from llmgw.providers.openai_compat import OpenAIProvider

    with pytest.raises(UnsupportedFeature):
        OpenAIProvider(client=_OpenAIStub(_openai_reply({"content": ""}))).complete(
            Request(model="gpt-4o-mini", messages=[Message.user("read", [Attachment(b"%PDF", "application/pdf")])]))


def test_openai_response_parsing_and_normalisation():
    from llmgw.providers.openai_compat import OllamaProvider, OpenAIProvider

    reply = _openai_reply(
        {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "order_status", "arguments": '{"order_id": "A-1042"}'}},
            {"id": "c2", "type": "function", "function": {"name": "order_status", "arguments": "{not json"}},
        ]},
        finish="length",
        usage={"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107, "prompt_tokens_details": {"cached_tokens": 64}},
    )
    llm = LLM("openai:gpt-4o-mini", provider=OpenAIProvider(client=_OpenAIStub(reply)), log_dir=False)
    r = llm.complete("Where is A-1042?", tools=[order_status])
    assert [c.arguments for c in r.tool_calls] == [{"order_id": "A-1042"}, {"_raw": "{not json"}]
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cached_tokens) == (100, 7, 64)
    assert r.finish_reason == "length" and r.text == ""

    stub = _OpenAIStub(_openai_reply({"content": "ok"}))
    OllamaProvider(client=stub).complete(Request(model="llama3.1", messages=[Message.user("hi")], max_tokens=20))
    assert stub.sent["max_tokens"] == 20 and "max_completion_tokens" not in stub.sent


# ── Gemini ────────────────────────────────────────────────────────────


class _GeminiStub:
    def __init__(self, reply: dict):
        from google.genai import types as gt

        self.sent: dict = {}
        self.reply = gt.GenerateContentResponse.model_validate(reply)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kw):
        self.sent = kw
        return self.reply


def _gemini_reply(parts, finish="STOP", usage=None):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}],
        "usage_metadata": usage or {"prompt_token_count": 10, "candidates_token_count": 4},
    }


def test_gemini_request_translation():
    from llmgw.providers.gemini import GeminiProvider

    stub = _GeminiStub(_gemini_reply([{"text": "It shipped."}]))
    GeminiProvider(client=stub).complete(Request(
        model="gemini-3.6-flash", messages=[*_conversation(), Message.tool_result(ToolCall("call_2", "order_status", {}), "late")],
        tools=[SPEC], tool_choice="required", temperature=0.1, max_tokens=64, json_schema={"type": "object"}))
    contents, cfg = stub.sent["contents"], stub.sent["config"]
    assert [c.role for c in contents] == ["user", "model", "user"]  # system goes to system_instruction
    image, text = contents[0].parts
    assert image.inline_data.mime_type == "image/png" and text.text == "Where is this order?"
    assert contents[1].parts[0].function_call.name == "order_status"
    responses = [p.function_response for p in contents[2].parts]
    assert len(responses) == 2 and responses[0].response == {"result": "shipped"}  # consecutive results batched
    assert cfg.system_instruction == "Be brief."
    assert cfg.automatic_function_calling.disable is True  # tools are never run by the SDK
    assert cfg.tools[0].function_declarations[0].name == "order_status"
    assert str(cfg.tool_config.function_calling_config.mode).endswith("ANY")
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == {"type": "object"}
    assert (cfg.temperature, cfg.max_output_tokens) == (0.1, 64)


def test_gemini_response_parsing_and_normalisation():
    from llmgw.providers.gemini import GeminiProvider

    reply = _gemini_reply(
        [{"text": "thinking...", "thought": True}, {"text": "Order A-1042 "}, {"text": "shipped."},
         {"function_call": {"name": "order_status", "args": {"order_id": "A-1042"}}}],
        finish="MAX_TOKENS",
        usage={"prompt_token_count": 30, "candidates_token_count": 8, "thoughts_token_count": 12, "cached_content_token_count": 16},
    )
    llm = LLM("gemini:gemini-3.6-flash", provider=GeminiProvider(client=_GeminiStub(reply)), log_dir=False)
    r = llm.complete("Where is A-1042?", tools=[order_status])
    assert r.text == "Order A-1042 shipped."  # thought parts are not answer text
    assert r.tool_calls[0].id == "call_3" and r.tool_calls[0].arguments == {"order_id": "A-1042"}
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cached_tokens) == (30, 20, 16)  # thoughts are billed
    assert r.finish_reason == "length"

    blocked = {"candidates": [], "prompt_feedback": {"block_reason": "SAFETY"}}
    r = LLM("gemini:g", provider=GeminiProvider(client=_GeminiStub(blocked)), log_dir=False).complete("x")
    assert r.finish_reason == "content_filter" and r.text == ""


def test_openai_warns_when_temperature_is_dropped(caplog):
    from llmgw.providers.openai_compat import OpenAIProvider

    stub = _OpenAIStub(_openai_reply({"content": "ok"}))
    p = OpenAIProvider(client=stub)
    for _ in range(2):
        p.complete(Request(model="o4-mini-test", messages=[Message.user("hi")], temperature=0))
    assert "temperature" not in stub.sent
    assert caplog.text.count("does not accept temperature") == 1  # once per model, not per call


def test_openai_strict_mode_only_for_compatible_schemas():
    from pydantic import BaseModel

    from llmgw.providers.openai_compat import OpenAIProvider

    class Item(BaseModel):
        name: str
        quantity: int

    class Invoice(BaseModel):
        vendor: str
        items: list[Item]

    class Loose(BaseModel):
        vendor: str
        note: str | None = None  # optional: strict mode would reject this schema

    stub = _OpenAIStub(_openai_reply({"content": '{"vendor": "Acme", "items": []}'}))
    llm = LLM("openai:gpt-4o-mini", provider=OpenAIProvider(client=stub), log_dir=False)
    llm.structured("Invoice?", Invoice)
    fmt = stub.sent["response_format"]["json_schema"]
    assert fmt["strict"] is True
    assert fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["$defs"]["Item"]["additionalProperties"] is False

    stub.reply = stub.reply.model_copy(update={"choices": [stub.reply.choices[0].model_copy(
        update={"message": stub.reply.choices[0].message.model_copy(update={"content": '{"vendor": "Acme"}'})})]})
    llm.structured("Loose?", Loose)
    fmt = stub.sent["response_format"]["json_schema"]
    assert "strict" not in fmt and "additionalProperties" not in fmt["schema"]


def test_streams_report_usage_for_cost_and_logs(tmp_path):
    from openai.types.chat import ChatCompletionChunk

    from llmgw.providers.gemini import GeminiProvider
    from llmgw.providers.openai_compat import OpenAIProvider

    def chunk(content=None, usage=None):
        return ChatCompletionChunk.model_validate({
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
            "choices": [] if content is None else [{"index": 0, "delta": {"content": content}}],
            **({"usage": usage} if usage else {}),
        })

    stub = _OpenAIStub(_openai_reply({"content": ""}))
    stub.chat.completions.create = lambda **kw: (stub.sent.update(kw), iter([
        chunk("Hel"), chunk("lo"), chunk(usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000})]))[1]
    llm = LLM("openai:gpt-4o-mini", provider=OpenAIProvider(client=stub), log_dir=tmp_path / "logs")
    assert "".join(llm.stream("hi")) == "Hello"  # usage is not passed to the caller
    assert stub.sent["stream_options"] == {"include_usage": True}
    assert llm.usage.input_tokens == 1_000_000 and llm.usage.cost_usd == pytest.approx(0.15)
    record = json.loads(next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8"))
    assert record["usage"]["input_tokens"] == 1_000_000

    from google.genai import types as gt

    def gchunk(text, prompt, out):
        return gt.GenerateContentResponse.model_validate({
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
            "usage_metadata": {"prompt_token_count": prompt, "candidates_token_count": out}})

    gstub = SimpleNamespace(models=SimpleNamespace(generate_content_stream=lambda **kw: iter(
        [gchunk("Or", 40, 1), gchunk("der", 40, 2), gchunk(" shipped", 40, 3)])))
    gllm = LLM("gemini:gemini-3.6-flash", provider=GeminiProvider(client=gstub), log_dir=False)
    assert "".join(gllm.stream("x")) == "Order shipped"
    assert (gllm.usage.input_tokens, gllm.usage.output_tokens) == (40, 3)  # last cumulative count, not a sum
