"""Provider-neutral request/response types."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

_EXTRA_MIME = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    ext = Path(filename).suffix.lower()
    return _EXTRA_MIME.get(ext) or mimetypes.guess_type(filename)[0] or default


@dataclass
class Attachment:
    """Binary input (image, audio, PDF) sent alongside a user message."""

    data: bytes
    mime: str

    @classmethod
    def from_path(cls, path: str | Path) -> Attachment:
        p = Path(path)
        return cls(p.read_bytes(), guess_mime(p.name))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments object


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for tool results
    attachments: list[Attachment] = field(default_factory=list)
    # Provider-native assistant turn. Replayed verbatim to the same provider so
    # things like Gemini thought signatures survive.
    raw: Any = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str, attachments: list[Attachment] | None = None) -> Message:
        return cls("user", content, attachments=list(attachments or []))

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return cls("assistant", content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls("tool", content, tool_call_id=call.id, name=call.name)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_tokens + other.cached_tokens,
            cost,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class Request:
    model: str
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[ToolSpec] | None = None
    tool_choice: str | None = None  # "auto" | "required" | "none" | <tool name>
    json_schema: dict[str, Any] | None = None
    schema_name: str = "response"
    extra: dict[str, Any] = field(default_factory=dict)  # passed through to the provider


@dataclass
class Response:
    text: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    raw: Any = None  # provider-native assistant turn, see Message.raw
    cached: bool = False

    def to_message(self) -> Message:
        return Message("assistant", self.text, tool_calls=list(self.tool_calls), raw=self.raw)


class UnsupportedFeature(Exception):
    """The selected provider cannot do what the request asked for."""


class StructuredOutputError(Exception):
    def __init__(self, message: str, last_text: str):
        super().__init__(message)
        self.last_text = last_text


_FINISH = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop",
    "length": "length", "max_tokens": "length",
    "tool_calls": "tool_calls", "function_call": "tool_calls", "tool_use": "tool_calls",
    "content_filter": "content_filter", "safety": "content_filter", "recitation": "content_filter",
    "blocklist": "content_filter", "prohibited_content": "content_filter", "spii": "content_filter",
}


def normalize_finish(raw: str | None) -> str | None:
    """Map provider finish reasons to one vocabulary: stop, length (cut off at
    max_tokens), tool_calls, content_filter; anything else is returned lower-cased.
    Gemini reports e.g. "FinishReason.MAX_TOKENS", OpenAI and Ollama "length"."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s.startswith("blocked"):
        return "content_filter"
    s = s.rsplit(".", 1)[-1]
    return _FINISH.get(s, s)
