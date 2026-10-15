from __future__ import annotations

import io
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


@dataclass
class Transcript:
    text: str
    language: str | None
    backend: str
    model: str | None = None
    segments: list[dict] = field(default_factory=list)


class STTBackend(Protocol):
    name: str

    def transcribe(self, audio: str | Path | bytes, language: str | None = None, prompt: str | None = None) -> Transcript: ...


def _read(audio: str | Path | bytes) -> tuple[bytes, str]:
    """Bytes plus a filename (for its extension). Accepts paths and http(s) URLs."""
    if isinstance(audio, bytes):
        return audio, "audio.wav"
    s = str(audio)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s, timeout=60) as r:  # noqa: S310 - user-supplied URL by design
            return r.read(), s.rsplit("/", 1)[-1].split("?")[0] or "audio"
    return Path(s).read_bytes(), Path(s).name


class _GeminiOut(BaseModel):
    language: str = Field(description="ISO 639-1 code of the spoken language, e.g. rw, en, fr, sw.")
    text: str = Field(description="Verbatim transcription in the spoken language.")


class GeminiSTT:
    """Audio understanding via Gemini through llmgw. Handles code-switching and
    lower-resource languages (e.g. Swahili, Yoruba, Kinyarwanda) better than many dedicated STT models."""

    name = "gemini"

    def __init__(self, llm=None, model: str | None = None):
        if llm is None:
            from llmgw import LLM

            llm = LLM(model, temperature=0)
        self.llm = llm

    def transcribe(self, audio, language=None, prompt=None) -> Transcript:
        from llmgw import Attachment
        from llmgw.types import guess_mime

        data, name = _read(audio)
        att = Attachment(data, guess_mime(name, default="audio/wav"))
        hint = f" The speaker is expected to use {language}." if language else ""
        ctx = f" Context: {prompt}" if prompt else ""
        out = self.llm.structured(
            "Transcribe this audio verbatim in the language spoken. Do not translate, summarise or "
            f"correct grammar. Mark unclear words as [inaudible].{hint}{ctx}",
            _GeminiOut,
            attachments=[att],
        )
        return Transcript(out.text.strip(), out.language, self.name, self.llm.model)


class OpenAISTT:
    """OpenAI transcription API (``gpt-4o-transcribe``, ``whisper-1``...)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-transcribe", client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client, self.model = client, model

    def transcribe(self, audio, language=None, prompt=None) -> Transcript:
        data, name = _read(audio)
        f = io.BytesIO(data)
        f.name = name
        kwargs = {"model": self.model, "file": f}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        r = self.client.audio.transcriptions.create(**kwargs)
        return Transcript(r.text.strip(), language, self.name, self.model)


class WhisperSTT:
    """Local faster-whisper (``pip install speech-tools[whisper]``). Free and offline."""

    name = "whisper"

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "default"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size

    def transcribe(self, audio, language=None, prompt=None) -> Transcript:
        data, name = _read(audio)
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".wav", delete=False) as tmp:
            tmp.write(data)
        try:
            segments, info = self.model.transcribe(tmp.name, language=language, initial_prompt=prompt)
            segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        return Transcript(" ".join(s["text"] for s in segs), info.language, self.name, self.model_size, segs)


def get_stt(name: str, **kw) -> STTBackend:
    return {"gemini": GeminiSTT, "openai": OpenAISTT, "whisper": WhisperSTT}[name](**kw)
