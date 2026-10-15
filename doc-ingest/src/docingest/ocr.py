from __future__ import annotations

import io
import os
from typing import Protocol

LLM_OCR_PROMPT = (
    "Transcribe all text in this page image as Markdown. Keep headings, lists and tables "
    "(as Markdown tables). Keep the original language; do not translate or summarise. "
    "Output only the transcription."
)

# The OCR model is configured on its own, independent of LLMGW_DEFAULT_MODEL.
DEFAULT_OCR_MODEL = "gemini:gemini-3.6-flash"


class OCRBackend(Protocol):
    def image_to_text(self, image: bytes, mime: str = "image/png") -> str: ...


class TesseractOCR:
    """Local OCR. Needs the tesseract binary on PATH plus ``pip install doc-ingest[tesseract]``."""

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def image_to_text(self, image: bytes, mime: str = "image/png") -> str:
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(io.BytesIO(image)), lang=self.lang)


class LLMOCR:
    """Vision-model OCR through llmgw. Much better on tables, handwriting and mixed
    layouts; costs tokens.

    The model is ``DOCINGEST_OCR_MODEL`` (default ``gemini:gemini-3.6-flash``), unless
    ``model`` or ``llm`` is passed. It does not follow ``LLMGW_DEFAULT_MODEL``, so the
    general model can change without affecting OCR. It must accept images.

    ``concurrency`` is how many pages are sent at once: ``DOCINGEST_OCR_CONCURRENCY``, or
    1 for Ollama (a local server answers one request at a time) and 4 otherwise.
    """

    def __init__(self, llm=None, prompt: str = LLM_OCR_PROMPT, model: str | None = None,
                 concurrency: int | None = None):
        from llmgw import LLM, load_env, parse_model

        load_env()
        if llm is None:
            # Re-running ingestion should not pay to OCR the same page again: the cache key is
            # the page image's hash plus model and prompt. DOCINGEST_OCR_CACHE=0 turns it off.
            cache_path = os.getenv("DOCINGEST_OCR_CACHE", ".llmgw/ocr-cache.sqlite").strip()
            cache: bool | str = False if cache_path.lower() in ("0", "off", "false", "") else cache_path
            llm = LLM(model or os.getenv("DOCINGEST_OCR_MODEL") or DEFAULT_OCR_MODEL, temperature=0, cache=cache)
        self.llm = llm
        self.prompt = prompt
        env = os.getenv("DOCINGEST_OCR_CONCURRENCY")
        local = parse_model(llm.model)[0] == "ollama"
        self.concurrency = max(1, concurrency or (int(env) if env else (1 if local else 4)))

    def image_to_text(self, image: bytes, mime: str = "image/png") -> str:
        from llmgw import Attachment

        return self.llm.text(self.prompt, attachments=[Attachment(image, mime)])


def get_ocr(name: str | None, **kw) -> OCRBackend | None:
    if not name or name == "none":
        return None
    if name == "tesseract":
        return TesseractOCR(**kw)
    if name in ("llm", "gemini", "vision"):
        return LLMOCR(**kw)
    raise ValueError(f"Unknown OCR backend {name!r} (tesseract | llm)")
