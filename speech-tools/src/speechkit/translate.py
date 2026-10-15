from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


class _Translation(BaseModel):
    translation: str


class _Equivalence(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5, description="5 = same meaning incl. numbers, quantities and negations; 1 = meaning changed.")
    issues: list[str] = Field(default_factory=list, description="Specific meaning changes, if any.")


def load_glossary(path: str | Path) -> dict[str, str]:
    """CSV with columns ``source,target`` (header optional)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 2 and r[0].strip()]
    if rows and rows[0][0].lower() == "source":
        rows = rows[1:]
    return {r[0].strip(): r[1].strip() for r in rows}


def _llm(llm):
    if llm is None:
        from llmgw import LLM

        return LLM(temperature=0)
    return llm


def translate(
    text: str,
    target: str,
    *,
    source: str | None = None,
    glossary: dict[str, str] | None = None,
    domain: str | None = None,
    audience: str | None = None,
    llm=None,
) -> str:
    """Faithful translation; glossary terms are mandatory. Numbers, units and quantities are preserved."""
    parts = [f"Translate the text into {target}" + (f" from {source}" if source else "") + "."]
    if domain:
        parts.append(f"Domain: {domain}.")
    if audience:
        parts.append(f"Readers: {audience}; use plain, natural wording they would use.")
    parts.append("Keep every number, unit, quantity and negation exactly. Do not add or omit information.")
    if glossary:
        parts.append("Always use these term translations:\n" + "\n".join(f"- {s} → {t}" for s, t in glossary.items()))
    parts.append(f"\n<text>\n{text}\n</text>")
    return _llm(llm).structured("\n".join(parts), _Translation).translation.strip()


@dataclass
class BackTranslationCheck:
    translation: str
    back_translation: str
    score: int
    issues: list[str]
    reasoning: str


def back_translate_check(
    text: str, target: str, *, source_language: str = "English", llm=None, checker=None, **kw
) -> BackTranslationCheck:
    """Translate, translate back, and have a (preferably different) model judge meaning drift.
    Useful QA for safety-critical content (instructions, medical, legal) where a wrong number
    or a lost "not" is dangerous."""
    llm = _llm(llm)
    checker = checker or llm
    fwd = translate(text, target, llm=llm, **kw)
    back = translate(fwd, source_language, source=target, llm=llm)
    v = checker.structured(
        "Compare the ORIGINAL and the BACK-TRANSLATION (which went through another language). "
        "Judge whether the meaning survived; ignore wording differences.\n\n"
        f"ORIGINAL:\n{text}\n\nBACK-TRANSLATION:\n{back}",
        _Equivalence,
    )
    return BackTranslationCheck(fwd, back, v.score, v.issues, v.reasoning)
