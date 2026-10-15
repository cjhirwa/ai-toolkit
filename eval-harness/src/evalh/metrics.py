"""Metrics take (example, output) and return a Score in [0, 1] (or None when not applicable)."""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .dataset import Example


@dataclass
class Output:
    text: str
    context: list[str] | None = None
    latency_ms: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    name: str
    value: float | None
    reason: str = ""


class Metric(Protocol):
    name: str

    def __call__(self, example: Example, output: Output, /) -> Score: ...  # positional: implementations name them freely


def normalize(s: Any) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _expected_list(ex: Example) -> list[str]:
    if ex.expected is None:
        return []
    return [str(e) for e in ex.expected] if isinstance(ex.expected, list) else [str(ex.expected)]


class ExactMatch:
    """Normalized output equals the expected answer (or any of a list)."""

    name = "exact_match"

    def __call__(self, ex: Example, out: Output) -> Score:
        exp = _expected_list(ex)
        if not exp:
            return Score(self.name, None, "no expected answer")
        return Score(self.name, float(any(normalize(out.text) == normalize(e) for e in exp)))


class Contains:
    """Every expected string appears in the output (case/punctuation-insensitive)."""

    name = "contains"

    def __call__(self, ex: Example, out: Output) -> Score:
        exp = _expected_list(ex)
        if not exp:
            return Score(self.name, None, "no expected answer")
        got = normalize(out.text)
        missing = [e for e in exp if normalize(e) not in got]
        return Score(self.name, 1.0 - len(missing) / len(exp), f"missing: {missing}" if missing else "")


class Regex:
    def __init__(self, pattern: str, name: str = "regex", flags: int = re.IGNORECASE):
        self.pattern = re.compile(pattern, flags)
        self.name = name

    def __call__(self, ex: Example, out: Output) -> Score:
        return Score(self.name, float(bool(self.pattern.search(out.text))))


class TokenF1:
    """SQuAD-style token overlap F1 against the best-matching expected answer."""

    name = "token_f1"

    def __call__(self, ex: Example, out: Output) -> Score:
        exp = _expected_list(ex)
        if not exp:
            return Score(self.name, None, "no expected answer")
        pred = normalize(out.text).split()

        def f1(ref: str) -> float:
            gold = normalize(ref).split()
            common = sum((Counter(pred) & Counter(gold)).values())
            if not common:
                return 0.0
            p, r = common / len(pred), common / len(gold)
            return 2 * p * r / (p + r)

        return Score(self.name, max(f1(e) for e in exp))


class JsonValid:
    """Output parses as JSON (and validates against ``schema`` if given)."""

    def __init__(self, schema: type[BaseModel] | None = None, name: str = "json_valid"):
        self.schema = schema
        self.name = name

    def __call__(self, ex: Example, out: Output) -> Score:
        try:
            data = json.loads(out.text)
            if self.schema:
                self.schema.model_validate(data)
            return Score(self.name, 1.0)
        except Exception as e:  # noqa: BLE001
            return Score(self.name, 0.0, str(e)[:300])


class FunctionMetric:
    """Wrap ``fn(example, output) -> float | bool | Score``."""

    def __init__(self, fn: Callable[[Example, Output], Any], name: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__

    def __call__(self, ex: Example, out: Output) -> Score:
        v = self.fn(ex, out)
        if isinstance(v, Score):
            return v
        return Score(self.name, None if v is None else float(v))


def metric(fn: Callable | None = None, *, name: str | None = None):
    """Decorator form of FunctionMetric."""
    if fn is None:
        return lambda f: FunctionMetric(f, name)
    return FunctionMetric(fn, name)


# ── LLM judges ───────────────────────────────────────────────────────

class Verdict(BaseModel):
    reasoning: str = Field(description="Brief justification, citing specifics.")
    score: int = Field(ge=1, le=5)


_JUDGE_SYSTEM = (
    "You are a strict, consistent evaluator. Judge only what the rubric asks. "
    "Do not reward length or style unless the rubric says so."
)


def _judge_llm():
    from llmgw import LLM

    return LLM(os.getenv("EVALH_JUDGE_MODEL") or None, temperature=0)


class LLMJudge:
    """Rubric-based 1–5 judge, normalized to [0, 1].

    ``needs`` lists the fields the rubric depends on; when the example lacks
    one (e.g. no reference answer) the score is None instead of a guess."""

    def __init__(self, name: str, rubric: str, *, needs: tuple[str, ...] = ("input", "output"), llm=None):
        self.name = name
        self.rubric = rubric
        self.needs = needs
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _judge_llm()
        return self._llm

    def __call__(self, ex: Example, out: Output) -> Score:
        context = out.context if out.context is not None else ex.context
        have = {"input": ex.input, "output": out.text, "expected": ex.expected, "context": context}
        missing = [n for n in self.needs if not have.get(n)]
        if missing:
            return Score(self.name, None, f"skipped: no {', '.join(missing)}")
        parts = [f"## Rubric: {self.name}\n{self.rubric}\n"]
        q = ex.input if isinstance(ex.input, str) else json.dumps(ex.input, ensure_ascii=False)
        parts.append(f"## Question / input\n{q}\n")
        if "context" in self.needs and context:
            parts.append("## Retrieved context\n" + "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context)) + "\n")
        if "expected" in self.needs:
            parts.append(f"## Reference answer\n{ex.expected}\n")
        parts.append(f"## Response to evaluate\n{out.text}\n")
        parts.append("Score the response from 1 (worst) to 5 (best) on this rubric only.")
        v = self.llm.structured("\n".join(parts), Verdict, system=_JUDGE_SYSTEM)
        return Score(self.name, (v.score - 1) / 4, v.reasoning)


def faithfulness(llm=None) -> LLMJudge:
    return LLMJudge(
        "faithfulness",
        "5: every factual claim in the response is supported by the retrieved context. "
        "3: mostly supported, with minor unsupported details. "
        "1: key claims are unsupported by or contradict the context. "
        "A response that correctly says the context lacks the answer is faithful.",
        needs=("context", "output"),
        llm=llm,
    )


def answer_relevance(llm=None) -> LLMJudge:
    return LLMJudge(
        "answer_relevance",
        "5: directly and completely addresses what was asked. "
        "3: partially addresses it or adds a lot of irrelevant material. "
        "1: does not address the question.",
        needs=("input", "output"),
        llm=llm,
    )


def correctness(llm=None) -> LLMJudge:
    return LLMJudge(
        "correctness",
        "Compare against the reference answer. 5: same essential facts, no contradictions. "
        "3: partly correct or missing important facts. 1: wrong or contradicts the reference. "
        "Wording differences do not matter.",
        needs=("input", "expected", "output"),
        llm=llm,
    )


def judge(name: str, rubric: str, *, needs: tuple[str, ...] = ("input", "output"), llm=None) -> LLMJudge:
    return LLMJudge(name, rubric, needs=needs, llm=llm)


BUILTIN: dict[str, Callable[..., Metric]] = {
    "exact_match": lambda llm=None: ExactMatch(),
    "contains": lambda llm=None: Contains(),
    "token_f1": lambda llm=None: TokenF1(),
    "json_valid": lambda llm=None: JsonValid(),
    "faithfulness": faithfulness,
    "answer_relevance": answer_relevance,
    "correctness": correctness,
}


def resolve_metrics(items: list[str | Metric], judge_llm=None) -> list[Metric]:
    out = []
    for m in items:
        if isinstance(m, str):
            if m not in BUILTIN:
                raise KeyError(f"Unknown metric {m!r}. Built-in: {sorted(BUILTIN)}")
            out.append(BUILTIN[m](judge_llm))
        else:
            out.append(m)
    return out
