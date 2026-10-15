"""Cascade: regex rules (free) → embedding nearest-examples (cheap) → LLM classifier (accurate).
Each stage only answers when it is confident; otherwise the next stage decides."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import Field, create_model

log = logging.getLogger("qroute")


@dataclass
class Route:
    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    handler: Callable[..., Any] | None = None


@dataclass
class Decision:
    route: str
    confidence: float
    stage: str  # "rules" | "embedding" | "llm" | "default" | "error"
    reason: str = ""
    language: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


class RuleClassifier:
    """Regex rules for things that must never be missed (e.g. danger signs) or are trivially obvious."""

    def __init__(self, rules: list[tuple[str, str]]):
        self.rules = [(re.compile(p, re.IGNORECASE), route) for p, route in rules]

    def classify(self, text: str) -> Decision | None:
        for pattern, route in self.rules:
            if m := pattern.search(text):
                return Decision(route, 1.0, "rules", f"matched {m.group(0)!r}")
        return None


class EmbeddingClassifier:
    """Scores each route by the mean of its ``top`` most similar examples (default: the
    single nearest one). Confident when the best route clears ``threshold`` and beats
    the runner-up by ``margin``."""

    def __init__(self, routes: list[Route], llm=None, threshold: float | str = "auto", margin: float = 0.05, top: int = 1):
        if llm is None:
            from llmgw import LLM

            llm = LLM()
        self.llm, self.margin, self.top = llm, margin, top
        self.labels: list[str] = []
        texts: list[str] = []
        for r in routes:
            for ex in r.examples:
                self.labels.append(r.name)
                texts.append(ex)
        if texts:
            e = np.asarray(llm.embed(texts, task="query"), dtype=np.float32)
            self.matrix = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
        else:
            self.matrix = np.zeros((0, 1), dtype=np.float32)
        self.label_arr = np.array(self.labels)
        self.calibration: dict[str, float] = {}
        self.threshold = self._calibrate() if threshold == "auto" else float(threshold)

    def _calibrate(self, fallback: float = 0.8) -> float:
        """Similarity scales differ a lot between embedding models (0.8 is high for one and
        ordinary for another), so derive the threshold from the route examples themselves:
        halfway between how similar examples are to their own route (median) and how similar
        they get to other routes (90th percentile)."""
        if len(set(self.labels)) < 2:
            log.warning("embedding threshold: need examples for at least two routes to calibrate; using %s", fallback)
            return fallback
        sims = self.matrix @ self.matrix.T
        np.fill_diagonal(sims, -np.inf)
        same, cross = [], []
        for i, label in enumerate(self.labels):
            own, other = self.label_arr == label, self.label_arr != label
            if own.sum() > 1:
                same.append(float(sims[i, own].max()))
            cross.append(float(sims[i, other].max()))
        if not same:
            log.warning("embedding threshold: need a route with two or more examples to calibrate; using %s", fallback)
            return fallback
        same_mid, cross_high = float(np.median(same)), float(np.percentile(cross, 90))
        self.calibration = {"same_route_median": same_mid, "cross_route_p90": cross_high}
        if same_mid <= cross_high:
            log.warning("embedding threshold: route examples are closer to other routes (%.2f) than to their own "
                        "(%.2f); add more, similar examples per route, or more queries will go to the LLM stage",
                        cross_high, same_mid)
        return max(cross_high, (same_mid + cross_high) / 2)

    def scores(self, text: str) -> dict[str, float]:
        if not len(self.labels):
            return {}
        q = np.asarray(self.llm.embed([text], task="query")[0], dtype=np.float32)
        sims = self.matrix @ (q / max(float(np.linalg.norm(q)), 1e-12))
        out = {}
        for label in dict.fromkeys(self.labels):
            s = np.sort(sims[self.label_arr == label])[::-1][: self.top]
            out[label] = float(s.mean())
        return out

    def classify(self, text: str) -> Decision | None:
        scores = self.scores(text)
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        decision = Decision(best, s1, "embedding", f"similarity {s1:.2f} vs next {s2:.2f}", scores=scores)
        return decision if s1 >= self.threshold and s1 - s2 >= self.margin else None


class LLMClassifier:
    def __init__(self, routes: list[Route], llm=None, instructions: str | None = None, detect_language: bool = True):
        if llm is None:
            from llmgw import LLM

            llm = LLM(temperature=0)
        self.llm, self.routes, self.instructions = llm, routes, instructions
        names = tuple(r.name for r in routes)
        fields: dict[str, Any] = {
            "reason": (str, Field(description="One sentence on why.")),
            "route": (Literal[names], ...),
            "confidence": (float, Field(ge=0, le=1)),
        }
        if detect_language:
            fields["language"] = (str, Field(description="ISO 639-1 code of the query language, e.g. en, rw, fr, sw."))
        self.schema = create_model("RouteDecision", **fields)

    def classify(self, text: str) -> Decision:
        lines = [f"- {r.name}: {r.description}" + (f" (e.g. {'; '.join(r.examples[:3])})" if r.examples else "") for r in self.routes]
        prompt = (
            "Classify the user query into exactly one route.\n\nRoutes:\n" + "\n".join(lines)
            + (f"\n\nGuidance:\n{self.instructions}" if self.instructions else "")
            + f"\n\nQuery:\n<<<\n{text}\n>>>"
        )
        out = self.llm.structured(prompt, self.schema)
        return Decision(out.route, float(out.confidence), "llm", out.reason, getattr(out, "language", None))


class Router:
    def __init__(
        self,
        routes: list[Route],
        *,
        rules: list[tuple[str, str]] | None = None,
        embedding: bool | EmbeddingClassifier = True,
        llm_fallback: bool | LLMClassifier = True,
        default_route: str | None = None,
        llm=None,
        embed_threshold: float | str = "auto",
        instructions: str | None = None,
        error_route: str | None = None,
    ):
        """``error_route`` is where a query goes when the embedding or LLM stage fails
        (provider down, rate limit, bad output). Make it the cautious route, such as asking
        for clarification or handing off to a person, not the normal answer path. Without
        it, the error propagates rather than the query being routed on a guess."""
        self.routes = {r.name: r for r in routes}
        self.rules = RuleClassifier(rules) if rules else None
        has_examples = any(r.examples for r in routes)
        if isinstance(embedding, EmbeddingClassifier):
            self.embedding: EmbeddingClassifier | None = embedding
        else:
            self.embedding = EmbeddingClassifier(routes, llm, embed_threshold) if embedding and has_examples else None
        if isinstance(llm_fallback, LLMClassifier):
            self.llm_classifier: LLMClassifier | None = llm_fallback
        else:
            self.llm_classifier = LLMClassifier(routes, llm, instructions) if llm_fallback else None
        self.default_route = default_route or routes[0].name
        if error_route is not None and error_route not in self.routes:
            raise ValueError(f"error_route {error_route!r} is not one of the routes {sorted(self.routes)}")
        self.error_route = error_route

    @classmethod
    def from_yaml(cls, path: str | Path, handlers: dict[str, Callable] | None = None, **kw) -> Router:
        """routes.yaml: {routes: [{name, description, examples}], rules: [{pattern, route}], default_route, instructions}"""
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        handlers = handlers or {}
        routes = [Route(r["name"], r.get("description", ""), r.get("examples") or [], handlers.get(r["name"])) for r in cfg["routes"]]
        rules = [(r["pattern"], r["route"]) for r in cfg.get("rules") or []]
        kw.setdefault("default_route", cfg.get("default_route"))
        kw.setdefault("instructions", cfg.get("instructions"))
        kw.setdefault("error_route", cfg.get("error_route"))
        if "embed_threshold" in cfg:
            kw.setdefault("embed_threshold", cfg["embed_threshold"])
        return cls(routes, rules=rules, **kw)

    def classify(self, text: str) -> Decision:
        if self.rules and (d := self.rules.classify(text)):
            return d
        try:
            return self._classify_models(text)
        except Exception as e:
            if self.error_route is None:
                raise
            log.warning("routing failed (%s: %s); sending query to %r", type(e).__name__, e, self.error_route)
            return Decision(self.error_route, 0.0, "error", f"classification failed: {type(e).__name__}")

    def _classify_models(self, text: str) -> Decision:
        emb_scores: dict[str, float] = {}
        if self.embedding:
            if d := self.embedding.classify(text):
                return d
            emb_scores = self.embedding.scores(text) if self.llm_classifier is None else {}
        if self.llm_classifier:
            return self.llm_classifier.classify(text)
        if emb_scores:
            best = max(emb_scores, key=lambda r: emb_scores[r])
            return Decision(best, emb_scores[best], "embedding", "below threshold; no LLM fallback", scores=emb_scores)
        return Decision(self.default_route, 0.0, "default", "no stage was confident")

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> Any:
        """Classify and call the route's handler as ``handler(text, decision, *args, **kwargs)``."""
        decision = self.classify(text)
        handler = self.routes[decision.route].handler
        if handler is None:
            raise LookupError(f"Route {decision.route!r} has no handler")
        return handler(text, decision, *args, **kwargs)
