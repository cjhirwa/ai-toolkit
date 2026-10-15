import json
from pathlib import Path

from llmgw import LLM, FakeProvider
from qroute import EmbeddingClassifier, Route, Router, evaluate

ROUTES = [
    Route("support", "product support", ["thermostat reset error code", "refund return policy"]),
    Route("sales", "pricing and buying", ["price discount bulk order", "quote for new units"]),
]


def fake_llm(responses=None):
    return LLM("fake:x", provider=FakeProvider(responses), log_dir=False)


def test_rules_win_first():
    r = Router(ROUTES, rules=[(r"\bsmok", "urgent")], embedding=False, llm_fallback=False)
    r.routes["urgent"] = Route("urgent", "")
    d = r.classify("The unit is smoking")
    assert d.route == "urgent" and d.stage == "rules"


def test_embedding_stage_then_llm_fallback():
    llm = fake_llm(lambda req: json.dumps({"reason": "about pricing", "route": "sales", "confidence": 0.9, "language": "en"}))
    router = Router(ROUTES, llm=llm, embed_threshold=0.5)
    d = router.classify("thermostat reset error for my unit")
    assert d.route == "support" and d.stage == "embedding"
    d2 = router.classify("completely unrelated words here")
    assert d2.stage == "llm" and d2.route == "sales" and d2.language == "en"


def test_dispatch_and_evaluate():
    llm = fake_llm()
    handlers = {"support": lambda text, d: f"SUP:{text}", "sales": lambda text, d: f"SAL:{text}"}
    routes = [Route(r.name, r.description, r.examples, handlers[r.name]) for r in ROUTES]
    router = Router(routes, embedding=EmbeddingClassifier(routes, llm, threshold=0.3, margin=0.0), llm_fallback=False)
    assert router("price discount") == "SAL:price discount"
    report = evaluate(router, [("refund policy", "support"), ("discount quote", "sales"), ("error code", "sales")])
    assert round(report["accuracy"], 2) == 0.67 and len(report["errors"]) == 1


def test_example_yaml_loads():
    path = Path(__file__).parents[1] / "examples" / "support_routes.yaml"
    router = Router.from_yaml(path, embedding=False, llm_fallback=False)
    assert router.classify("the unit is smoking").route == "urgent"
    assert router.classify("there is a burning smell").route == "urgent"
    assert router.classify("hello").stage == "default"


def test_model_failure_goes_to_error_route_or_raises(monkeypatch):
    import pytest

    monkeypatch.setattr("time.sleep", lambda s: None)  # skip the gateway's retry backoff

    def down(req):
        raise RuntimeError("503 UNAVAILABLE")

    routes = ROUTES + [Route("clarify", "ask for more detail")]
    safe = Router(routes, llm=fake_llm(down), embedding=False, error_route="clarify")
    d = safe.classify("anything at all")
    assert d.route == "clarify" and d.stage == "error"
    assert safe.classify("smoke") is not None
    with pytest.raises(RuntimeError):
        Router(routes, llm=fake_llm(down), embedding=False).classify("anything at all")
    with pytest.raises(ValueError, match="error_route"):
        Router(routes, llm=fake_llm(down), error_route="nope")


def test_example_yaml_has_a_cautious_error_route():
    path = Path(__file__).parents[1] / "examples" / "support_routes.yaml"
    assert Router.from_yaml(path, embedding=False, llm_fallback=False).error_route == "needs_clarification"


def test_threshold_calibrates_to_the_embedding_model():
    from qroute import EmbeddingClassifier

    routes = [
        Route("support", "", ["thermostat reset error code", "thermostat reset button hold", "error code on thermostat screen"]),
        Route("sales", "", ["price discount bulk order", "bulk order price quote", "discount on a bulk order"]),
    ]
    clf = EmbeddingClassifier(routes, fake_llm())
    cal = clf.calibration
    assert cal["cross_route_p90"] <= clf.threshold <= cal["same_route_median"]
    assert clf.classify("thermostat reset error") is not None  # close to its route: confident
    assert clf.classify("weather forecast for tomorrow") is None  # near nothing: left to the LLM stage
    fixed = EmbeddingClassifier(routes, fake_llm(), threshold=0.99)
    assert fixed.threshold == 0.99 and fixed.calibration == {}
    single = EmbeddingClassifier([Route("only", "", ["a b", "c d"])], fake_llm())
    assert single.threshold == 0.8  # cannot calibrate with one route
