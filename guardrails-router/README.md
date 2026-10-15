# guardrails-router (`qroute`)

Classifies each incoming query (in scope or not, safety, intent, language) and sends it to the right handler. It checks the cheapest method first and only moves to a more expensive one when the cheaper one isn't confident:

1. **Rules** (regex, free). Use these for things that must never be missed, such as safety reports.
2. **Embeddings** (one embedding call). The query is compared with labelled examples, and the route is chosen only if it clears a similarity threshold and beats the runner-up by a margin.
3. **LLM** (one structured call). This also detects the query language.

```python
from qroute import Router

router = Router.from_yaml("examples/support_routes.yaml", handlers={
    "in_scope": lambda q, d: rag.answer(q),
    "urgent": lambda q, d: "Unplug the unit now. Connecting you to a person. ...",
    "out_of_scope": lambda q, d: "I can only help with Acme products and orders.",
    "needs_clarification": lambda q, d: "Which product is it, and what does the screen show?",
})
d = router.classify("My thermostat shows E4 after I changed the batteries")
# Decision(route='in_scope', confidence=0.86, stage='embedding', language=None, ...)
answer = router("My thermostat shows E4 after I changed the batteries")
```

See [examples/support_routes.yaml](examples/support_routes.yaml) for the config format: routes with a description and examples, regex rules, a default route, an error route, and LLM guidance.

**When a model stage fails.** If the embedding or LLM call raises (provider down, rate limit, bad output), the query goes to `error_route` with `stage="error"`. Point it at the cautious route, such as asking for clarification or a human hand-off, never at the normal answer path. Without an `error_route` the exception propagates, so a failure is never silently routed on a guess. Rules run first and use no model, so put must-never-miss patterns there.

## Measuring it
You can test the router against any CSV with a column of queries and a column of expected labels:

```
qroute eval routes.yaml labelled.csv --text-col question --label-col route --limit 200
qroute eval routes.yaml labelled.csv --no-llm       # how far do rules + embeddings get on their own?
qroute classify routes.yaml "Can you recommend a good restaurant?"
```

The report shows accuracy, precision and recall per label, how many queries each stage handled, and the misclassified queries. Use it to tune `embed_threshold` and add examples until most traffic is handled before the LLM stage.

**The embedding threshold is calibrated by default** (`embed_threshold: auto`). Similarity scales differ between embedding models, so a fixed 0.8 is strict for one model and loose for another. Instead the router compares the route examples with each other and sets the threshold halfway between how close examples are to their own route and how close they get to other routes; `router.embedding.calibration` shows the numbers. It needs at least two routes and a route with two or more examples, otherwise it falls back to 0.8 with a warning. A number (in Python or the YAML) fixes the threshold instead.
