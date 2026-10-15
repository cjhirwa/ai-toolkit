# prompt-registry (`promptreg`)

Versioned prompt templates. Each version is a separate YAML file, `prompts/<name>/v<N>.yaml`, so git history and diffs stay readable.

```yaml
# prompts/rag_answer/v1.yaml
description: Answer a customer question from retrieved passages
model: gemini:gemini-3.6-flash   # optional hint
params: {temperature: 0.2}       # optional hint
system: |
  You are a support assistant for {{ audience }}. Answer only from the passages.
template: |
  Passages:
  {% for p in passages %}[{{ loop.index }}] {{ p }}
  {% endfor %}
  Question: {{ question }}
```

```python
from promptreg import Registry
from llmgw import LLM

reg = Registry("prompts")
p = reg.get("rag_answer")              # latest; or "rag_answer@v1"
r = p.render(passages=docs, question=q, audience="Acme customers")
LLM(p.model).text(r.user, system=r.system, **p.params)
```

Rendering fails loudly when a variable is missing (Jinja `StrictUndefined`). `p.variables` lists the variables a template expects.

## CLI

```
promptreg list
promptreg show rag_answer@v2
promptreg new rag_answer                 # copies the latest version to v(N+1) so you can edit it
promptreg new triage --template "Classify: {{ q }}" --description "router prompt"
promptreg diff rag_answer 1 2
promptreg render rag_answer --var question="How do I reset it?" --var 'passages=["..."]' --var audience=customers
```

The root folder comes from `--root`, or from `PROMPTS_DIR` (default `prompts`).

## Comparing versions with eval-harness
See [examples/compare_versions.py](examples/compare_versions.py). It runs the same dataset through two prompt versions and prints an `evalh` comparison.
