# eval-harness (`evalh`)

Run any pipeline over a dataset and score each output with rule-based metrics and LLM judges. Every run is stored in SQLite, so you can compare it against later runs.

```python
from evalh import load_dataset, run_eval, judge

data = load_dataset("golden.jsonl")          # {id, input, expected?, context?, ...}

def my_rag(ex):
    answer, chunks = rag.answer(ex.input)
    return {"output": answer, "context": [c.text for c in chunks]}   # context feeds faithfulness

run = run_eval(
    my_rag, data,
    ["correctness", "faithfulness", "answer_relevance", "token_f1",
     judge("tone", "5: friendly, plain language, no jargon ...")],
    name="rag-hybrid-v2", concurrency=8,
)
print(run)                      # summary per metric, error rate, latency p50/p95
run.failures("faithfulness")    # examples scoring < 0.5
```

## Metrics

| name | type | needs |
|---|---|---|
| `exact_match`, `contains`, `token_f1` | rule-based | `expected` |
| `json_valid` / `JsonValid(Model)` | rule-based | – |
| `Regex(pattern)` | rule-based | – |
| `correctness` | LLM judge | `expected` |
| `faithfulness` | LLM judge | context, either from the task output or from the example |
| `answer_relevance` | LLM judge | – |
| `judge(name, rubric, needs=...)` | your own rubric | as declared |
| `@metric` | any function `(example, output) -> float/bool/Score` | – |

Judges return a 1–5 score and their reasoning, normalized to 0–1. If an example lacks a field the judge needs, the judge returns `None` and the summary skips it instead of guessing. The judge model is set with `EVALH_JUDGE_MODEL`, or you can pass `judge_llm=LLM(...)`.

## CLI

```
evalh run --task my_pipeline.py:answer --dataset golden.jsonl --metrics correctness,faithfulness --name baseline
evalh runs
evalh show baseline
evalh compare baseline hybrid-v2 --html cmp.html --open    # per-metric deltas + per-example regressions
evalh report hybrid-v2 --open                              # worst examples first, with judge reasoning
```

Run references can be an id, a unique id prefix, or a run name, which selects the latest run with that name. The database path comes from `--db`, or from `EVALH_DB` (default `evals.db`).

## Making runs comparable
- **Stable ids.** Rows without an `id` get one from a hash of their input and expected answer, so reordering or adding rows keeps `compare` matching the right examples. Duplicate ids are an error.
- **Is the change real?** `compare` reports, per metric, the mean change over examples scored in both runs with a 95% paired bootstrap confidence interval, and marks it `significant` only when the interval excludes zero. With small test sets most differences are noise; trust the interval, not the raw averages.
- **Counts.** The summary stores `n_<metric>`: how many examples each mean covers. A judge call that fails scores `None` and is left out, so `compare` warns when two runs scored different numbers of examples.
- **Provenance.** Each run's config records the default, embedding and judge models, the dataset's SHA-256 (from the CLI), the git commit and whether the tree had uncommitted changes.
- **Independent judge.** If the judge would be the same model as the default model the task uses, `run_eval` warns and records `same_model_judge: true`. Set `EVALH_JUDGE_MODEL` to a different, ideally stronger, model.

## Interrupted runs
Each example is saved as soon as it is scored, so a crash or Ctrl-C loses only the examples in flight. `evalh runs` marks unfinished runs; continue one with `evalh run ... --resume <run id or name>` (or `run_eval(..., resume=...)`). Examples already scored are kept, and missing ones, or ones whose task or judge failed, run again under the same run id.

## Tips
- Set `LLM(..., cache=True)` in the pipeline under test so reruns that only change metrics cost nothing.
- `synth-data` produces datasets in this format, and `prompt-registry` versions are a natural thing to compare.
