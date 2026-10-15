"""Run one dataset through two versions of a prompt and compare them with evalh.

    uv run python prompt-registry/examples/compare_versions.py rag_answer 1 2 golden.jsonl
"""

import sys

from evalh import compare, load_dataset, run_eval
from llmgw import LLM
from promptreg import Registry

name, va, vb, dataset_path = sys.argv[1:5]
reg = Registry("prompts")
data = load_dataset(dataset_path)


def task_for(version: str):
    prompt = reg.get(name, version)
    llm = LLM(prompt.model, cache=True, **prompt.params)

    def task(ex):
        variables = ex.input if isinstance(ex.input, dict) else {"question": ex.input}
        if ex.context and "passages" in prompt.variables:
            variables.setdefault("passages", ex.context)
        r = prompt.render(**variables)
        return {"output": llm.text(r.user, system=r.system), "context": ex.context}

    task.__name__ = prompt.id
    return task


metrics = ["correctness", "faithfulness", "answer_relevance"]
run_a = run_eval(task_for(va), data, metrics, name=f"{name}@v{va}")
run_b = run_eval(task_for(vb), data, metrics, name=f"{name}@v{vb}")
print(compare(run_a, run_b))
