# agent-loop (`agentloop`)

A tool-calling agent loop that is short enough to read in full ([agent.py](src/agentloop/agent.py)), with a JSONL trace of every step and an HTML viewer for those traces.

```python
from agentloop import Agent, tool
from retrievekit import Index

idx = Index.load("index/")
search_idx = idx.retriever("hybrid")

@tool
def search(query: str, k: int = 5) -> list[dict]:
    """Search the product documentation.

    Args:
        query: What to look for.
        k: Number of passages.
    """
    return [{"id": h.id, "source": h.metadata.get("source"), "page": h.metadata.get("page"), "text": h.text}
            for h in search_idx.retrieve(query, k)]

agent = Agent(tools=[search], system="Answer from the documentation and cite [source p.N].", max_steps=6)
result = agent.run("My thermostat shows error E4 after I changed the batteries. What should I do?")
print(result.output, result.usage, result.trace_path)
```

## What the loop does
- Tools are plain functions. Their type hints and docstring become the JSON schema, via `llmgw.tool`.
- If a tool raises an exception, the error goes back to the model as the tool result so it can recover. An unknown tool name gets the list of available tools.
- When `max_steps` is reached, the agent makes one final call with tools disabled and returns a best-effort answer (`stopped="max_steps"`).
- If the final answer was cut off at `max_tokens`, `stopped` is `"truncated"`, not `"final"`; raise `max_tokens` on the LLM.
- Cap a run with `Agent(..., max_tokens=20_000, max_cost_usd=0.10, max_seconds=60)`. When a budget is reached the run stops with `stopped="budget"` and `result.budget` says which one; set them for anything that runs unattended.
- When the model asks for several tools in one step, they run one after another. `Agent(..., parallel_tools=True)` runs them at the same time (results keep their order); only use it if your tools are safe to call from several threads.
- Long runs resend the whole conversation each step. `Agent(..., max_history_messages=30)` sends the task plus the latest messages instead, never separating a tool result from the request that produced it; `result.messages` keeps everything.
- It works with any llmgw provider that supports tools: Gemini, OpenAI and Ollama.

## Traces
Each run writes to `.agentloop/traces/<run_id>-<name>.jsonl`, with events `run_start`, `llm`, `tool` and `run_end`. Each event records tokens, latency, arguments, results and errors.

```
agentloop ls
agentloop view            # renders the latest trace to HTML and opens it
agentloop view .agentloop/traces/2026...jsonl -o trace.html --no-open
```

Pass `tracer=False` to turn tracing off, or `on_step=callback` to watch steps live.
