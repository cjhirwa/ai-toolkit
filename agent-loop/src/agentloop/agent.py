from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from llmgw import LLM, Message, Tool, ToolCall, Usage, as_tool

from .tracing import NullTracer, Tracer

_MAX_RESULT_CHARS = 12000


@dataclass
class ToolExecution:
    call: ToolCall
    result: str
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class Step:
    index: int
    text: str
    tool_calls: list[ToolExecution] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    output: str
    steps: list[Step]
    messages: list[Message]
    usage: Usage
    stopped: str  # "final" | "max_steps" | "truncated" (cut off at max_tokens) | "budget"
    trace_path: str | None = None
    budget: str | None = None  # which budget stopped the run, when stopped == "budget"


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        s = result
    else:
        try:
            s = json.dumps(result, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o)))
        except TypeError:
            s = str(result)
    if len(s) > _MAX_RESULT_CHARS:
        s = s[:_MAX_RESULT_CHARS] + f"\n...[truncated {len(s) - _MAX_RESULT_CHARS} chars]"
    return s


class Agent:
    """The whole loop, nothing hidden:

        while steps remain:
            response = llm(messages, tools)
            if no tool calls: return response.text
            for each call: run the tool, append its result (or error) as a tool message

    Tool exceptions are returned to the model as errors so it can recover.

    Besides ``max_steps``, a run can be capped by ``max_tokens`` (input + output, all
    steps), ``max_cost_usd`` and ``max_seconds``. When a cap is reached the run stops with
    ``stopped="budget"`` and the reason in ``AgentResult.budget``, instead of continuing.
    ``parallel_tools=True`` runs the tool calls of one step concurrently (results keep their order).
    ``max_history_messages`` keeps long runs from resending everything: the model sees the task
    plus the latest messages; ``AgentResult.messages`` still holds the full history.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        tools: Sequence[Tool | Callable] = (),
        *,
        system: str | None = None,
        max_steps: int = 10,
        name: str = "agent",
        tracer: Tracer | bool = True,
        on_step: Callable[[Step], None] | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        max_seconds: float | None = None,
        parallel_tools: bool = False,
        max_history_messages: int | None = None,
    ):
        self.llm = llm or LLM()
        self.tools: dict[str, Tool] = {t.name: t for t in map(as_tool, tools)}
        self.system = system
        self.max_steps = max_steps
        self.name = name
        self.tracer = tracer if isinstance(tracer, Tracer) else (Tracer() if tracer else NullTracer())
        self.on_step = on_step
        self.max_tokens, self.max_cost_usd, self.max_seconds = max_tokens, max_cost_usd, max_seconds
        # Run the tool calls of one step at the same time. Off by default: tools may share
        # state that is not safe to use from several threads.
        self.parallel_tools = parallel_tools
        self.max_history_messages = max_history_messages

    def _window(self, messages: list[Message]) -> list[Message]:
        """What the model sees: everything, or with ``max_history_messages`` the original task
        plus the most recent messages. A tool result is never sent without the assistant
        message that requested it: if the cut falls among tool results it moves forward to the
        next step, or, when those results belong to the latest step, back to keep that step
        whole (exceeding the limit only then)."""
        limit = self.max_history_messages
        if limit is None or len(messages) <= limit:
            return messages
        head = messages[:1]  # the task
        start = len(messages) - (limit - 1)
        forward = start
        while forward < len(messages) and messages[forward].role == "tool":
            forward += 1
        if forward < len(messages):
            start = forward
        else:
            while start > 1 and messages[start].role == "tool":
                start -= 1
        return head + messages[start:]

    def _over_budget(self, usage: Usage, started: float) -> str | None:
        if self.max_tokens is not None and usage.input_tokens + usage.output_tokens >= self.max_tokens:
            return f"token budget of {self.max_tokens} reached ({usage.input_tokens + usage.output_tokens})"
        if self.max_cost_usd is not None and (usage.cost_usd or 0.0) >= self.max_cost_usd:
            return f"cost budget of ${self.max_cost_usd} reached (${usage.cost_usd:.4f})"
        if self.max_seconds is not None and time.monotonic() - started >= self.max_seconds:
            return f"time budget of {self.max_seconds}s reached"
        return None

    def _execute(self, call: ToolCall) -> ToolExecution:
        t0 = time.perf_counter()
        tool = self.tools.get(call.name)
        if tool is None:
            res = ToolExecution(call, f"Error: unknown tool {call.name!r}. Available: {sorted(self.tools)}", "unknown tool")
        else:
            try:
                res = ToolExecution(call, _stringify(tool.run(call.arguments)))
            except Exception as e:  # noqa: BLE001
                res = ToolExecution(call, f"Error: {type(e).__name__}: {e}", f"{type(e).__name__}: {e}")
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res

    def run(self, task: str | list[Message]) -> AgentResult:
        messages = [Message.user(task)] if isinstance(task, str) else list(task)
        tools = list(self.tools.values())
        self.tracer.start(self.name, task=task if isinstance(task, str) else "[messages]",
                          model=self.llm.model, tools=list(self.tools), system=self.system)
        steps: list[Step] = []
        usage = Usage()
        started = time.monotonic()

        for i in range(self.max_steps):
            resp = self.llm.complete(self._window(messages), system=self.system, tools=tools or None)
            usage = usage + resp.usage
            messages.append(resp.to_message())
            step = Step(i, resp.text, usage=resp.usage, latency_ms=resp.latency_ms)
            self.tracer.event(
                "llm", step=i, text=resp.text, usage=resp.usage.to_dict(), latency_ms=resp.latency_ms,
                tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls],
            )
            if self.parallel_tools and len(resp.tool_calls) > 1:
                with ThreadPoolExecutor(max_workers=len(resp.tool_calls)) as pool:
                    executions = list(pool.map(self._execute, resp.tool_calls))  # results stay in call order
            else:
                executions = [self._execute(call) for call in resp.tool_calls]
            for call, ex in zip(resp.tool_calls, executions):
                step.tool_calls.append(ex)
                messages.append(Message.tool_result(call, ex.result))
                self.tracer.event("tool", step=i, name=call.name, arguments=call.arguments,
                                  result=ex.result, error=ex.error, latency_ms=ex.latency_ms)
            steps.append(step)
            if self.on_step:
                self.on_step(step)
            if resp.tool_calls and (reason := self._over_budget(usage, started)):
                self.tracer.event("run_end", stopped="budget", budget=reason, output=resp.text, usage=usage.to_dict(), steps=len(steps))
                return AgentResult(resp.text, steps, messages, usage, "budget", str(self.tracer.path or "") or None, budget=reason)
            if not resp.tool_calls:
                # A reply cut off at max_tokens is not a finished answer; say so instead of "final".
                stopped = "truncated" if resp.finish_reason == "length" else "final"
                self.tracer.event("run_end", stopped=stopped, output=resp.text, usage=usage.to_dict(), steps=len(steps))
                return AgentResult(resp.text, steps, messages, usage, stopped, str(self.tracer.path or "") or None)

        # Out of steps: ask for a best-effort answer with tools disabled.
        messages.append(Message.user("You have reached the step limit. Give your best final answer now from what you found."))
        resp = self.llm.complete(self._window(messages), system=self.system, tools=tools or None, tool_choice="none" if tools else None)
        usage = usage + resp.usage
        messages.append(resp.to_message())
        self.tracer.event("run_end", stopped="max_steps", output=resp.text, usage=usage.to_dict(), steps=len(steps))
        return AgentResult(resp.text, steps, messages, usage, "max_steps", str(self.tracer.path or "") or None)
