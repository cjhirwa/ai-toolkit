from agentloop import Agent, Tracer, read_trace, render, tool
from llmgw import LLM, FakeProvider, Response, ToolCall


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def fail() -> str:
    """Always fails."""
    raise RuntimeError("database down")


def test_agent_runs_tools_and_traces(tmp_path):
    script = [
        Response("", "m", tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3}), ToolCall("2", "fail", {})]),
        Response("", "m", tool_calls=[ToolCall("3", "nope", {})]),
        "The answer is 5.",
    ]
    provider = FakeProvider(script)
    agent = Agent(LLM("fake:m", provider=provider, log_dir=False), [add, fail], tracer=Tracer(tmp_path), name="calc")
    result = agent.run("What is 2+3?")
    assert result.output == "The answer is 5." and result.stopped == "final"
    execs = [e for s in result.steps for e in s.tool_calls]
    assert execs[0].result == "5"
    assert execs[1].error and "database down" in execs[1].result
    assert "unknown tool" in execs[2].result
    # the model saw tool results on the next turn
    assert [m.role for m in provider.requests[1].messages] == ["user", "assistant", "tool", "tool"]
    events = read_trace(result.trace_path)
    assert [e["kind"] for e in events].count("tool") == 3
    assert "database down" in render(result.trace_path)


def test_agent_stops_at_max_steps():
    loop = Response("", "m", tool_calls=[ToolCall("1", "add", {"a": 1, "b": 1})])
    provider = FakeProvider(lambda req: loop if req.tool_choice != "none" else "best effort")
    agent = Agent(LLM("fake:m", provider=provider, log_dir=False), [add], max_steps=3, tracer=False)
    result = agent.run("loop forever")
    assert result.stopped == "max_steps" and result.output == "best effort" and len(result.steps) == 3


def test_truncated_answer_is_not_reported_as_final(tmp_path):
    from agentloop import Agent
    from llmgw import LLM, FakeProvider, Response

    llm = LLM("fake:x", provider=FakeProvider([Response(text="The answer is", model="x", finish_reason="length")]), log_dir=False)
    result = Agent(llm=llm, tracer=False).run("question")
    assert result.stopped == "truncated"


def test_budgets_stop_a_runaway_agent():
    from agentloop import Agent
    from llmgw import LLM, FakeProvider, Response, ToolCall, Usage, tool

    @tool
    def lookup(q: str) -> str:
        """Look something up.

        Args:
            q: Query.
        """
        return "nothing useful"

    def always_tool(req):  # a model that never stops calling tools
        return Response(text="", model="x", tool_calls=[ToolCall("c", "lookup", {"q": "x"})],
                        usage=Usage(input_tokens=400, output_tokens=100, cost_usd=0.01))

    def agent(**budget):
        llm = LLM("fake:x", provider=FakeProvider(always_tool), log_dir=False)
        return Agent(llm=llm, tools=[lookup], tracer=False, max_steps=50, **budget).run("loop forever")

    r = agent(max_tokens=1200)
    assert r.stopped == "budget" and "token budget" in r.budget and len(r.steps) == 3
    r = agent(max_cost_usd=0.05)
    assert r.stopped == "budget" and "cost budget" in r.budget and len(r.steps) == 5
    r = agent(max_seconds=0)
    assert r.stopped == "budget" and "time budget" in r.budget and len(r.steps) == 1


def test_parallel_tools_run_together_and_keep_order():
    import time

    from agentloop import Agent
    from llmgw import LLM, FakeProvider, Response, ToolCall, tool

    @tool
    def slow_lookup(q: str) -> str:
        """Slow lookup.

        Args:
            q: Query.
        """
        time.sleep(0.3)
        return f"result for {q}"

    calls = [ToolCall(f"c{i}", "slow_lookup", {"q": f"q{i}"}) for i in range(3)]
    script = [Response(text="", model="x", tool_calls=calls), "done"]

    def run(parallel):
        llm = LLM("fake:x", provider=FakeProvider(list(script)), log_dir=False)
        start = time.perf_counter()
        r = Agent(llm=llm, tools=[slow_lookup], tracer=False, parallel_tools=parallel).run("look up three things")
        return time.perf_counter() - start, r

    serial_time, _ = run(False)
    parallel_time, result = run(True)
    assert serial_time > 0.85 and parallel_time < 0.6
    assert [ex.result for ex in result.steps[0].tool_calls] == ["result for q0", "result for q1", "result for q2"]


def test_history_window_keeps_task_and_never_orphans_tool_results():
    from agentloop import Agent
    from llmgw import LLM, FakeProvider, Response, ToolCall, tool

    @tool
    def lookup(q: str) -> str:
        """Look up.

        Args:
            q: Query.
        """
        return "found"

    sent = []

    def model(req):
        sent.append([m.role for m in req.messages])
        if len(sent) < 6:
            return Response(text="", model="x", tool_calls=[ToolCall(f"c{len(sent)}", "lookup", {"q": "x"}),
                                                            ToolCall(f"d{len(sent)}", "lookup", {"q": "y"})])
        return "final answer"

    llm = LLM("fake:x", provider=FakeProvider(model), log_dir=False)
    result = Agent(llm=llm, tools=[lookup], tracer=False, max_history_messages=5).run("the task")
    assert result.stopped == "final" and len(result.messages) > 15  # full history kept
    for roles in sent:
        assert roles[0] == "user" and len(roles) <= 5
        assert roles[1:2] != ["tool"]  # the window never starts with an orphaned tool result
        for i, role in enumerate(roles[1:], start=1):  # every tool result follows its request
            if role == "tool":
                assert roles[i - 1] in ("assistant", "tool")
    assert sent[-1][-3:] == ["assistant", "tool", "tool"]  # the latest step is complete


def test_history_window_keeps_an_oversized_latest_step_whole():
    from agentloop import Agent
    from llmgw import Message, ToolCall

    calls = [ToolCall(f"c{i}", "lookup", {}) for i in range(4)]
    msgs = [Message.user("task"), Message(role="assistant", content="", tool_calls=calls),
            *[Message.tool_result(c, "r") for c in calls]]
    window = Agent(tracer=False, max_history_messages=3, llm=object())._window(msgs)  # type: ignore[arg-type]
    assert [m.role for m in window] == ["user", "assistant", "tool", "tool", "tool", "tool"]
