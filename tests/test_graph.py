"""Graph-level tests: routing (tool vs. no tool), the tool-call loop guard,
and a follow-up question reusing an earlier search (minimum test scenarios
4, 5, and 9)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import MAX_TOOL_ROUNDS, get_compiled_graph
from database.repositories import ThreadRepository
from tests.conftest import tool_call_message


def _new_thread(user_id: str = "alice"):
    return ThreadRepository().create_thread(user_id)


def test_simple_question_does_not_call_tavily(scripted_llm, monkeypatch):
    """Minimum test scenario 5: a question that doesn't need web search
    must not trigger Tavily."""
    import agent.tools as tools_module

    def fail_if_called(include_domains=None):
        raise AssertionError("tavily_search should not have been invoked for a simple question")

    monkeypatch.setattr(tools_module, "_get_tavily_tool", fail_if_called)
    scripted_llm([AIMessage(content="2 + 2 is 4.")])

    thread = _new_thread()
    graph = get_compiled_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2 + 2?")], "user_id": "alice", "thread_id": thread.thread_id},
        config={"configurable": {"thread_id": thread.thread_id}},
    )

    assert result.get("tool_activity", []) == []
    assert result["messages"][-1].content == "2 + 2 is 4."


def test_current_event_question_calls_tavily_and_returns_sources(scripted_llm, monkeypatch):
    """Minimum test scenario 4: a current question triggers Tavily and the
    answer comes with sources."""
    import agent.tools as tools_module

    class FakeTavily:
        def invoke(self, payload):
            return {"results": [{"title": "Today's news", "url": "https://news.example.com", "content": "Something happened."}]}

    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(tools_module, "_get_tavily_tool", lambda include_domains=None: FakeTavily())

    scripted_llm(
        [
            tool_call_message("tavily_search", {"query": "today's news"}),
            AIMessage(content="Here's what I found in the news."),
        ]
    )

    thread = _new_thread()
    graph = get_compiled_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="What's in the news today?")],
            "user_id": "alice",
            "thread_id": thread.thread_id,
        },
        config={"configurable": {"thread_id": thread.thread_id}},
    )

    assert len(result["tool_activity"]) == 1
    assert result["tool_activity"][0]["tool"] == "tavily_search"
    assert result["sources"][0]["url"] == "https://news.example.com"
    assert result["messages"][-1].content == "Here's what I found in the news."


def test_followup_question_reuses_previous_search_without_a_new_call(scripted_llm, monkeypatch):
    """Minimum test scenario 9: a follow-up question that depends on the
    previous search result should be answerable from conversation history
    alone."""
    import agent.tools as tools_module

    class FakeTavily:
        def invoke(self, payload):
            return {"results": [{"title": "Mars weather", "url": "https://example.com/mars", "content": "-60C average."}]}

    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(tools_module, "_get_tavily_tool", lambda include_domains=None: FakeTavily())

    scripted_llm(
        [
            tool_call_message("tavily_search", {"query": "average temperature on Mars"}),
            AIMessage(content="Mars averages about -60C."),
            AIMessage(content="That's about -76F."),
        ]
    )

    thread = _new_thread()
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread.thread_id}}

    graph.invoke(
        {
            "messages": [HumanMessage(content="What's the average temperature on Mars?")],
            "user_id": "alice",
            "thread_id": thread.thread_id,
        },
        config=config,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is that in Fahrenheit?")], "user_id": "alice", "thread_id": thread.thread_id},
        config=config,
    )

    assert result["messages"][-1].content == "That's about -76F."
    # Only the first turn called a tool; the follow-up was answered from
    # the message history already carried by the checkpointer.
    assert len(result["tool_activity"]) == 1


def test_tool_call_loop_is_capped(scripted_llm):
    """Spec section 10: 'Limit the number of ... graph loops.' A model that
    keeps requesting tools must be stopped after MAX_TOOL_ROUNDS instead of
    looping forever."""
    responses = [
        tool_call_message("get_thread_summary", {}, call_id=f"call_{i}") for i in range(MAX_TOOL_ROUNDS + 1)
    ]
    scripted_llm(responses)

    thread = _new_thread()
    graph = get_compiled_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="loop please")], "user_id": "alice", "thread_id": thread.thread_id},
        config={"configurable": {"thread_id": thread.thread_id}},
    )

    assert len(result["tool_activity"]) == MAX_TOOL_ROUNDS
    assert isinstance(result["messages"][-1], AIMessage)


def test_llm_error_produces_a_message_instead_of_crashing(scripted_llm, monkeypatch):
    """Minimum test scenario 8 (tool/LLM failure must not crash the app)."""
    import agent.graph as graph_module

    class ExplodingLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("provider is down")

    monkeypatch.setattr(graph_module, "get_llm", lambda: ExplodingLLM())

    thread = _new_thread()
    graph = get_compiled_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="hello")], "user_id": "alice", "thread_id": thread.thread_id},
        config={"configurable": {"thread_id": thread.thread_id}},
    )

    assert "provider is down" in result["messages"][-1].content
