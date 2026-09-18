"""Tool-level tests: each tool's happy path and its error handling.

Tools are called via their `.func` (the raw function `@tool` wraps) so we
can pass the InjectedState/InjectedToolCallId arguments explicitly without
needing a running graph -- this keeps these tests fast and focused on one
tool at a time. Graph-level wiring (routing, InjectedState actually being
filled from state) is covered in test_graph.py.
"""

from __future__ import annotations

from langgraph.types import Command

import agent.tools as tools_module


class FakeTavilyOk:
    def __init__(self, results):
        self._results = results

    def invoke(self, payload):
        return {"results": self._results}


class FakeTavilyBroken:
    def invoke(self, payload):
        raise RuntimeError("network exploded")


def test_tavily_search_missing_api_key_returns_error_not_exception(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = tools_module.tavily_search.func(query="capital of France", tool_call_id="tc1")
    assert isinstance(result, Command)
    activity = result.update["tool_activity"][0]
    assert activity["is_error"] is True
    assert "TAVILY_API_KEY" in activity["output_summary"]


def test_tavily_search_success_returns_sources(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        tools_module,
        "_get_tavily_tool",
        lambda include_domains=None: FakeTavilyOk([{"title": "Example", "url": "https://example.com", "content": "Some snippet."}]),
    )

    result = tools_module.tavily_search.func(query="example query", tool_call_id="tc2")
    assert result.update["sources"][0]["url"] == "https://example.com"
    assert result.update["tool_activity"][0]["is_error"] is False


def test_tavily_search_restricts_to_requested_domain(monkeypatch):
    """Spec section 5: if the user asks for a specific website, that domain
    is passed through as an allowed domain."""
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    captured = {}

    def fake_factory(include_domains=None):
        captured["include_domains"] = include_domains
        return FakeTavilyOk([{"title": "Wiki", "url": "https://en.wikipedia.org/x", "content": "..."}])

    monkeypatch.setattr(tools_module, "_get_tavily_tool", fake_factory)

    result = tools_module.tavily_search.func(
        query="langgraph", tool_call_id="tc14", include_domains=["wikipedia.org"]
    )
    assert captured["include_domains"] == ["wikipedia.org"]
    assert result.update["tool_activity"][0]["input"]["include_domains"] == ["wikipedia.org"]


def test_tavily_search_without_domains_passes_none(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    captured = {}

    def fake_factory(include_domains=None):
        captured["include_domains"] = include_domains
        return FakeTavilyOk([{"title": "T", "url": "https://example.com", "content": "..."}])

    monkeypatch.setattr(tools_module, "_get_tavily_tool", fake_factory)

    tools_module.tavily_search.func(query="general question", tool_call_id="tc15")
    assert captured["include_domains"] is None


def test_tavily_search_empty_results_is_handled_gracefully(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(tools_module, "_get_tavily_tool", lambda include_domains=None: FakeTavilyOk([]))

    result = tools_module.tavily_search.func(query="nonexistent xyz", tool_call_id="tc3")
    assert result.update["search_results"] == []
    assert "No web results" in result.update["tool_activity"][0]["output_summary"]


def test_tavily_search_tool_failure_returns_message_not_exception(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(tools_module, "_get_tavily_tool", lambda include_domains=None: FakeTavilyBroken())

    result = tools_module.tavily_search.func(query="anything", tool_call_id="tc4")
    assert result.update["tool_activity"][0]["is_error"] is True
    assert "network exploded" in result.update["tool_activity"][0]["output_summary"]


def test_save_search_delete_memory_roundtrip():
    save_result = tools_module.save_memory.func(
        memory_type="profile",
        key_or_topic="name",
        value_or_content="Yahia",
        user_id="u1",
        tool_call_id="tc5",
    )
    assert save_result.update["tool_activity"][0]["is_error"] is False

    search_result = tools_module.search_memory.func(query="name", user_id="u1", tool_call_id="tc6")
    assert "Yahia" in search_result.update["messages"][0].content

    delete_result = tools_module.delete_memory.func(
        memory_type="profile", key_or_id="name", user_id="u1", tool_call_id="tc7"
    )
    assert delete_result.update["tool_activity"][0]["is_error"] is False

    second_delete = tools_module.delete_memory.func(
        memory_type="profile", key_or_id="name", user_id="u1", tool_call_id="tc8"
    )
    assert second_delete.update["tool_activity"][0]["is_error"] is True  # nothing left to delete


def test_save_memory_refuses_credentials():
    result = tools_module.save_memory.func(
        memory_type="profile",
        key_or_topic="api_key",
        value_or_content="my api key is sk-12345",
        user_id="u1",
        tool_call_id="tc9",
    )
    assert result.update["tool_activity"][0]["is_error"] is True
    assert "Refused" in result.update["messages"][0].content

    # Make sure it really wasn't stored.
    from database.repositories import ProfileMemoryRepository

    assert ProfileMemoryRepository().get("u1", "api_key") is None


def test_search_memory_is_scoped_to_the_calling_user():
    tools_module.save_memory.func(
        memory_type="research",
        key_or_topic="alice's secret topic",
        value_or_content="alice's private finding",
        user_id="alice",
        tool_call_id="tc10",
    )
    result = tools_module.search_memory.func(query="secret", user_id="bob", tool_call_id="tc11")
    assert "No saved memory matched" in result.update["messages"][0].content


def test_get_thread_summary_with_no_messages():
    result = tools_module.get_thread_summary.func(thread_id="th1", messages=[], tool_call_id="tc12")
    assert "no messages yet" in result.update["messages"][0].content.lower()


def test_get_thread_summary_with_messages():
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = [HumanMessage(content="Hi"), AIMessage(content="Hello!")]
    result = tools_module.get_thread_summary.func(thread_id="th2", messages=msgs, tool_call_id="tc13")
    content = result.update["messages"][0].content
    assert "th2" in content
    assert "Hi" in content
