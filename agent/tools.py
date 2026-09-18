"""Tools available to the ResearchMate agent.

Design notes:

- ``user_id`` / ``thread_id`` / ``messages`` are injected from graph state
  via ``InjectedState`` rather than being arguments the model fills in.
  This is what keeps one user's data from leaking to another: the LLM
  physically cannot pass a different user_id, because the field is stripped
  from the tool schema it sees and filled in by the graph at call time.
- Every tool returns a ``Command`` that updates ``tool_activity`` (for the
  Streamlit "tool activity" panel) and appends the required ``ToolMessage``
  back into ``messages``. ``tavily_search`` additionally updates
  ``search_results`` / ``sources``.
- Every tool wraps its risky work (network calls, sqlite calls, bad input)
  in try/except and returns a clear error message instead of raising, per
  the "tool errors must return useful messages instead of crashing the
  application" rule in the spec.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from database.repositories import ProfileMemoryRepository, ResearchMemoryRepository

_profile_repo = ProfileMemoryRepository()
_research_repo = ResearchMemoryRepository()

MAX_SEARCH_RESULTS = 5
_SENSITIVE_MARKERS = ("password", "api key", "apikey", "secret key", "credit card", "ssn", "social security")


def _get_tavily_tool(include_domains: Optional[list[str]] = None):
    """Build the underlying Tavily search tool lazily so importing this
    module never fails just because TAVILY_API_KEY isn't set yet -- unit
    tests can monkeypatch this function to avoid real network calls.

    ``include_domains`` restricts the search to specific sites, used when
    the user asks about a particular website (spec section 5).
    """
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=MAX_SEARCH_RESULTS,
        topic="general",
        search_depth="basic",
        include_raw_content=False,
        include_domains=include_domains or [],
    )


def _activity(tool_name: str, tool_input: dict, summary: str, is_error: bool) -> dict:
    return {"tool": tool_name, "input": tool_input, "output_summary": summary, "is_error": is_error}


@tool
def tavily_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    include_domains: Optional[list[str]] = None,
) -> Command:
    """Search the public web for current information using Tavily.

    Use this when the user asks about something recent, time-sensitive, or
    outside general knowledge -- news, prices, schedules, current facts, or
    anything that benefits from a live, citable source.

    Args:
        query: A focused search query capturing what to look up.
        include_domains: Optional list of bare domains to restrict the
            search to, e.g. ["wikipedia.org", "python.org"]. Set this only
            when the user names a specific website they want searched;
            leave it out for a general web search.
    """
    tool_input = {"query": query}
    if include_domains:
        tool_input["include_domains"] = include_domains

    if not os.environ.get("TAVILY_API_KEY"):
        message = (
            "Web search is unavailable: TAVILY_API_KEY is not set. Answer "
            "from existing knowledge and tell the user this could not be "
            "verified with a live search."
        )
        return Command(
            update={
                "tool_activity": [_activity("tavily_search", tool_input, message, True)],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    try:
        raw = _get_tavily_tool(include_domains).invoke({"query": query})
    except Exception as exc:  # noqa: BLE001 - tools must never crash the app
        message = f"Tavily search failed ({type(exc).__name__}): {exc}"
        return Command(
            update={
                "tool_activity": [_activity("tavily_search", tool_input, message, True)],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    if not results:
        scope = f" on {', '.join(include_domains)}" if include_domains else ""
        message = f"No web results found for '{query}'{scope}."
        return Command(
            update={
                "tool_activity": [_activity("tavily_search", tool_input, message, False)],
                "search_results": [],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    sources, lines = [], []
    for r in results[:MAX_SEARCH_RESULTS]:
        title = r.get("title") or r.get("url") or "Untitled source"
        url = r.get("url", "")
        snippet = (r.get("content") or "")[:400]
        sources.append({"title": title, "url": url, "snippet": snippet})
        lines.append(f"- {title} ({url}): {snippet}")

    summary_text = f"Found {len(sources)} result(s) for '{query}':\n" + "\n".join(lines)
    scope = f" (restricted to {', '.join(include_domains)})" if include_domains else ""
    return Command(
        update={
            "tool_activity": [
                _activity(
                    "tavily_search", tool_input, f"{len(sources)} result(s) for '{query}'{scope}", False
                )
            ],
            "search_results": sources,
            "sources": sources,
            "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
        }
    )


@tool
def search_memory(
    query: str,
    user_id: Annotated[str, InjectedState("user_id")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search this user's saved long-term memory (profile facts and past
    research notes) for anything relevant to the query.

    Args:
        query: What to look for in the user's saved memory.
    """
    try:
        notes = _research_repo.search(user_id, query)
        needle = query.strip().lower()
        profile_hits = [
            f for f in _profile_repo.get_all(user_id)
            if needle in f.key.lower() or needle in f.value.lower()
        ]
    except Exception as exc:  # noqa: BLE001
        message = f"Memory search failed ({type(exc).__name__}): {exc}"
        return Command(
            update={
                "tool_activity": [_activity("search_memory", {"query": query}, message, True)],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    lines = [f"- profile: {f.key} = {f.value}" for f in profile_hits]
    for n in notes:
        src = f" (source: {n.source_url})" if n.source_url else ""
        lines.append(f"- research [{n.id}] {n.topic}: {n.content}{src}")

    message = (
        f"No saved memory matched '{query}'."
        if not lines
        else f"Found {len(lines)} saved item(s) matching '{query}':\n" + "\n".join(lines)
    )
    return Command(
        update={
            "tool_activity": [_activity("search_memory", {"query": query}, message, False)],
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        }
    )


@tool
def save_memory(
    memory_type: Literal["profile", "research"],
    key_or_topic: str,
    value_or_content: str,
    user_id: Annotated[str, InjectedState("user_id")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    source_title: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Command:
    """Save a durable fact to this user's long-term memory.

    Use memory_type="profile" for things like the user's name, preferred
    language, or answer style (key_or_topic is a short key such as "name").
    Use memory_type="research" for a finding worth keeping across threads
    (key_or_topic is the topic, value_or_content is what to remember,
    source_title/source_url are optional citation info).

    Never call this to store passwords, API keys, or other credentials --
    refuse and explain instead.
    """
    tool_input = {"memory_type": memory_type, "key_or_topic": key_or_topic}
    if any(marker in value_or_content.lower() for marker in _SENSITIVE_MARKERS):
        message = "Refused: that looks like a credential or sensitive secret, which ResearchMate never stores in memory."
        return Command(
            update={
                "tool_activity": [_activity("save_memory", tool_input, message, True)],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    try:
        if memory_type == "profile":
            _profile_repo.upsert(user_id, key_or_topic.strip(), value_or_content.strip())
            message = f"Saved profile memory: {key_or_topic} = {value_or_content}"
        else:
            _research_repo.add(
                user_id, key_or_topic.strip(), value_or_content.strip(), source_title, source_url
            )
            message = f"Saved research memory under topic '{key_or_topic}'."
    except Exception as exc:  # noqa: BLE001
        message = f"Saving memory failed ({type(exc).__name__}): {exc}"
        return Command(
            update={
                "tool_activity": [_activity("save_memory", tool_input, message, True)],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    return Command(
        update={
            "tool_activity": [_activity("save_memory", tool_input, message, False)],
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        }
    )


@tool
def delete_memory(
    memory_type: Literal["profile", "research"],
    key_or_id: str,
    user_id: Annotated[str, InjectedState("user_id")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Delete a previously saved memory when the user asks to forget it.

    For memory_type="profile", key_or_id is the profile key (e.g. "name").
    For memory_type="research", key_or_id is the numeric research memory id
    (call search_memory first if you don't already know it).
    """
    tool_input = {"memory_type": memory_type, "key_or_id": key_or_id}
    try:
        if memory_type == "profile":
            deleted = _profile_repo.delete(user_id, key_or_id.strip())
        else:
            deleted = _research_repo.delete(user_id, int(key_or_id))
    except Exception as exc:  # noqa: BLE001
        message = f"Delete failed ({type(exc).__name__}): {exc}"
        return Command(
            update={
                "tool_activity": [_activity("delete_memory", tool_input, message, True)],
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )

    message = (
        f"Deleted {memory_type} memory '{key_or_id}'."
        if deleted
        else f"No {memory_type} memory found for '{key_or_id}' -- nothing to delete."
    )
    return Command(
        update={
            "tool_activity": [_activity("delete_memory", tool_input, message, not deleted)],
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        }
    )


@tool
def get_thread_summary(
    thread_id: Annotated[str, InjectedState("thread_id")],
    messages: Annotated[list, InjectedState("messages")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Summarize what has been discussed in the current conversation thread
    so far -- a quick recap, without needing another full model turn."""
    turns = [m for m in messages if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    if not turns:
        message = "This thread has no messages yet."
    else:
        recap_lines = []
        for m in turns[-6:]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            text = m.content if isinstance(m.content, str) else str(m.content)
            recap_lines.append(f"{role}: {text[:200]}")
        message = f"Thread {thread_id} recap ({len(turns)} message(s) total):\n" + "\n".join(recap_lines)

    return Command(
        update={
            "tool_activity": [_activity("get_thread_summary", {"thread_id": thread_id}, message, False)],
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        }
    )


ALL_TOOLS = [tavily_search, search_memory, save_memory, delete_memory, get_thread_summary]
