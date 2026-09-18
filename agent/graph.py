"""LangGraph workflow for ResearchMate.

Flow (matches spec section 4's recommended graph flow):

    START -> load_memory -> call_model -> [tool call?] -> tool_node -> call_model (loop)
                                        -> save_memory -> final_response -> END

``call_model`` decides, through normal LLM tool-calling, whether a tool is
needed. The conditional edge after it inspects the latest AIMessage: if it
requested tool calls *and* the loop guard (MAX_TOOL_ROUNDS) hasn't tripped,
we go run the tool(s) and loop back to call_model; otherwise we move on to
save_memory -> final_response -> END.
"""

from __future__ import annotations

import os
import sqlite3
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.memory import (
    finalize_thread,
    format_profile_context,
    format_research_context,
    load_user_context,
)
from agent.prompts import render_system_prompt
from agent.state import GraphState
from agent.tools import ALL_TOOLS
from database.setup import get_checkpoint_db_path

# Caps how many times the model may loop back into tool calling within a
# single turn -- required by spec section 10 ("Limit the number of search
# results and graph loops").
MAX_TOOL_ROUNDS = 3


def get_llm():
    """Build the chat model from environment configuration.

    LLM_PROVIDER selects "openai" (default) or "anthropic" so the same
    graph runs with whichever API key is available; LLM_MODEL overrides the
    default model name for either provider.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-latest")
        return ChatAnthropic(model=model, temperature=0)

    from langchain_openai import ChatOpenAI

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _load_memory_node(state: GraphState) -> dict:
    """Read relevant user memories and reset the per-run scratch fields.

    tool_activity/sources are append-only across the whole thread (see
    agent/state.py) so they are intentionally NOT cleared here -- the UI
    layer diffs their length before/after a turn instead.
    """
    profile_context, research_context = load_user_context(state["user_id"])
    return {
        "profile_context": profile_context,
        "research_context": research_context,
        "tool_call_rounds": 0,
    }


def _call_model_node(state: GraphState) -> dict:
    """Understand the request and decide whether a tool is needed."""
    llm = get_llm().bind_tools(ALL_TOOLS)
    system_text = render_system_prompt(
        format_profile_context(state.get("profile_context", {})),
        format_research_context(state.get("research_context", [])),
    )
    messages = [SystemMessage(content=system_text), *state["messages"]]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - an LLM error must not crash the app
        response = AIMessage(
            content=(
                "I ran into an error talking to the language model "
                f"({type(exc).__name__}: {exc}). Please check that the API "
                "key is set correctly and try again."
            )
        )
    return {"messages": [response]}


def _route_after_model(state: GraphState) -> str:
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if has_tool_calls and state.get("tool_call_rounds", 0) < MAX_TOOL_ROUNDS:
        return "tools"
    return "save_memory"


def _count_tool_round(state: GraphState) -> dict:
    return {"tool_call_rounds": state.get("tool_call_rounds", 0) + 1}


def _save_memory_node(state: GraphState) -> dict:
    """Store only useful facts or preferences.

    Explicit saves already happened inside the save_memory / delete_memory
    tools (they write straight to SQLite so the model gets immediate
    confirmation). This node's job is the per-turn bookkeeping that has to
    happen exactly once regardless of how many tool rounds occurred: keep
    the thread record's title/updated_at in sync.
    """
    first_user_message = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)), None
    )
    finalize_thread(state["user_id"], state["thread_id"], first_user_message)
    return {}


def _final_response_node(state: GraphState) -> dict:
    """Return a clear answer with sources and tool information.

    The final AIMessage, collected sources, and tool_activity log are
    already sitting in state; this node is a dedicated seam (matching the
    spec's explicit "Final response" responsibility) for any future
    response post-processing without touching the rest of the graph.
    """
    return {}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(checkpointer):
    graph = StateGraph(GraphState)

    graph.add_node("load_memory", _load_memory_node)
    graph.add_node("call_model", _call_model_node)
    graph.add_node("count_tool_round", _count_tool_round)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("save_memory", _save_memory_node)
    graph.add_node("final_response", _final_response_node)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "call_model")
    graph.add_conditional_edges(
        "call_model",
        _route_after_model,
        {"tools": "count_tool_round", "save_memory": "save_memory"},
    )
    graph.add_edge("count_tool_round", "tools")
    graph.add_edge("tools", "call_model")
    graph.add_edge("save_memory", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """Persistent SQLite checkpointer (spec section 10: use a persistent
    checkpointer for the final version, not the in-memory one)."""
    conn = sqlite3.connect(get_checkpoint_db_path(), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph(get_checkpointer())
