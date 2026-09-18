"""LangGraph state definition for ResearchMate.

Per spec section 3, four memory categories exist. Here is where each one
actually lives at runtime:

- Short-term thread memory  -> ``messages`` (this TypedDict), persisted
                                automatically every step by the SqliteSaver
                                checkpointer keyed on thread_id.
- User profile memory       -> NOT stored in this state long-term. It is
                                loaded from SQLite into ``profile_context``
                                at the start of a run (load_memory node) and
                                written back through the ``save_memory``
                                tool / node when something new is learned.
- Research memory           -> same pattern, via ``research_context`` and
                                ``sources``.
- Temporary working state   -> ``current_query``, ``search_results``,
                                ``tool_activity`` -- scratch space for a
                                single graph run, reset each new turn.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ToolActivity(TypedDict):
    tool: str
    input: dict
    output_summary: str
    is_error: bool


class Source(TypedDict):
    title: str
    url: str
    snippet: str


class GraphState(TypedDict):
    # Identity -- who this run belongs to. Set once per invoke, never
    # mutated by a node, and never supplied to the LLM as an editable field.
    user_id: str
    thread_id: str

    # Short-term thread memory (checkpointed automatically).
    messages: Annotated[list[BaseMessage], add_messages]

    # Long-term memory, loaded read-only at the start of each run.
    profile_context: dict[str, str]
    research_context: list[dict[str, Any]]

    # Temporary working state for the current run only (overwritten each
    # turn by load_memory -- no accumulation reducer).
    current_query: str
    search_results: list[dict[str, Any]]

    # Full, append-only audit trail for the whole thread: every tool call
    # made across every turn, and every source ever surfaced. The
    # checkpointer persists this, so "restart the app and reopen a thread"
    # keeps the full history. The Streamlit layer records how long these
    # lists were *before* a turn's invoke() and diffs against their length
    # *after* to show "what happened this turn" without needing a reset.
    tool_activity: Annotated[list[ToolActivity], operator.add]
    sources: Annotated[list[Source], operator.add]

    # Loop guard so a confused model can't call tools forever. Plain field
    # (no reducer) -- overwritten fresh by load_memory every turn.
    tool_call_rounds: int
