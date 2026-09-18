"""Shared pytest fixtures.

Every test runs against fresh, temporary SQLite files instead of the real
data/ directory, and never needs a real TAVILY_API_KEY / OPENAI_API_KEY --
the LLM is replaced with a deterministic ScriptedLLM and Tavily calls are
monkeypatched at the point they'd hit the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_databases(tmp_path, monkeypatch):
    """Point every test at fresh, temporary SQLite files and reset the
    lru_cache'd checkpointer/graph singletons so tests never leak state
    into one another."""
    memory_db = tmp_path / "memory.sqlite"
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("RESEARCHMATE_MEMORY_DB", str(memory_db))
    monkeypatch.setenv("RESEARCHMATE_CHECKPOINT_DB", str(checkpoint_db))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    from database.setup import init_all

    init_all()

    import agent.graph as graph_module

    graph_module.get_checkpointer.cache_clear()
    graph_module.get_compiled_graph.cache_clear()

    yield

    graph_module.get_checkpointer.cache_clear()
    graph_module.get_compiled_graph.cache_clear()


class ScriptedLLM:
    """Deterministic stand-in for a chat model: give it a queue of
    AIMessage objects and each .invoke() call returns the next one, so
    graph tests are fast, offline, and reproducible."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invocations: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self._responses:
            from langchain_core.messages import AIMessage

            return AIMessage(content="(scripted LLM ran out of responses)")
        return self._responses.pop(0)


@pytest.fixture
def scripted_llm(monkeypatch):
    """Factory fixture: call with a list of AIMessages to install a
    ScriptedLLM as agent.graph.get_llm's return value for this test."""

    def _install(responses):
        llm = ScriptedLLM(responses)
        import agent.graph as graph_module

        monkeypatch.setattr(graph_module, "get_llm", lambda: llm)
        return llm

    return _install


def tool_call_message(tool_name: str, args: dict, call_id: str = "call_1"):
    """Build an AIMessage that requests a single tool call, for use with
    ScriptedLLM in graph-level tests."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )
