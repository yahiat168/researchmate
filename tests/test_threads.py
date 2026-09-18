"""Thread and cross-thread memory tests -- minimum test scenarios 1, 2, 3,
7, and 10."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import get_compiled_graph
from database.repositories import ProfileMemoryRepository, ThreadRepository
from tests.conftest import tool_call_message


def test_thread_keeps_context_across_turns_in_the_same_thread(scripted_llm):
    """Scenario 1: tell the assistant your name in Thread A, then ask for
    your name in the same thread -- the checkpointer must carry the full
    history into the second call."""
    threads = ThreadRepository()
    thread = threads.create_thread("alice")

    scripted_llm([AIMessage(content="Nice to meet you, Yahia."), AIMessage(content="Your name is Yahia.")])

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread.thread_id}}

    graph.invoke(
        {"messages": [HumanMessage(content="My name is Yahia.")], "user_id": "alice", "thread_id": thread.thread_id},
        config=config,
    )
    graph.invoke(
        {"messages": [HumanMessage(content="What's my name?")], "user_id": "alice", "thread_id": thread.thread_id},
        config=config,
    )

    state = graph.get_state(config)
    human_messages = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert human_messages == ["My name is Yahia.", "What's my name?"]


def test_preference_saved_in_one_thread_is_available_in_a_new_thread(scripted_llm):
    """Scenario 2: save a preference in Thread A, create Thread B, and
    confirm the preference is still available (long-term memory, not tied
    to any one thread)."""
    threads = ThreadRepository()
    profile = ProfileMemoryRepository()
    thread_a = threads.create_thread("alice")

    scripted_llm(
        [
            tool_call_message("save_memory", {"memory_type": "profile", "key_or_topic": "language", "value_or_content": "Arabic"}),
            AIMessage(content="Got it, I'll keep that in mind."),
        ]
    )

    graph = get_compiled_graph()
    graph.invoke(
        {
            "messages": [HumanMessage(content="Please remember I prefer Arabic.")],
            "user_id": "alice",
            "thread_id": thread_a.thread_id,
        },
        config={"configurable": {"thread_id": thread_a.thread_id}},
    )

    thread_b = threads.create_thread("alice")
    assert profile.get("alice", "language").value == "Arabic"

    state_b = graph.get_state({"configurable": {"thread_id": thread_b.thread_id}})
    # Thread B is brand new -- it has no messages of its own yet, even
    # though the *profile* memory carries over.
    assert not state_b.values.get("messages")


def test_second_user_cannot_see_first_users_threads_or_memory():
    """Scenario 3: create a second user and confirm the first user's
    memories are not available to them."""
    threads = ThreadRepository()
    profile = ProfileMemoryRepository()
    profile.upsert("alice", "name", "Alice")
    thread_a = threads.create_thread("alice")
    thread_b = threads.create_thread("bob")

    assert [t.thread_id for t in threads.list_threads("bob")] == [thread_b.thread_id]
    assert profile.get_all("bob") == []
    assert threads.get_thread(thread_a.thread_id, "bob") is None


def test_two_threads_do_not_mix_histories(scripted_llm):
    """Scenario 10: switch between two threads and confirm their histories
    are not mixed."""
    threads = ThreadRepository()
    thread_a = threads.create_thread("alice")
    thread_b = threads.create_thread("alice")

    scripted_llm([AIMessage(content="ok-a"), AIMessage(content="ok-b")])

    graph = get_compiled_graph()
    graph.invoke(
        {"messages": [HumanMessage(content="Hello from thread A")], "user_id": "alice", "thread_id": thread_a.thread_id},
        config={"configurable": {"thread_id": thread_a.thread_id}},
    )
    graph.invoke(
        {"messages": [HumanMessage(content="Hello from thread B")], "user_id": "alice", "thread_id": thread_b.thread_id},
        config={"configurable": {"thread_id": thread_b.thread_id}},
    )

    state_a = graph.get_state({"configurable": {"thread_id": thread_a.thread_id}})
    state_b = graph.get_state({"configurable": {"thread_id": thread_b.thread_id}})

    a_texts = [m.content for m in state_a.values["messages"] if isinstance(m, HumanMessage)]
    b_texts = [m.content for m in state_b.values["messages"] if isinstance(m, HumanMessage)]
    assert a_texts == ["Hello from thread A"]
    assert b_texts == ["Hello from thread B"]


def test_thread_history_survives_a_simulated_restart(scripted_llm):
    """Scenario 7: restart the application and reopen an existing thread --
    simulated here by dropping the cached checkpointer/graph and rebuilding
    fresh objects against the same sqlite file."""
    import agent.graph as graph_module

    threads = ThreadRepository()
    thread = threads.create_thread("alice")

    scripted_llm([AIMessage(content="Sure, that's saved.")])
    graph1 = get_compiled_graph()
    graph1.invoke(
        {"messages": [HumanMessage(content="Remember this thread.")], "user_id": "alice", "thread_id": thread.thread_id},
        config={"configurable": {"thread_id": thread.thread_id}},
    )

    # Simulate an app restart: drop cached singletons, rebuild against the
    # same underlying sqlite file (env vars are unchanged for this test).
    graph_module.get_checkpointer.cache_clear()
    graph_module.get_compiled_graph.cache_clear()

    graph2 = get_compiled_graph()
    state = graph2.get_state({"configurable": {"thread_id": thread.thread_id}})
    human_texts = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert human_texts == ["Remember this thread."]
