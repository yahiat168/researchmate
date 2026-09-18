"""ResearchMate -- Streamlit chat interface.

SQLite (via database/repositories.py and the LangGraph checkpointer) is the
single source of truth for threads and memory; this file only renders what
is already persisted and never keeps conversation state solely in
st.session_state (spec section 10: "Do not use st.session_state as the only
conversation database").
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import get_compiled_graph
from database.repositories import ProfileMemoryRepository, ResearchMemoryRepository, ThreadRepository
from database.setup import init_all

st.set_page_config(page_title="ResearchMate", page_icon="🔎", layout="wide")

# Safe to call every rerun -- only ever CREATE TABLE IF NOT EXISTS.
init_all()

thread_repo = ThreadRepository()
profile_repo = ProfileMemoryRepository()
research_repo = ResearchMemoryRepository()

# ---------------------------------------------------------------------------
# Session defaults (UI selection only -- not the data store itself)
# ---------------------------------------------------------------------------

if "user_id" not in st.session_state:
    st.session_state.user_id = "guest"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# ---------------------------------------------------------------------------
# Sidebar: user switch, thread create/switch, memory management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🔎 ResearchMate")

    user_id_input = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Switch this to act as a different user. Each user's threads "
        "and memory are completely separate.",
    )
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.session_state.thread_id = None
        st.rerun()

    user_id = st.session_state.user_id.strip()

    st.divider()
    st.subheader("Conversation threads")

    threads = thread_repo.list_threads(user_id) if user_id else []

    if st.button("+ New thread", use_container_width=True, disabled=not user_id):
        new_thread = thread_repo.create_thread(user_id)
        st.session_state.thread_id = new_thread.thread_id
        st.rerun()

    if threads:
        labels = {t.thread_id: f"{t.title}  ·  {t.thread_id[:8]}" for t in threads}
        ids = list(labels.keys())
        if st.session_state.thread_id not in labels:
            st.session_state.thread_id = ids[0]
        selected = st.radio(
            "Existing threads",
            options=ids,
            format_func=lambda tid: labels[tid],
            index=ids.index(st.session_state.thread_id),
            label_visibility="collapsed",
        )
        if selected != st.session_state.thread_id:
            st.session_state.thread_id = selected
            st.rerun()
    else:
        st.caption("No threads yet — create one to start chatting.")

    st.divider()
    with st.expander("🧠 Memory management", expanded=False):
        st.caption("Profile memory")
        profile_facts = profile_repo.get_all(user_id) if user_id else []
        if not profile_facts:
            st.caption("Nothing saved yet.")
        for f in profile_facts:
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{f.key}**: {f.value}")
            if c2.button("🗑", key=f"del_profile_{f.id}"):
                profile_repo.delete_by_id(user_id, f.id)
                st.rerun()

        st.caption("Research memory")
        research_notes = research_repo.get_all(user_id) if user_id else []
        if not research_notes:
            st.caption("Nothing saved yet.")
        for n in research_notes:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"**{n.topic}**: {n.content}")
                if n.source_url:
                    st.caption(f"[{n.source_title or n.source_url}]({n.source_url})")
            if c2.button("🗑", key=f"del_research_{n.id}"):
                research_repo.delete(user_id, n.id)
                st.rerun()

# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------

if not user_id or not st.session_state.thread_id:
    st.title("🔎 ResearchMate")
    st.info("Enter a user ID and create a thread in the sidebar to get started.")
    st.stop()

thread_id = st.session_state.thread_id
current_thread = thread_repo.get_thread(thread_id, user_id)
if current_thread is None:
    # Thread existed for a different user_id, or was deleted underneath us.
    st.session_state.thread_id = None
    st.rerun()

st.title("🔎 ResearchMate")
st.caption(f"Thread: **{current_thread.title}**  ·  ID: `{thread_id}`  ·  User: `{user_id}`")

try:
    graph = get_compiled_graph()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not initialize the agent graph: {type(exc).__name__}: {exc}")
    st.stop()

config = {"configurable": {"thread_id": thread_id}}

snapshot = graph.get_state(config)
state_values = snapshot.values if snapshot and snapshot.values else {}
history = state_values.get("messages", [])
all_tool_activity = state_values.get("tool_activity", [])
all_sources = state_values.get("sources", [])

for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)
    # ToolMessages and tool-call-only AIMessages aren't rendered as chat
    # bubbles; they live in the "Tool activity" panel below instead.

if all_tool_activity:
    with st.expander(f"🛠 Tool activity for this thread ({len(all_tool_activity)} call(s))"):
        for entry in reversed(all_tool_activity):
            icon = "⚠️" if entry.get("is_error") else "✅"
            st.markdown(f"{icon} **{entry['tool']}** — {entry['output_summary']}")

if all_sources:
    seen_urls: set[str] = set()
    deduped = []
    for s in reversed(all_sources):
        if s["url"] in seen_urls:
            continue
        seen_urls.add(s["url"])
        deduped.append(s)
    with st.expander(f"🔗 Sources found in this thread ({len(deduped)})"):
        for s in deduped:
            st.markdown(f"- [{s['title']}]({s['url']})")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_message = st.chat_input("Ask ResearchMate anything...")
if user_message:
    with st.chat_message("user"):
        st.write(user_message)

    prior_activity_len = len(all_tool_activity)
    prior_sources_len = len(all_sources)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_message)],
                        "user_id": user_id,
                        "thread_id": thread_id,
                    },
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - never crash the app
                st.error(
                    f"Something went wrong while processing your message: "
                    f"{type(exc).__name__}: {exc}"
                )
                st.stop()

        final_ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        answer = final_ai_messages[-1].content if final_ai_messages else "(no response generated)"
        st.write(answer)

        turn_activity = result.get("tool_activity", [])[prior_activity_len:]
        turn_sources = result.get("sources", [])[prior_sources_len:]

        if turn_activity:
            with st.expander(f"🛠 Tool activity this turn ({len(turn_activity)})", expanded=True):
                for entry in turn_activity:
                    icon = "⚠️" if entry.get("is_error") else "✅"
                    st.markdown(f"{icon} **{entry['tool']}** — {entry['output_summary']}")

        if turn_sources:
            st.markdown("**Sources:**")
            seen: set[str] = set()
            for s in turn_sources:
                if s["url"] in seen:
                    continue
                seen.add(s["url"])
                st.markdown(f"- [{s['title']}]({s['url']})")

    st.rerun()
