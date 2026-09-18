"""Node-level memory helpers used by agent/graph.py.

These are distinct from the memory *tools* in tools.py: the tools are
things the model chooses to call mid-conversation, while these two
functions are unconditional graph nodes that always run once per turn --
"load memory" at the start (read-only) and "save memory" at the end
(bookkeeping / thread housekeeping). Actual writes triggered by the model
happen through the save_memory / delete_memory tools, which already commit
straight to SQLite; the save_memory node's job is to make sure the thread
record stays in sync and to summarize what was saved this turn for the UI.
"""

from __future__ import annotations

from typing import Any

from database.repositories import ProfileMemoryRepository, ResearchMemoryRepository, ThreadRepository

_profile_repo = ProfileMemoryRepository()
_research_repo = ResearchMemoryRepository()
_thread_repo = ThreadRepository()

MAX_RESEARCH_NOTES_IN_CONTEXT = 8


def load_user_context(user_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Read a user's profile memory and recent research memory from SQLite.

    Returns (profile_context, research_context) ready to drop into the
    system prompt and into graph state.
    """
    profile_facts = _profile_repo.get_all(user_id)
    profile_context = {f.key: f.value for f in profile_facts}

    research_notes = _research_repo.get_all(user_id)[:MAX_RESEARCH_NOTES_IN_CONTEXT]
    research_context = [
        {
            "id": n.id,
            "topic": n.topic,
            "content": n.content,
            "source_title": n.source_title,
            "source_url": n.source_url,
        }
        for n in research_notes
    ]
    return profile_context, research_context


def format_profile_context(profile_context: dict[str, str]) -> str:
    if not profile_context:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in profile_context.items())


def format_research_context(research_context: list[dict[str, Any]]) -> str:
    if not research_context:
        return ""
    lines = []
    for note in research_context:
        src = f" (source: {note['source_url']})" if note.get("source_url") else ""
        lines.append(f"- [{note['id']}] {note['topic']}: {note['content']}{src}")
    return "\n".join(lines)


def finalize_thread(user_id: str, thread_id: str, first_user_message: str | None) -> None:
    """Housekeeping run once per turn by the 'save memory' graph node:
    bump the thread's updated_at, and give a brand-new thread a real title
    derived from the first user message instead of the "New conversation"
    placeholder.
    """
    thread = _thread_repo.get_thread(thread_id, user_id)
    if thread is not None and thread.title == "New conversation" and first_user_message:
        title = first_user_message.strip().splitlines()[0][:60]
        _thread_repo.rename(thread_id, user_id, title or "New conversation")
    else:
        _thread_repo.touch(thread_id, user_id)
