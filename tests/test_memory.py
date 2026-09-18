"""Long-term memory repository tests: profile memory, research memory, and
user isolation (spec section 2 & minimum test scenario 3)."""

from __future__ import annotations

from database.repositories import ProfileMemoryRepository, ResearchMemoryRepository, ThreadRepository


def test_profile_memory_upsert_and_get():
    repo = ProfileMemoryRepository()
    repo.upsert("alice", "name", "Alice")
    repo.upsert("alice", "name", "Alicia")  # same key -> overwrite, not duplicate
    facts = repo.get_all("alice")
    assert len(facts) == 1
    assert facts[0].value == "Alicia"


def test_profile_memory_delete_is_idempotent_and_reported():
    repo = ProfileMemoryRepository()
    repo.upsert("bob", "language", "English")
    assert repo.delete("bob", "language") is True
    assert repo.get_all("bob") == []
    assert repo.delete("bob", "language") is False  # already gone -> no crash, just False


def test_research_memory_add_and_keyword_search():
    repo = ResearchMemoryRepository()
    repo.add("alice", "rust vs go", "Rust has no GC; Go does.", "Blog", "https://example.com/a")
    repo.add("alice", "python packaging", "uv is fast.", None, None)

    hits = repo.search("alice", "rust")
    assert len(hits) == 1
    assert hits[0].topic == "rust vs go"
    assert hits[0].source_url == "https://example.com/a"


def test_research_memory_delete():
    repo = ResearchMemoryRepository()
    note = repo.add("alice", "topic", "content")
    assert repo.delete("alice", note.id) is True
    assert repo.get_all("alice") == []


def test_memory_is_isolated_between_users():
    """Minimum test scenario 3: create a second user and confirm the first
    user's memories are not available."""
    profile = ProfileMemoryRepository()
    research = ResearchMemoryRepository()
    profile.upsert("alice", "name", "Alice")
    research.add("alice", "topic", "alice's private finding")

    assert profile.get_all("bob") == []
    assert research.get_all("bob") == []
    assert profile.get_all("alice")[0].value == "Alice"


def test_a_user_cannot_delete_another_users_memory_by_guessing_ids():
    profile = ProfileMemoryRepository()
    profile.upsert("alice", "name", "Alice")
    # bob tries to delete alice's "name" key -- scoped delete must no-op.
    assert profile.delete("bob", "name") is False
    assert profile.get_all("alice")[0].value == "Alice"


def test_thread_repository_scopes_by_user():
    threads = ThreadRepository()
    t1 = threads.create_thread("alice", "Alice's thread")
    t2 = threads.create_thread("bob", "Bob's thread")

    assert threads.get_thread(t1.thread_id, "alice") is not None
    assert threads.get_thread(t1.thread_id, "bob") is None  # wrong user -> not found
    assert [t.thread_id for t in threads.list_threads("alice")] == [t1.thread_id]
    assert [t.thread_id for t in threads.list_threads("bob")] == [t2.thread_id]
