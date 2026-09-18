"""Repository layer for ResearchMate's long-term memory.

Every method takes an explicit ``user_id`` and every query filters on it.
This is what guarantees "one user's memories and conversations separate
from another user's data" (spec section 2) -- there is no code path in this
file that can read or write a row without a user_id, and the agent layer
only ever passes the *current authenticated* user_id in (see
agent/tools.py, which injects it from graph state rather than letting the
model supply it).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from database.setup import connection_scope


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Dataclasses (plain, serializable rows -- keeps the agent layer decoupled
# from sqlite3.Row objects).
# ---------------------------------------------------------------------------


@dataclass
class Thread:
    thread_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class ProfileFact:
    id: int
    user_id: str
    key: str
    value: str
    updated_at: str


@dataclass
class ResearchNote:
    id: int
    user_id: str
    topic: str
    content: str
    source_title: Optional[str]
    source_url: Optional[str]
    created_at: str


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserRepository:
    def ensure_user(self, user_id: str) -> None:
        """Create the user row if it doesn't exist yet. Idempotent."""
        with connection_scope() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
                (user_id, _now()),
            )

    def list_users(self) -> list[str]:
        with connection_scope() as conn:
            rows = conn.execute("SELECT user_id FROM users ORDER BY user_id").fetchall()
        return [r["user_id"] for r in rows]


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


class ThreadRepository:
    def __init__(self) -> None:
        self._users = UserRepository()

    def create_thread(self, user_id: str, title: str = "New conversation") -> Thread:
        self._users.ensure_user(user_id)
        thread_id = new_id("thread_")
        now = _now()
        with connection_scope() as conn:
            conn.execute(
                """INSERT INTO threads (thread_id, user_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (thread_id, user_id, title, now, now),
            )
        return Thread(thread_id, user_id, title, now, now)

    def get_thread(self, thread_id: str, user_id: str) -> Optional[Thread]:
        """Fetch a thread, scoped to user_id so one user can never load
        another user's thread by guessing/reusing a thread_id."""
        with connection_scope() as conn:
            row = conn.execute(
                "SELECT * FROM threads WHERE thread_id = ? AND user_id = ?",
                (thread_id, user_id),
            ).fetchone()
        return self._row_to_thread(row) if row else None

    def list_threads(self, user_id: str) -> list[Thread]:
        with connection_scope() as conn:
            rows = conn.execute(
                "SELECT * FROM threads WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_thread(r) for r in rows]

    def touch(self, thread_id: str, user_id: str) -> None:
        with connection_scope() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ? AND user_id = ?",
                (_now(), thread_id, user_id),
            )

    def rename(self, thread_id: str, user_id: str, title: str) -> None:
        with connection_scope() as conn:
            conn.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ? AND user_id = ?",
                (title, _now(), thread_id, user_id),
            )

    @staticmethod
    def _row_to_thread(row: sqlite3.Row) -> Thread:
        return Thread(
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# ---------------------------------------------------------------------------
# Profile memory
# ---------------------------------------------------------------------------


class ProfileMemoryRepository:
    def __init__(self) -> None:
        self._users = UserRepository()

    def upsert(self, user_id: str, key: str, value: str) -> None:
        self._users.ensure_user(user_id)
        with connection_scope() as conn:
            conn.execute(
                """INSERT INTO profile_memory (user_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, key)
                   DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (user_id, key, value, _now()),
            )

    def get_all(self, user_id: str) -> list[ProfileFact]:
        with connection_scope() as conn:
            rows = conn.execute(
                "SELECT * FROM profile_memory WHERE user_id = ? ORDER BY key",
                (user_id,),
            ).fetchall()
        return [
            ProfileFact(r["id"], r["user_id"], r["key"], r["value"], r["updated_at"])
            for r in rows
        ]

    def get(self, user_id: str, key: str) -> Optional[ProfileFact]:
        with connection_scope() as conn:
            row = conn.execute(
                "SELECT * FROM profile_memory WHERE user_id = ? AND key = ?",
                (user_id, key),
            ).fetchone()
        if not row:
            return None
        return ProfileFact(row["id"], row["user_id"], row["key"], row["value"], row["updated_at"])

    def delete(self, user_id: str, key: str) -> bool:
        with connection_scope() as conn:
            cur = conn.execute(
                "DELETE FROM profile_memory WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
        return cur.rowcount > 0

    def delete_by_id(self, user_id: str, fact_id: int) -> bool:
        with connection_scope() as conn:
            cur = conn.execute(
                "DELETE FROM profile_memory WHERE user_id = ? AND id = ?",
                (user_id, fact_id),
            )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Research memory
# ---------------------------------------------------------------------------


class ResearchMemoryRepository:
    def __init__(self) -> None:
        self._users = UserRepository()

    def add(
        self,
        user_id: str,
        topic: str,
        content: str,
        source_title: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> ResearchNote:
        self._users.ensure_user(user_id)
        now = _now()
        with connection_scope() as conn:
            cur = conn.execute(
                """INSERT INTO research_memory
                   (user_id, topic, content, source_title, source_url, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, topic, content, source_title, source_url, now),
            )
            note_id = cur.lastrowid
        return ResearchNote(note_id, user_id, topic, content, source_title, source_url, now)

    def get_all(self, user_id: str) -> list[ResearchNote]:
        with connection_scope() as conn:
            rows = conn.execute(
                "SELECT * FROM research_memory WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def search(self, user_id: str, query: str, limit: int = 5) -> list[ResearchNote]:
        """Simple, dependency-free keyword search (LIKE match on topic and
        content). Good enough for a beginner-scope capstone; swappable for
        embeddings/FTS later without touching the agent layer."""
        like = f"%{query.strip()}%"
        with connection_scope() as conn:
            rows = conn.execute(
                """SELECT * FROM research_memory
                   WHERE user_id = ? AND (topic LIKE ? OR content LIKE ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, like, like, limit),
            ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def delete(self, user_id: str, note_id: int) -> bool:
        with connection_scope() as conn:
            cur = conn.execute(
                "DELETE FROM research_memory WHERE user_id = ? AND id = ?",
                (user_id, note_id),
            )
        return cur.rowcount > 0

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> ResearchNote:
        return ResearchNote(
            id=row["id"],
            user_id=row["user_id"],
            topic=row["topic"],
            content=row["content"],
            source_title=row["source_title"],
            source_url=row["source_url"],
            created_at=row["created_at"],
        )
