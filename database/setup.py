"""Database bootstrap for ResearchMate.

Two SQLite files are used on purpose (see schema.sql for why):

- ``data/memory.sqlite``      -> long-term memory (users, threads, profile
                                  memory, research memory), owned by
                                  ``repositories.py``.
- ``data/checkpoints.sqlite`` -> short-term thread memory, owned entirely by
                                  LangGraph's ``SqliteSaver`` checkpointer.

Both paths are configurable via environment variables so tests can point
them at temporary files instead of the real data/ directory.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

DEFAULT_MEMORY_DB_PATH = str(DATA_DIR / "memory.sqlite")
DEFAULT_CHECKPOINT_DB_PATH = str(DATA_DIR / "checkpoints.sqlite")


def get_memory_db_path() -> str:
    """Path to the long-term memory database (overridable for tests)."""
    return os.environ.get("RESEARCHMATE_MEMORY_DB", DEFAULT_MEMORY_DB_PATH)


def get_checkpoint_db_path() -> str:
    """Path to the LangGraph checkpointer database (overridable for tests)."""
    return os.environ.get("RESEARCHMATE_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB_PATH)


def _ensure_parent_dir(path: str) -> None:
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with sane defaults (FK enforcement, Row
    factory) to the long-term memory database.
    """
    path = db_path or get_memory_db_path()
    _ensure_parent_dir(path)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection_scope(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager that commits on success and rolls back on error."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create the long-term memory tables if they do not already exist.

    Safe to call on every app startup -- it only ever CREATEs IF NOT EXISTS.
    """
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connection_scope(db_path) as conn:
        conn.executescript(schema_sql)


def init_all() -> None:
    """Initialize both storage layers used by the app.

    The checkpointer's own tables are created lazily by LangGraph the first
    time ``SqliteSaver.setup()`` runs (see agent/graph.py), but we make sure
    the data/ directory exists here so both files can be created cleanly on
    a first run / after a restart.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    _ensure_parent_dir(get_checkpoint_db_path())


if __name__ == "__main__":
    init_all()
    print(f"Initialized memory DB at {get_memory_db_path()}")
    print(f"Checkpoint DB will be created at {get_checkpoint_db_path()}")
