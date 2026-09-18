-- ResearchMate long-term memory schema.
--
-- IMPORTANT: This database (data/memory.sqlite) stores ONLY long-term,
-- cross-thread data: users, thread metadata, profile memory, and research
-- memory. Short-term thread memory (messages, tool calls, in-progress
-- research) lives separately in the LangGraph checkpointer database
-- (data/checkpoints.sqlite), managed automatically by
-- langgraph.checkpoint.sqlite.SqliteSaver. Keeping these in two files makes
-- the "short-term vs long-term" split in the spec physically explicit,
-- not just logical.

PRAGMA foreign_keys = ON;

-- One row per known user. Created lazily the first time a user_id is seen.
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- One row per conversation thread. thread_id doubles as the LangGraph
-- checkpointer's thread_id, so the two storage layers stay in sync.
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id);

-- Profile memory: small key/value facts about a user (name, language,
-- preferred answer style, etc.) that apply across every thread.
CREATE TABLE IF NOT EXISTS profile_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_profile_user ON profile_memory(user_id);

-- Research memory: saved topics / findings / source links that should be
-- available to a user in any future thread.
CREATE TABLE IF NOT EXISTS research_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    topic       TEXT NOT NULL,
    content     TEXT NOT NULL,
    source_title TEXT,
    source_url  TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_research_user ON research_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_research_topic ON research_memory(user_id, topic);
