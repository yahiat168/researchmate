# ResearchMate

A personal web research assistant built with **LangGraph**, **Tavily**, **Streamlit**, and **SQLite**. It answers questions, searches the live web when needed, remembers useful details about you across conversations, and keeps every conversation in its own thread — with one user's data never visible to another.

This is the Beginner Capstone project: *Build a Web Research Assistant with Tools and Memory*.

## 1. What it does

- Understands your message and decides whether it needs a tool (web search, memory lookup, etc.) or can answer directly.
- Searches the web with Tavily for current information and cites its sources.
- Remembers useful facts (name, language, preferences) and research findings across *every* thread for a given user.
- Keeps each conversation thread's own message history separate via a persistent checkpointer.
- Shows you exactly which tools ran and which sources were found, in an expandable panel.
- Lets you forget a saved memory on request.
- Keeps two different users' data fully isolated from each other.

## 2. Architecture

```
                        ┌───────────────────────────────┐
                        │      Streamlit UI (app.py)     │
                        │  chat · threads · memory panel │
                        └───────────────┬────────────────┘
                                         │ graph.invoke(messages, user_id, thread_id)
                                         ▼
   START → load_memory → call_model ──► (tool call?) ──► tools → call_model (loop, max 3 rounds)
                                         │
                                         └──► save_memory → final_response → END
```

| Node | Responsibility |
|---|---|
| `load_memory` | Reads the user's profile + research memory from SQLite into context. |
| `call_model` | Sends the conversation + memory context to the LLM; the LLM decides whether to answer directly or call a tool. |
| `tools` | Executes `tavily_search`, `search_memory`, `save_memory`, `delete_memory`, or `get_thread_summary`. |
| `save_memory` (node) | Per-turn bookkeeping: keeps the thread's title/timestamp in sync. (Actual memory writes happen immediately inside the `save_memory`/`delete_memory` tools.) |
| `final_response` | Dedicated seam that returns the answer, sources, and tool log to the UI. |

### Memory design

| Memory type | What it stores | Scope | Implementation |
|---|---|---|---|
| Short-term thread memory | Messages, tool calls, in-progress research | One thread | `SqliteSaver` checkpointer → `data/checkpoints.sqlite` |
| User profile memory | Name, language, answer style, preferences | All threads, one user | `profile_memory` table via `ProfileMemoryRepository` |
| Research memory | Saved topics, findings, source links | All threads, one user | `research_memory` table via `ResearchMemoryRepository` |
| Temporary working state | Current query, search results | One graph run | Plain fields on the LangGraph state |

Two separate SQLite files are used on purpose: `data/checkpoints.sqlite` holds *only* what the LangGraph checkpointer manages (short-term thread state), and `data/memory.sqlite` holds *only* the long-term, cross-thread tables (`users`, `threads`, `profile_memory`, `research_memory`). This makes the short-term/long-term split physical, not just conceptual, and means user isolation can be checked with one glance at `memory.sqlite`'s `WHERE user_id = ?` filters — see `database/repositories.py`.

**User isolation, concretely:** `user_id` is never a field the LLM fills in. It's injected into every tool from the graph's own state (`InjectedState`), so a tool call can only ever touch the current session's `user_id` — there is no code path in `agent/tools.py` or `database/repositories.py` that reads/writes memory without a `user_id` filter.

## 3. Tools

| Tool | Purpose |
|---|---|
| `tavily_search` | Find current information on the public web. |
| `search_memory` | Find relevant long-term memories for the current user. |
| `save_memory` | Save a user preference, fact, or research note. |
| `delete_memory` | Remove a memory when asked to forget it. |
| `get_thread_summary` | Summarize the current conversation thread. |

Every tool catches its own errors (bad/missing API key, network failure, empty results, bad input) and returns a clear message instead of raising — the app never crashes because a tool failed.

## 4. Project structure

```
researchmate/
├── app.py                  # Streamlit UI
├── agent/
│   ├── graph.py             # LangGraph workflow (nodes, routing, checkpointer)
│   ├── state.py              # Graph state schema
│   ├── tools.py               # tavily_search, search_memory, save_memory, delete_memory, get_thread_summary
│   ├── memory.py                # load/save helpers used by the graph nodes
│   └── prompts.py                # system prompt
├── database/
│   ├── setup.py             # DB bootstrap, connection helpers
│   ├── repositories.py       # UserRepository, ThreadRepository, ProfileMemoryRepository, ResearchMemoryRepository
│   └── schema.sql              # long-term memory schema
├── tests/
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_threads.py
│   └── test_graph.py
├── data/                    # SQLite files are written here at runtime (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

## 5. Setup

**Requirements:** Python 3.11+, a [Tavily](https://app.tavily.com) API key (free tier available), and an API key for one AI model provider — Google Gemini, OpenAI, or Anthropic.

```bash
cd researchmate
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env: set TAVILY_API_KEY, and pick ONE model provider block
```

`.env.example` has a ready-made block for each provider — uncomment the one you have a key for. The default is Gemini (`LLM_PROVIDER=google`, `GOOGLE_API_KEY=...`). If you were given a custom OpenAI-compatible endpoint, use `LLM_PROVIDER=openai` and set `OPENAI_BASE_URL` to it.

**Before running the app, verify your setup:**

```bash
python check_setup.py
```

This checks your Python version, packages, and `.env`, then makes one real call to Tavily and one real tool-calling test against your model. Every failure it finds prints exactly what to do about it.

## 6. Running the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Enter a user ID in the sidebar, click **+ New thread**, and start chatting. SQLite files are created automatically in `data/` on first run.

## 7. Running the tests

```bash
pytest -q
```

The tests use a temporary SQLite database per test (via `RESEARCHMATE_MEMORY_DB` / `RESEARCHMATE_CHECKPOINT_DB` env overrides) and a fake Tavily client, so they never require real API keys or touch your real `data/` files. They cover the minimum test scenarios from the spec: name recall within a thread, preference persistence across threads, user isolation, Tavily called vs. not called, remember-then-forget, restart persistence, invalid-key/empty-results/tool-failure handling, follow-up questions, and thread isolation.

## 8. Known limitations

- `search_memory` uses simple SQL `LIKE` keyword matching, not semantic/embedding search — good enough for the scope of this capstone, but it won't catch paraphrases.
- The LLM choice is configurable (`LLM_PROVIDER`) but only OpenAI- and Anthropic-compatible chat models have been wired up.
- Tool-call loops are capped at 3 rounds per turn to bound cost and latency; a genuinely multi-step research task may need more and would currently get a best-effort partial answer.
- Thread titles are auto-generated from the first message; there's no manual rename UI yet (the repository method exists — `ThreadRepository.rename` — it's just not exposed in the sidebar).
- No authentication: "user ID" is a free-text field, not a login. Isolation between users is enforced at the data layer, not by verifying who's typing.
