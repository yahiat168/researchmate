"""Setup checker for ResearchMate.

Run this BEFORE `streamlit run app.py`:

    python check_setup.py

It checks your Python version, your installed packages, your .env file, and
then makes one real call to Tavily and one real call to your AI model
(including a tool-calling test, which is what the app actually depends on).
Every failure prints what to do about it in plain English.
"""

from __future__ import annotations

import os
import sys

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

problems: list[str] = []


def report(status: str, message: str, fix: str | None = None) -> None:
    print(f"{status} {message}")
    if fix:
        print(f"       -> {fix}")
    if status == FAIL and fix:
        problems.append(fix)


print("=" * 66)
print("ResearchMate setup check")
print("=" * 66)

# --- 1. Python version ---------------------------------------------------
if sys.version_info >= (3, 11):
    report(OK, f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
else:
    report(
        FAIL,
        f"Python {sys.version_info.major}.{sys.version_info.minor} is too old",
        "Install Python 3.11 or newer from https://python.org and re-create the virtual environment.",
    )

# --- 2. Packages ---------------------------------------------------------
missing = []
for package, import_name in [
    ("langgraph", "langgraph"),
    ("langgraph-checkpoint-sqlite", "langgraph.checkpoint.sqlite"),
    ("langchain-core", "langchain_core"),
    ("langchain-tavily", "langchain_tavily"),
    ("streamlit", "streamlit"),
    ("python-dotenv", "dotenv"),
]:
    try:
        __import__(import_name)
    except ImportError:
        missing.append(package)

if missing:
    report(
        FAIL,
        f"Missing packages: {', '.join(missing)}",
        "Run: pip install -r requirements.txt   (with your virtual environment activated)",
    )
else:
    report(OK, "Core packages installed")

# --- 3. .env file --------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

if os.path.exists(".env"):
    report(OK, ".env file found")
else:
    report(
        FAIL,
        ".env file not found",
        "Copy .env.example to .env, then open .env and paste your keys in.",
    )

# --- 4. Tavily -----------------------------------------------------------
tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
if not tavily_key or tavily_key.startswith("your-"):
    report(
        FAIL,
        "TAVILY_API_KEY is not set (or still the placeholder)",
        "Get a free key at https://app.tavily.com and put it in .env as TAVILY_API_KEY=...",
    )
else:
    try:
        from langchain_tavily import TavilySearch

        res = TavilySearch(max_results=2).invoke({"query": "what is python programming"})
        count = len(res.get("results", [])) if isinstance(res, dict) else 0
        if count:
            report(OK, f"Tavily web search works ({count} results returned)")
        else:
            report(WARN, "Tavily responded but returned no results (key is probably fine)")
    except Exception as exc:
        report(
            FAIL,
            f"Tavily call failed: {type(exc).__name__}: {exc}",
            "Check the key is correct and has credits left at https://app.tavily.com",
        )

# --- 5. The AI model -----------------------------------------------------
provider = os.environ.get("LLM_PROVIDER", "openai").lower()
key_names = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
key_name = key_names.get(provider, "OPENAI_API_KEY")
model_key = os.environ.get(key_name, "").strip()

print(f"       (provider = {provider}, model = {os.environ.get('LLM_MODEL', 'default')})")

if not model_key or model_key.startswith("your-"):
    report(
        FAIL,
        f"{key_name} is not set (or still the placeholder)",
        f"Put your key in .env as {key_name}=...  and make sure LLM_PROVIDER={provider} is correct.",
    )
else:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agent.graph import get_llm
        from agent.tools import ALL_TOOLS

        llm = get_llm()
        reply = llm.invoke("Reply with exactly the word: ready")
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        report(OK, f"AI model responded: {text.strip()[:40]!r}")

        # The app depends on tool calling, so test that specifically.
        bound = llm.bind_tools(ALL_TOOLS)
        tool_reply = bound.invoke(
            "Search the web for today's top technology news. You must use a tool."
        )
        if getattr(tool_reply, "tool_calls", None):
            names = ", ".join(tc["name"] for tc in tool_reply.tool_calls)
            report(OK, f"Tool calling works (model requested: {names})")
        else:
            report(
                WARN,
                "Model replied without requesting a tool",
                "This can be normal once, but if search never triggers in the app, "
                "try a stronger model via LLM_MODEL in .env (e.g. gemini-2.5-flash).",
            )
    except Exception as exc:
        msg = str(exc)
        fix = f"Check {key_name} is valid and LLM_PROVIDER={provider} matches the key you have."
        if "not found" in msg.lower() or "404" in msg or "model" in msg.lower():
            fix = (
                "The model name may not be available on your key. Try setting "
                "LLM_MODEL in .env to gemini-2.5-flash or gemini-1.5-flash."
            )
        report(FAIL, f"AI model call failed: {type(exc).__name__}: {msg[:200]}", fix)

# --- Summary -------------------------------------------------------------
print("=" * 66)
if problems:
    print(f"{len(problems)} problem(s) to fix:\n")
    for i, fix in enumerate(problems, 1):
        print(f"  {i}. {fix}")
    print("\nFix those, then run this again: python check_setup.py")
    sys.exit(1)
else:
    print("Everything works. Start the app with:  streamlit run app.py")
    sys.exit(0)
