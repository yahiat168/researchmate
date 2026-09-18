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

# --- 4b. Custom endpoint probe (LiteLLM / other OpenAI-compatible proxies)
def probe_endpoint(base_url: str, api_key: str):
    """Try GET {base}/models on the given URL and on its /v1 variant.

    Returns (working_base_url, [model names]) or (None, []). Proxies such as
    LiteLLM expose their own model aliases, and the right base URL may or
    may not need a /v1 suffix -- so we find out rather than guessing.
    """
    import json
    import urllib.error
    import urllib.request

    candidates = [base_url.rstrip("/")]
    if not base_url.rstrip("/").endswith("/v1"):
        candidates.append(base_url.rstrip("/") + "/v1")

    for candidate in candidates:
        try:
            req = urllib.request.Request(
                f"{candidate}/models", headers={"Authorization": f"Bearer {api_key}"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in payload.get("data", []) if m.get("id")]
            return candidate, models
        except Exception:
            continue
    return None, []


# --- 4c. Pick a working chat model automatically --------------------------
# Gateways list everything they host -- video, image, audio, embedding and
# chat models together -- and many entries are retired upstream even though
# they're still advertised. Rather than making the user guess one at a
# time, filter to plausible chat models and actually try them.

NON_CHAT_MARKERS = (
    "veo", "imagen", "lyria", "embedding", "transcribe", "tts", "audio",
    "robotics", "image-generation", "live", "vision",
)


def rank_chat_candidates(models: list[str], required_prefix: str | None) -> list[str]:
    """Best-guess ordering of which listed models can actually chat."""
    candidates = []
    for name in models:
        if name.endswith("/*") or "*" in name:
            continue  # permission wildcard, not a real model
        if required_prefix and not name.startswith(required_prefix):
            continue
        short = name.split("/")[-1].lower()
        if any(marker in short for marker in NON_CHAT_MARKERS):
            continue
        # Prefer flash/pro chat models; prefer stable over preview; prefer
        # higher version numbers (newer models are likelier to still exist).
        score = 0
        if "flash" in short or "pro" in short:
            score -= 10
        if "preview" in short or "exp" in short:
            score += 5
        if "lite" in short:
            score += 2
        version = re.search(r"(\d+)\.(\d+)", short)
        if version:
            score -= int(version.group(1)) * 2 + int(version.group(2)) * 0.1
        candidates.append((score, name))
    candidates.sort(key=lambda pair: pair[0])
    return [name for _, name in candidates]


def try_models(candidates: list[str], limit: int = 6):
    """Call each candidate until one answers. Returns (name, supports_tools)."""
    from agent.graph import get_llm
    from agent.tools import ALL_TOOLS

    original = os.environ.get("LLM_MODEL")
    try:
        for name in candidates[:limit]:
            os.environ["LLM_MODEL"] = name
            try:
                llm = get_llm()
                llm.invoke("Reply with exactly the word: ready")
            except Exception as exc:
                reason = "not allowed" if "403" in str(exc) else "unavailable"
                print(f"         tried {name} -- {reason}")
                continue
            # It answers. Now check the thing the app actually needs.
            try:
                bound = llm.bind_tools(ALL_TOOLS)
                reply = bound.invoke("Search the web for today's news. You must use a tool.")
                return name, bool(getattr(reply, "tool_calls", None))
            except Exception:
                return name, False
        return None, False
    finally:
        if original is not None:
            os.environ["LLM_MODEL"] = original


import re  # noqa: E402  (used by rank_chat_candidates)

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
    discovered_models: list[str] = []
    required_prefix: str | None = None

    # If a custom endpoint is configured, confirm it's reachable and show
    # which model names it actually offers before we try to use one.
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if provider == "openai" and base_url:
        working, models = probe_endpoint(base_url, model_key)
        if working is None:
            report(
                FAIL,
                f"Could not reach the endpoint at {base_url}",
                "Check the URL is exactly right and your key is valid for it. "
                "If it needs a /v1 on the end, add it to OPENAI_BASE_URL in .env.",
            )
        else:
            if working != base_url.rstrip("/"):
                report(
                    WARN,
                    f"Endpoint works, but at {working} (not the URL in your .env)",
                    f"Change OPENAI_BASE_URL in .env to: {working}",
                )
            else:
                report(OK, f"Endpoint reachable at {working}")
            if models:
                discovered_models = models
                current = os.environ.get("LLM_MODEL", "").strip()
                # A "family/*" entry is a permission wildcard: it tells us
                # every usable model must carry that prefix.
                for entry in models:
                    if entry.endswith("/*"):
                        required_prefix = entry[:-1]
                        break
                print(f"       Endpoint offers {len(models)} model name(s)"
                      + (f"; your team is limited to {required_prefix}*" if required_prefix else ""))
                if current and current not in models:
                    report(
                        WARN,
                        f"LLM_MODEL is '{current}', which is not in the endpoint's list",
                        "Not always fatal (gateways list aliases inconsistently), "
                        "but if the call below fails this is why.",
                    )

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
        import re

        msg = str(exc)
        lowered = msg.lower()
        fix = f"Check {key_name} is valid and LLM_PROVIDER={provider} matches the key you have."

        # Gateways (LiteLLM and friends) often restrict a team to a model
        # pattern like ['gemini/*'] and report the allowed list in the error.
        # If it's there, turn it into the exact model name to use.
        allowed = re.search(r"can only access models=\[([^\]]+)\]", msg)
        if allowed:
            patterns = [pat.strip().strip("'\"") for pat in allowed.group(1).split(",")]
            current = os.environ.get("LLM_MODEL", "").strip()
            suggestion = patterns[0]
            if suggestion.endswith("/*"):
                prefix = suggestion[:-1]  # "gemini/*" -> "gemini/"
                base = current.split("/")[-1] or "gemini-2.0-flash"
                suggestion = f"{prefix}{base}"
            fix = (
                f"Your key is valid, but this gateway only allows {patterns}. "
                f"Set LLM_MODEL in .env to: {suggestion}"
            )
        elif "not found" in lowered or "404" in msg:
            fix = (
                "The model name may not be available on your key. Try another, "
                "e.g. gemini/gemini-2.5-flash or gemini/gemini-1.5-flash."
            )
        elif "401" in msg or "unauthor" in lowered or "api key not valid" in lowered:
            fix = (
                f"The gateway rejected {key_name}. Check the key was pasted whole, "
                f"with no quotes or trailing spaces, and that LLM_PROVIDER={provider} "
                "and OPENAI_BASE_URL point at the right service."
            )

        print(f"{FAIL} AI model call with '{os.environ.get('LLM_MODEL', '?')}' failed: "
              f"{type(exc).__name__}: {msg[:160]}")

        # The model name is the usual culprit, and we may already know every
        # name this endpoint offers -- so try them instead of asking the user
        # to guess again.
        auto_fixed = False
        if discovered_models and ("401" not in msg and "unauthor" not in lowered):
            candidates = rank_chat_candidates(discovered_models, required_prefix)
            if candidates:
                print(f"       Auto-testing up to 6 of {len(candidates)} chat models...")
                found, supports_tools = try_models(candidates)
                if found:
                    auto_fixed = True
                    report(OK, f"Found a working model: {found}")
                    report(
                        OK if supports_tools else WARN,
                        "Tool calling works with it" if supports_tools
                        else "It answers, but did not request a tool on the test prompt",
                    )
                    print()
                    print("       >>> Set this line in your .env file: <<<")
                    print(f"       LLM_MODEL={found}")
                    print()
                    problems.append(f"Set LLM_MODEL={found} in .env, then run this again.")
                else:
                    report(
                        FAIL,
                        "None of the chat models tried would respond",
                        "Ask the Sprints team which model name your key is meant to use.",
                    )
        if not auto_fixed and not discovered_models:
            problems.append(fix)
            print(f"       -> {fix}")

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
