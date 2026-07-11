"""Thin wrapper around Groq's free-tier API for Qwen3.

reasoning_effort="none" disables Qwen3's <think> block entirely: the app only
needs structured JSON output, so paying for a hidden reasoning trace on every
call just burns tokens against the free-tier rate limit for no benefit here.
"""

import json
import re
import signal
import time

from groq import Groq, RateLimitError

from app import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


HARD_CALL_TIMEOUT_SECONDS = 40

# Groq's free tier is 30 RPM / 6K TPM. Every 429 we hit has been followed, most
# of the time, by a multi-minute stall on the *next* call regardless of retry
# strategy (fresh client, hard timeout -- see chat()) -- something in the
# client/connection state degrades after a rate-limit hit. Proactively pacing
# calls comfortably under the limit avoids triggering 429s in the first place,
# which sidesteps the stall entirely rather than trying to recover from it.
PROACTIVE_THROTTLE_SECONDS = 8.0


class _HardTimeout(Exception):
    pass


def _raise_hard_timeout(signum, frame):
    raise _HardTimeout()


def get_client() -> Groq:
    # max_retries=0: we handle retries ourselves so failures surface immediately
    # instead of compounding with the SDK's own internal retry/backoff.
    return Groq(api_key=config.GROQ_API_KEY, timeout=30.0, max_retries=0)


def _call_with_hard_timeout(client: Groq, **kwargs):
    """The Groq/httpx client's own timeout has intermittently failed to fire after
    a 429 (observed: process hangs with ~0% CPU for 10+ minutes past the configured
    30s timeout). signal.alarm forcibly interrupts the call at the OS level as a
    backstop, since it doesn't depend on the HTTP client's own timeout machinery.
    signal.alarm only works on the main thread -- if called from a worker thread
    (e.g. LangGraph's node executor), fall back to relying on the client's own
    timeout instead of crashing."""
    try:
        signal.signal(signal.SIGALRM, _raise_hard_timeout)
        signal.alarm(HARD_CALL_TIMEOUT_SECONDS)
    except ValueError:
        return client.chat.completions.create(**kwargs)
    try:
        return client.chat.completions.create(**kwargs)
    finally:
        signal.alarm(0)


def chat(client: Groq, messages: list[dict], max_tokens: int = 800, temperature: float = 0.2) -> str:
    time.sleep(PROACTIVE_THROTTLE_SECONDS)
    completion = None
    for attempt in range(3):
        try:
            completion = _call_with_hard_timeout(
                client,
                model=config.LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort="none",
            )
            break
        except (RateLimitError, _HardTimeout) as exc:
            wait_seconds = 15 * (attempt + 1)
            reason = "rate limited" if isinstance(exc, RateLimitError) else "hard timeout hit"
            print(f"  {reason}, waiting {wait_seconds}s (attempt {attempt + 1}/3)")
            time.sleep(wait_seconds)
            # Fresh client on retry: a stale connection in the pool after a 429
            # has intermittently caused the next request to hang indefinitely
            # rather than respect the client's own timeout.
            client = get_client()
    if completion is None:
        raise RuntimeError("Groq call failed after 3 attempts (rate limit or hard timeout)")

    text = completion.choices[0].message.content or ""
    return _THINK_RE.sub("", text).strip()


def chat_json(client: Groq, messages: list[dict], max_tokens: int = 800) -> dict:
    raw = chat(client, messages, max_tokens=max_tokens, temperature=0.0)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
