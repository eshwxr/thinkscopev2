"""Thin wrapper around Groq's free-tier API for Qwen3.

Three defenses against free-tier limits, in order of preference:
1. Disk cache (app/data/llm_cache/): identical prompt + params returns the
   stored response without an API call at all -- re-runs of eval scripts
   burn zero quota. Safe because all calls use temperature 0.
2. Proactive inter-call delay: stay under the 30 RPM / 6K TPM caps most of
   the time instead of bouncing off them.
3. Exponential backoff + fresh client on 429 (and on hard timeout): waits
   grow 5s -> 10s -> 20s -> 40s with jitter. Fresh client per retry because
   a stale connection after a 429 has intermittently hung indefinitely,
   ignoring the SDK's own timeout.

reasoning_effort="none" disables Qwen3's <think> block entirely: the app only
needs structured JSON output, so a hidden reasoning trace just burns tokens
against the rate limit for no benefit here.
"""

import hashlib
import json
import random
import re
import signal
import time

from groq import Groq, RateLimitError

from app import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

HARD_CALL_TIMEOUT_SECONDS = 40
PROACTIVE_THROTTLE_SECONDS = 4.0
MAX_ATTEMPTS = 5

CACHE_DIR = config.DATA_DIR / "llm_cache"


class _HardTimeout(Exception):
    pass


def _raise_hard_timeout(signum, frame):
    raise _HardTimeout()


def get_client() -> Groq:
    # max_retries=0: we handle retries ourselves so failures surface immediately
    # instead of compounding with the SDK's own internal retry/backoff.
    return Groq(api_key=config.GROQ_API_KEY, timeout=30.0, max_retries=0)


def _cache_key(messages: list[dict], max_tokens: int, temperature: float) -> str:
    payload = json.dumps(
        {"model": config.LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_read(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())["text"]
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _cache_write(key: str, text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"text": text}))


def _call_with_hard_timeout(client: Groq, **kwargs):
    """signal.alarm forcibly interrupts a call the HTTP client's own timeout
    fails to cut off (observed intermittently after 429s: ~0% CPU hangs for
    10+ minutes past the configured 30s timeout). Alarm only works on the
    main thread -- from a worker thread (e.g. LangGraph's executor), fall
    back to the client's own timeout instead of crashing."""
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
    key = _cache_key(messages, max_tokens, temperature)
    cached = _cache_read(key)
    if cached is not None:
        return cached

    time.sleep(PROACTIVE_THROTTLE_SECONDS)
    completion = None
    for attempt in range(MAX_ATTEMPTS):
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
            wait_seconds = min(5 * (2**attempt), 60) + random.uniform(0, 2)
            reason = "rate limited" if isinstance(exc, RateLimitError) else "hard timeout hit"
            print(f"  {reason}, waiting {wait_seconds:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(wait_seconds)
            client = get_client()
    if completion is None:
        raise RuntimeError(f"Groq call failed after {MAX_ATTEMPTS} attempts (rate limit or hard timeout)")

    text = _THINK_RE.sub("", completion.choices[0].message.content or "").strip()
    _cache_write(key, text)
    return text


def chat_json(client: Groq, messages: list[dict], max_tokens: int = 800) -> dict:
    raw = chat(client, messages, max_tokens=max_tokens, temperature=0.0)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
