"""Thin wrapper around Groq's free-tier API for Qwen3.

reasoning_effort="none" disables Qwen3's <think> block entirely: the app only
needs structured JSON output, so paying for a hidden reasoning trace on every
call just burns tokens against the free-tier rate limit for no benefit here.
"""

import json
import re
import time

from groq import Groq, RateLimitError

from app import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


SELF_THROTTLE_SECONDS = 2.5


def get_client() -> Groq:
    # max_retries=0: we handle retries ourselves so failures surface immediately
    # instead of compounding with the SDK's own internal retry/backoff.
    return Groq(api_key=config.GROQ_API_KEY, timeout=30.0, max_retries=0)


def chat(client: Groq, messages: list[dict], max_tokens: int = 800, temperature: float = 0.2) -> str:
    completion = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort="none",
            )
            break
        except RateLimitError:
            wait_seconds = 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait_seconds}s (attempt {attempt + 1}/3)")
            time.sleep(wait_seconds)
    if completion is None:
        raise RuntimeError("Groq rate limit exceeded after 3 attempts")

    time.sleep(SELF_THROTTLE_SECONDS)
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
