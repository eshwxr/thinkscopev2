from groq import Groq

from app import llm

CRITIC_PROMPT = """You are the Critic in a research assistant system.
You are given a sub-question and chunks retrieved from a paper corpus to answer it.
Judge whether the chunks are sufficient to construct a reasonably accurate answer --
"sufficient" means a correct, useful answer can be built from what's here, even if it
isn't exhaustive or doesn't cover every possible nuance. Do not require the chunks to
be a complete textbook explanation; require only that they support a correct answer.
If not sufficient, propose a reformulated search query that would retrieve better chunks
(e.g. more specific terms, synonyms, or a narrower angle).
Respond with ONLY JSON, no other text:
{"sufficient": true/false, "reason": "...", "reformulated_query": "..." }
"reformulated_query" should equal the sub-question if sufficient is true."""


CHUNK_PREVIEW_CHARS = 600


def critique(client: Groq, sub_question: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(f"[{c['id']}] {c['text'][:CHUNK_PREVIEW_CHARS]}" for c in chunks)
    messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": f"Sub-question: {sub_question}\n\nRetrieved chunks:\n{context}"},
    ]
    return llm.chat_json(client, messages, max_tokens=300)
