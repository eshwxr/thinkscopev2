"""Citation verification: force the synthesis LLM to tag each claim with the
chunk ID it came from, then a second LLM call checks that the cited chunk
actually supports the claim.

Deliberately NOT semantic span-matching (token-to-chunk embedding alignment)
per SOP -- that's a separate, deeper NLP problem. LLM-tagged chunk-ID +
verify-the-chunk-supports-it gets most of the credibility for a fraction
of the effort.
"""

import re

from groq import Groq

from app import llm

CITED_SYNTHESIS_PROMPT = """You are a research assistant. Answer the question using ONLY the
provided context chunks. Write the answer as short claims (one sentence each). After every
claim, cite the exact chunk ID it is based on in square brackets, e.g. [chunk_id_here].
Respond with ONLY JSON, no other text:
{"claims": [{"text": "...", "chunk_id": "..."}, ...]}"""

VERIFY_PROMPT = """You are a fact-checker. You are given a claim and a source text chunk it was
attributed to. Judge whether the chunk actually supports the claim.
Respond with ONLY JSON, no other text: {"supported": true/false}"""


def synthesize_with_citations(client: Groq, question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{c['id']}] {c['text'][:500]}" for c in chunks)
    messages = [
        {"role": "system", "content": CITED_SYNTHESIS_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nContext chunks:\n{context}"},
    ]
    try:
        result = llm.chat_json(client, messages, max_tokens=600)
        return result.get("claims", [])
    except Exception:
        return []


def verify_claim(client: Groq, claim_text: str, chunk_text: str) -> bool:
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": f"Claim: {claim_text}\n\nSource chunk: {chunk_text[:500]}"},
    ]
    try:
        result = llm.chat_json(client, messages, max_tokens=100)
        return bool(result.get("supported", False))
    except Exception:
        return False


def verify_claims(client: Groq, claims: list[dict], chunks: list[dict]) -> list[dict]:
    chunk_by_id = {c["id"]: c for c in chunks}
    verified = []
    for claim in claims:
        chunk = chunk_by_id.get(claim.get("chunk_id"))
        supported = verify_claim(client, claim["text"], chunk["text"]) if chunk else False
        verified.append({**claim, "verified": supported})
    return verified


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_claims(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def verify_ungrounded_answer(client: Groq, answer_text: str, chunks: list[dict]) -> list[dict]:
    """Baseline for comparison: split a free-form (non-cited) answer into claims and check
    each against ALL retrieved chunks combined, since there's no specific citation to check."""
    combined_context = "\n\n".join(c["text"][:500] for c in chunks)
    results = []
    for claim_text in split_claims(answer_text):
        supported = verify_claim(client, claim_text, combined_context)
        results.append({"text": claim_text, "verified": supported})
    return results
