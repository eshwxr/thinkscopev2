from groq import Groq

from app import llm

SYNTHESIS_PROMPT = """You are a research assistant. Answer the user's question using ONLY
the provided context chunks. Be concise (2-4 sentences). If the context doesn't fully
answer the question, answer with what's available rather than refusing."""


def synthesize_answer(client: Groq, question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['id']}] {c['text'][:500]}" for c in chunks)
    messages = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    return llm.chat(client, messages, max_tokens=300)
