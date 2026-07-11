from groq import Groq

from app import llm

PLANNER_PROMPT = """You are the Planner in a research assistant system.
Break the user's query into 1-3 focused sub-questions that, together, fully answer it.
If the query is already narrow, return exactly one sub-question equal to the original query.
Respond with ONLY JSON, no other text: {"sub_questions": ["...", ...]}"""


def plan(client: Groq, query: str) -> list[str]:
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": query},
    ]
    result = llm.chat_json(client, messages)
    sub_questions = result["sub_questions"]
    return sub_questions[:3] if sub_questions else [query]
