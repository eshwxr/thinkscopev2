from app import config, planner
from app.graph import run_retrieval_critic_loop
from app.llm import get_client


def answer_query(query: str) -> dict:
    client = get_client()
    try:
        sub_questions = planner.plan(client, query)
    except Exception:
        # Any planner failure (unparseable JSON, exhausted rate-limit retries,
        # network error) falls back to treating the original query as the
        # single sub-question, instead of crashing the whole run.
        sub_questions = [query]

    results = []
    for sub_question in sub_questions:
        loop_result = run_retrieval_critic_loop(sub_question)
        results.append(
            {
                "sub_question": sub_question,
                "resolved": loop_result["resolved"],
                "retries_used": loop_result["retries"],
                "hit_max_retries": loop_result["retries"] >= config.MAX_RETRIEVAL_RETRIES
                and not loop_result["resolved"],
                "reason": loop_result["reason"],
                "chunks": loop_result["chunks"],
            }
        )

    return {"query": query, "sub_questions": sub_questions, "results": results}
