"""Phase 3: evaluation harness -- faithfulness, answer relevancy, context precision.

Implements the same three metric definitions RAGAS uses, but scored directly
through our own Groq client (app.llm) instead of the `ragas` package's
evaluate() pipeline: in testing, ragas 0.2.10's internal async executor
deadlocked against Groq's free tier (its own progress timer showed jobs
completing in seconds while real wall-clock time showed multi-minute
stalls with zero CPU activity -- a deadlock, not a slow network call, since
a genuine timeout would have surfaced through RunConfig's timeout/max_retries).
Our own LLM wrapper already has working self-throttling and retry logic
(proven across Phases 1-2), so metrics are scored with single-purpose LLM
calls through it instead.

Retrieval + generation is evaluated directly (single retrieve -> synthesize
call, not routed through the Planner/Critic retry loop) so the retrieval
method's effect on generation quality is isolated and comparable across
pipeline versions: dense-only (Phase 1 baseline) vs hybrid RRF (Phase 2).
"""

import json

from app import config, synthesis
from app.hybrid_retrieval import retrieve_hybrid
from app.llm import chat_json, get_client
from app.retrieval import retrieve_dense

EVAL_SAMPLE_SIZE = 10

FAITHFULNESS_PROMPT = """You are evaluating whether an answer's claims are faithful to (fully
supported by) the given context. Break the answer into its individual factual claims, then
judge what fraction are directly supported by the context.
Respond with ONLY JSON: {"faithfulness": <float 0.0-1.0>, "reasoning": "..."}"""

RELEVANCY_PROMPT = """You are evaluating whether an answer is relevant to the question asked
(not whether it's correct or grounded -- just whether it actually addresses the question).
Respond with ONLY JSON: {"answer_relevancy": <float 0.0-1.0>, "reasoning": "..."}"""

CONTEXT_PRECISION_PROMPT = """You are evaluating retrieval quality. Given a question and a list
of retrieved context chunks, judge what fraction of the chunks are actually relevant/useful
for answering the question.
Respond with ONLY JSON: {"context_precision": <float 0.0-1.0>, "reasoning": "..."}"""


def score_faithfulness(client, question: str, answer: str, contexts: list[str]) -> float:
    context_text = "\n\n".join(contexts)
    messages = [
        {"role": "system", "content": FAITHFULNESS_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context_text}"},
    ]
    try:
        return float(chat_json(client, messages, max_tokens=300).get("faithfulness", 0.0))
    except Exception:
        return 0.0


def score_answer_relevancy(client, question: str, answer: str) -> float:
    messages = [
        {"role": "system", "content": RELEVANCY_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ]
    try:
        return float(chat_json(client, messages, max_tokens=300).get("answer_relevancy", 0.0))
    except Exception:
        return 0.0


def score_context_precision(client, question: str, contexts: list[str]) -> float:
    context_text = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    messages = [
        {"role": "system", "content": CONTEXT_PRECISION_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nRetrieved chunks:\n{context_text}"},
    ]
    try:
        return float(chat_json(client, messages, max_tokens=300).get("context_precision", 0.0))
    except Exception:
        return 0.0


def score_query(client, retrieval_fn, question: str) -> dict:
    chunks = retrieval_fn(question)
    contexts = [c["text"] for c in chunks]
    answer = synthesis.synthesize_answer(client, question, chunks)

    return {
        "query": question,
        "faithfulness": score_faithfulness(client, question, answer, contexts),
        "answer_relevancy": score_answer_relevancy(client, question, answer),
        "context_precision": score_context_precision(client, question, contexts),
    }


def run_eval(version_name: str, retrieval_fn, questions: list[str]) -> dict:
    print(f"\n=== Eval: {version_name} ===", flush=True)
    client = get_client()

    rows = []
    for i, q in enumerate(questions):
        row = score_query(client, retrieval_fn, q)
        rows.append(row)
        print(
            f"  [{i+1}/{len(questions)}] {q[:50]} -- "
            f"F={row['faithfulness']:.2f} R={row['answer_relevancy']:.2f} P={row['context_precision']:.2f}",
            flush=True,
        )

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    import csv

    with (results_dir / f"phase3_eval_{version_name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    scores = {
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "answer_relevancy": sum(r["answer_relevancy"] for r in rows) / n,
        "context_precision": sum(r["context_precision"] for r in rows) / n,
    }
    print(f"{version_name}: {scores}", flush=True)
    return scores


def main() -> None:
    questions = json.loads((config.EVAL_DIR / "questions.json").read_text())[:EVAL_SAMPLE_SIZE]

    dense_scores = run_eval("dense_baseline", retrieve_dense, questions)
    hybrid_scores = run_eval("hybrid_phase2", retrieve_hybrid, questions)

    summary = {"dense_baseline": dense_scores, "hybrid_phase2": hybrid_scores}
    results_dir = config.EVAL_DIR / "results"
    (results_dir / "phase3_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
