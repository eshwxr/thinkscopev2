"""CI smoke test for the eval harness. Runs on a small fixed fixture (no
corpus, no retrieval) so it works in CI without the 84MB vector DB, which is
gitignored. Catches integration breakage -- LLM API changes, code bugs -- on
every push. NOT a quality-regression test: that needs the real corpus and
runs locally via `python -m app.ragas_eval` before merging pipeline changes,
per SOP Phase 3 (results stored under app/eval/results/).
"""

import sys

from app.llm import get_client
from app.ragas_eval import score_answer_relevancy, score_context_precision, score_faithfulness

QUESTION = "What is BM25?"
ANSWER = "BM25 is a probabilistic ranking function used in information retrieval that scores documents based on query term frequency and inverse document frequency."
CONTEXTS = [
    "BM25 is a bag-of-words retrieval function that ranks documents based on the query terms appearing in each document, widely used as a strong sparse retrieval baseline."
]


def main() -> None:
    client = get_client()

    faithfulness = score_faithfulness(client, QUESTION, ANSWER, CONTEXTS)
    relevancy = score_answer_relevancy(client, QUESTION, ANSWER)
    precision = score_context_precision(client, QUESTION, CONTEXTS)

    print(f"faithfulness={faithfulness}, answer_relevancy={relevancy}, context_precision={precision}")

    if faithfulness == 0.0 and relevancy == 0.0 and precision == 0.0:
        print("FAIL: all metrics returned 0.0 -- likely an API/parsing failure, not a real score")
        sys.exit(1)
    print("OK: eval harness runs end-to-end")


if __name__ == "__main__":
    main()
