"""Runs eval for ONE query as a short-lived process and appends a CSV row.

Long-running processes making many sequential Groq calls have intermittently
hung partway through (even with proactive throttling, hard timeouts, and
fresh-client-on-retry -- see app/llm.py); short single/paired-query processes
have been reliable every time in testing. Driving the harness as a sequence
of short-lived subprocess calls sidesteps whatever degrades over a long
session, at the cost of re-paying Python/model-load startup overhead per call.
"""

import csv
import json
import sys

from app import config
from app.hybrid_retrieval import retrieve_hybrid
from app.llm import get_client
from app.ragas_eval import score_query
from app.retrieval import retrieve_dense

RETRIEVAL_FNS = {"dense_baseline": retrieve_dense, "hybrid_phase2": retrieve_hybrid}


def main() -> None:
    version_name = sys.argv[1]
    index = int(sys.argv[2])

    questions = json.loads((config.EVAL_DIR / "questions.json").read_text())
    question = questions[index]

    client = get_client()
    row = score_query(client, RETRIEVAL_FNS[version_name], question)
    print(f"[{version_name} #{index}] {question[:50]} -- F={row['faithfulness']:.2f} "
          f"R={row['answer_relevancy']:.2f} P={row['context_precision']:.2f}")

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"phase3_eval_{version_name}.csv"

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
