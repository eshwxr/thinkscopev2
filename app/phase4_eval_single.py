"""Runs Phase 4 citation-verification eval for ONE query as a short-lived
process and appends a CSV row. Same rationale as app/ragas_eval_single.py:
short-lived per-query processes have been reliable, long-running loops making
many sequential Groq calls have not.
"""

import csv
import json
import sys

from app import config
from app.llm import get_client
from app.phase4_eval import score_query


def main() -> None:
    index = int(sys.argv[1])

    questions = json.loads((config.EVAL_DIR / "questions.json").read_text())
    question = questions[index]

    client = get_client()
    row = score_query(client, question)
    print(f"[#{index}] {question[:50]} -- cited={row['cited_verified_ratio']:.0%} ungrounded={row['ungrounded_verified_ratio']:.0%}")

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "phase4_citation_verification.csv"

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
