"""Phase 1 checkpoint: % of queries resolved without hitting max retries.

This is the real number behind Bullet 1 ("reduced unresolved queries").
Run this once Phase 1 is stable, before tagging v0-baseline.
"""

import csv
import json

from app import config
from app.pipeline import answer_query


def run_checkpoint(output_name: str = "phase1_checkpoint.csv") -> None:
    questions = json.loads((config.EVAL_DIR / "questions.json").read_text())

    rows = []
    for i, query in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {query[:60]}", flush=True)
        result = answer_query(query)
        for r in result["results"]:
            rows.append(
                {
                    "query": query,
                    "sub_question": r["sub_question"],
                    "resolved": r["resolved"],
                    "retries_used": r["retries_used"],
                    "hit_max_retries": r["hit_max_retries"],
                }
            )
            status = "OK" if r["resolved"] else "UNRESOLVED"
            print(f"    [{status}] {r['sub_question'][:60]}", flush=True)

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / output_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    resolved = sum(1 for r in rows if r["resolved"])
    hit_max = sum(1 for r in rows if r["hit_max_retries"])
    print(f"\nSub-questions evaluated: {total}")
    print(f"Resolved: {resolved} ({resolved/total:.1%})")
    print(f"Hit max retries unresolved: {hit_max} ({hit_max/total:.1%})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    run_checkpoint()
