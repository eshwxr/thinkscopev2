"""Phase 4 checkpoint: % of claims with a verified source, cited pipeline
vs ungrounded baseline (free-form synthesis, no citation instructions).
"""

import csv
import json

from app import config, synthesis
from app.citation_verifier import synthesize_with_citations, verify_claims, verify_ungrounded_answer
from app.hybrid_retrieval import retrieve_hybrid
from app.llm import get_client

SAMPLE_SIZE = 10


def score_query(client, query: str) -> dict:
    chunks = retrieve_hybrid(query)

    cited_claims = synthesize_with_citations(client, query, chunks)
    cited_verified = verify_claims(client, cited_claims, chunks)

    ungrounded_answer = synthesis.synthesize_answer(client, query, chunks)
    ungrounded_verified = verify_ungrounded_answer(client, ungrounded_answer, chunks)

    cited_ratio = sum(1 for c in cited_verified if c["verified"]) / len(cited_verified) if cited_verified else 0.0
    ungrounded_ratio = (
        sum(1 for c in ungrounded_verified if c["verified"]) / len(ungrounded_verified)
        if ungrounded_verified
        else 0.0
    )

    return {
        "query": query,
        "cited_claim_count": len(cited_verified),
        "cited_verified_ratio": cited_ratio,
        "ungrounded_claim_count": len(ungrounded_verified),
        "ungrounded_verified_ratio": ungrounded_ratio,
    }


def run_phase4_eval(output_name: str = "phase4_citation_verification.csv") -> None:
    questions = json.loads((config.EVAL_DIR / "questions.json").read_text())[:SAMPLE_SIZE]
    client = get_client()

    rows = []
    for i, query in enumerate(questions):
        row = score_query(client, query)
        rows.append(row)
        print(
            f"[{i+1}/{len(questions)}] {query[:50]} -- "
            f"cited={row['cited_verified_ratio']:.0%} ungrounded={row['ungrounded_verified_ratio']:.0%}",
            flush=True,
        )

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / output_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    avg_cited = sum(r["cited_verified_ratio"] for r in rows) / n
    avg_ungrounded = sum(r["ungrounded_verified_ratio"] for r in rows) / n
    print(f"\nQueries evaluated: {n}")
    print(f"Cited pipeline avg verified-claim ratio: {avg_cited:.1%}")
    print(f"Ungrounded baseline avg verified-claim ratio: {avg_ungrounded:.1%}")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    run_phase4_eval()
