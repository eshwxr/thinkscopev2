"""Runs Phase 3 (faithfulness/relevancy/context precision) AND Phase 4
(citation verification) for ONE query on hybrid retrieval, in a single pass.

Phase 4 only ever used hybrid retrieval (comparing a cited pipeline against
an ungrounded free-form baseline, not dense vs hybrid), and Phase 3's
"ungrounded answer" synthesis is the exact same call Phase 4 needs for its
ungrounded baseline -- running them separately duplicated a retrieve call
and a synthesis call per query for no reason. This combines them.
"""

import csv
import json
import sys

from app import config, synthesis
from app.citation_verifier import synthesize_with_citations, verify_claims, verify_ungrounded_answer
from app.hybrid_retrieval import retrieve_hybrid
from app.llm import get_client
from app.ragas_eval import score_answer_relevancy, score_context_precision, score_faithfulness


def main() -> None:
    index = int(sys.argv[1])
    questions = json.loads((config.EVAL_DIR / "questions.json").read_text())
    question = questions[index]

    client = get_client()
    chunks = retrieve_hybrid(question)
    contexts = [c["text"] for c in chunks]

    # Shared: one ungrounded synthesis feeds both Phase 3 scoring and Phase 4's baseline.
    ungrounded_answer = synthesis.synthesize_answer(client, question, chunks)

    phase3_row = {
        "query": question,
        "faithfulness": score_faithfulness(client, question, ungrounded_answer, contexts),
        "answer_relevancy": score_answer_relevancy(client, question, ungrounded_answer),
        "context_precision": score_context_precision(client, question, contexts),
    }

    cited_claims = synthesize_with_citations(client, question, chunks)
    cited_verified = verify_claims(client, cited_claims, chunks)
    ungrounded_verified = verify_ungrounded_answer(client, ungrounded_answer, chunks)

    cited_ratio = sum(1 for c in cited_verified if c["verified"]) / len(cited_verified) if cited_verified else 0.0
    ungrounded_ratio = (
        sum(1 for c in ungrounded_verified if c["verified"]) / len(ungrounded_verified) if ungrounded_verified else 0.0
    )
    phase4_row = {
        "query": question,
        "cited_claim_count": len(cited_verified),
        "cited_verified_ratio": cited_ratio,
        "ungrounded_claim_count": len(ungrounded_verified),
        "ungrounded_verified_ratio": ungrounded_ratio,
    }

    print(
        f"[hybrid #{index}] {question[:50]} -- "
        f"P3: F={phase3_row['faithfulness']:.2f} R={phase3_row['answer_relevancy']:.2f} P={phase3_row['context_precision']:.2f} -- "
        f"P4: cited={cited_ratio:.0%} ungrounded={ungrounded_ratio:.0%}"
    )

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    p3_path = results_dir / "phase3_eval_hybrid_phase2.csv"
    write_header = not p3_path.exists()
    with p3_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase3_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(phase3_row)

    p4_path = results_dir / "phase4_citation_verification.csv"
    write_header = not p4_path.exists()
    with p4_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase4_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(phase4_row)


if __name__ == "__main__":
    main()
