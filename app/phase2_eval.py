"""Phase 2 checkpoint: recall@5 / precision@5, dense-only vs hybrid (BM25+RRF).

Ground truth method: since manually judging relevance for 20-30 queries against
a 10k-chunk corpus is a separate multi-hour task on its own, we use a standard
IR proxy instead -- each paper's title is used as a query, and any chunk from
that same paper is treated as relevant. This is honest and documented, not
hidden: it's a paper-level proxy, not manually-graded chunk-level judgment.

Because each paper has 20-200+ chunks, classic recall@5 (relevant retrieved /
total relevant) is near-zero for every method by construction and wouldn't
discriminate quality. We report two proxy metrics instead:
  - precision@5: fraction of the top-5 chunks that belong to the target paper
  - hit@5 (paper-level recall): 1 if any of the top-5 chunks belong to the
    target paper, else 0
"""

import csv
import json
import random

from app import config
from app.hybrid_retrieval import retrieve_hybrid
from app.retrieval import get_collection, retrieve_dense

SAMPLE_SIZE = 25
TOP_K = 5


def build_labeled_queries(sample_size: int = SAMPLE_SIZE, seed: int = 42) -> list[dict]:
    collection = get_collection()
    data = collection.get(include=["metadatas"])

    papers: dict[str, str] = {}
    for meta in data["metadatas"]:
        papers.setdefault(meta["paper_id"], meta["title"])

    rng = random.Random(seed)
    sampled_ids = rng.sample(list(papers.keys()), min(sample_size, len(papers)))
    return [{"query": papers[pid], "target_paper_id": pid} for pid in sampled_ids]


def score(chunks: list[dict], target_paper_id: str, top_k: int = TOP_K) -> dict:
    top = chunks[:top_k]
    hits = sum(1 for c in top if c["metadata"]["paper_id"] == target_paper_id)
    return {
        "precision_at_5": hits / top_k,
        "hit_at_5": 1 if hits > 0 else 0,
    }


def run_phase2_eval(output_name: str = "phase2_dense_vs_hybrid.csv") -> None:
    labeled_queries = build_labeled_queries()
    (config.EVAL_DIR / "phase2_labeled_queries.json").write_text(json.dumps(labeled_queries, indent=2))

    rows = []
    for i, item in enumerate(labeled_queries):
        query, target = item["query"], item["target_paper_id"]

        dense_chunks = retrieve_dense(query, top_k=TOP_K)
        hybrid_chunks = retrieve_hybrid(query, top_k=TOP_K)

        dense_score = score(dense_chunks, target)
        hybrid_score = score(hybrid_chunks, target)

        rows.append(
            {
                "query": query,
                "dense_precision_at_5": dense_score["precision_at_5"],
                "dense_hit_at_5": dense_score["hit_at_5"],
                "hybrid_precision_at_5": hybrid_score["precision_at_5"],
                "hybrid_hit_at_5": hybrid_score["hit_at_5"],
            }
        )
        print(f"[{i+1}/{len(labeled_queries)}] {query[:60]}", flush=True)

    results_dir = config.EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / output_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    dense_p = sum(r["dense_precision_at_5"] for r in rows) / n
    hybrid_p = sum(r["hybrid_precision_at_5"] for r in rows) / n
    dense_hit = sum(r["dense_hit_at_5"] for r in rows) / n
    hybrid_hit = sum(r["hybrid_hit_at_5"] for r in rows) / n

    print(f"\nQueries evaluated: {n}")
    print(f"Dense-only  precision@5: {dense_p:.1%}, hit@5: {dense_hit:.1%}")
    print(f"Hybrid RRF  precision@5: {hybrid_p:.1%}, hit@5: {hybrid_hit:.1%}")
    if dense_p > 0:
        print(f"Precision@5 improvement: {(hybrid_p - dense_p) / dense_p:+.1%}")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    run_phase2_eval()
