"""Hybrid retrieval: dense (ChromaDB/all-MiniLM-L6-v2) + sparse (BM25), fused via
Reciprocal Rank Fusion. Each method retrieves a wider candidate pool than the final
top_k so fusion has enough signal to reorder well.

RRF score(d) = sum over each ranking r containing d of 1 / (RRF_K + rank_r(d))
Standard RRF_K=60, per the original Cormack et al. RRF paper's default.
"""

from app import config
from app.bm25_index import get_bm25_index
from app.retrieval import retrieve_dense

RRF_K = 60
CANDIDATE_POOL = 20


def retrieve_hybrid(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    dense_results = retrieve_dense(query, top_k=CANDIDATE_POOL)
    bm25_results = get_bm25_index().search(query, top_k=CANDIDATE_POOL)

    chunk_by_id = {}
    rrf_scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results):
        chunk_by_id[chunk["id"]] = chunk
        rrf_scores[chunk["id"]] = rrf_scores.get(chunk["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, chunk in enumerate(bm25_results):
        chunk_by_id.setdefault(chunk["id"], chunk)
        rrf_scores[chunk["id"]] = rrf_scores.get(chunk["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    return [{**chunk_by_id[cid], "rrf_score": rrf_scores[cid]} for cid in ranked_ids]
