"""FastAPI service wrapping the full pipeline: Planner -> Retrieval (hybrid) ->
Critic loop -> cited answer synthesis -> citation verification.

Local Docker + FastAPI only, per SOP -- no Redis, no cloud deployment.
"""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from app import planner
from app.citation_verifier import synthesize_with_citations, verify_claims
from app.hybrid_retrieval import retrieve_hybrid
from app.llm import get_client

app = FastAPI(title="ThinkScope V2")


class QueryRequest(BaseModel):
    query: str


class ClaimResponse(BaseModel):
    text: str
    chunk_id: str | None = None
    verified: bool


class QueryResponse(BaseModel):
    query: str
    sub_questions: list[str]
    claims: list[ClaimResponse]
    verified_claim_ratio: float
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    t0 = time.time()
    client = get_client()

    sub_questions = planner.plan(client, request.query)

    all_claims = []
    for sub_question in sub_questions:
        chunks = retrieve_hybrid(sub_question)
        raw_claims = synthesize_with_citations(client, sub_question, chunks)
        verified = verify_claims(client, raw_claims, chunks)
        all_claims.extend(verified)

    verified_count = sum(1 for c in all_claims if c["verified"])
    ratio = verified_count / len(all_claims) if all_claims else 0.0

    return QueryResponse(
        query=request.query,
        sub_questions=sub_questions,
        claims=[ClaimResponse(**c) for c in all_claims],
        verified_claim_ratio=ratio,
        latency_ms=(time.time() - t0) * 1000,
    )
