# ThinkScope V2

Autonomous multi-agent research intelligence RAG system. Built from scratch (not forked from V1) as a placement project — see [SOP.md](./SOP.md) for the full build plan, target resume bullets, and phase-by-phase scope, and [WRITEUP.md](./WRITEUP.md) for the final polished resume bullets and interview walkthrough.

## Status

**Phases 1-4 complete.** Real numbers below, all reproducible via the scripts referenced. FastAPI service verified locally; Docker image build is pending (paused mid-build due to low disk space on the dev machine — `Dockerfile`/`.dockerignore` are in place and tested up to the dependency-install stage, will finish once space is freed).

## Architecture

```
Planner -> Retrieval (hybrid dense + BM25, RRF fusion) -> Critic (retry loop, max 2)
        -> Cited Answer Synthesis -> Citation Verifier -> FastAPI /query endpoint
```

- **Orchestration:** LangGraph (`app/graph.py`)
- **Retrieval:** ChromaDB (`all-MiniLM-L6-v2` dense embeddings) + BM25 (`rank-bm25`), fused via Reciprocal Rank Fusion (`app/hybrid_retrieval.py`)
- **LLM:** Qwen3-32B via Groq's free tier
- **Corpus:** 110 arXiv papers (RAG/LLM-agents topic), fixed-size chunking with overlap, 10,713 chunks
- **Eval:** custom harness implementing RAGAS's metric definitions (faithfulness, answer relevancy, context precision) — see below for why not the `ragas` package directly
- **Serving:** FastAPI (`app/api.py`) + Docker, local only (no Redis, no cloud)

## Results (real, measured — not estimated)

**v2 numbers (current — 8-question eval set built from actual paper content, not generic definitional questions):**

| Phase | Metric | Result |
|---|---|---|
| **1 — Multi-agent core** | Resolved rate, dense vs hybrid (8 questions, 10 sub-questions) | dense: **40.0%** (4/10) · hybrid: **50.0%** (5/10) — up sharply from v1's 6.7%, see note below |
| **2 — Hybrid search** | precision@5, dense-only vs hybrid RRF (25 title-as-query samples, unchanged methodology) | **76.8% → 91.2%** (+18.8% relative) |
| **3 — Eval harness** | faithfulness / answer relevancy / context precision, dense vs hybrid (8 queries) | dense: 0.706 / 0.819 / 0.469 · hybrid: **0.944 / 0.931 / 0.650** — hybrid now wins on all three metrics |
| **4 — Citation verification** | % claims with a verified source, cited pipeline vs ungrounded baseline (8 queries) | **93.8% vs 43.8%** ungrounded (+114.3% relative) |
| **4 — Production latency** | Median latency, `/query` endpoint, local FastAPI (4 samples) | **~56.9s median** — dominated by free-tier self-throttling (Groq's 30 RPM / 6K TPM cap), not model/infra latency; see note below |

**Why the numbers moved between v1 and v2:** v1's eval questions were generic and definitional ("What is BM25?"), asked against a corpus of 110 research papers that *use* these concepts rather than explain them — so even a working pipeline scored low. v2's 8 questions were built by sampling actual paper content and writing questions whose answers demonstrably exist in specific chunks (e.g. "What F1-score does FAIR-RAG achieve on HotpotQA?"). No retrieval, chunking, retry, or prompt logic was tuned to raise these numbers — only the question set changed, closing the corpus/question mismatch identified after the v1 write-up. Phase 3's dense-vs-hybrid contradiction from v1 (where hybrid didn't clearly win) also resolved once the questions were better-matched to the corpus — that discrepancy was very likely a symptom of the same mismatch, not a real property of hybrid search. v1's numbers and the CSVs behind them are kept in `app/eval/results/*_dense.csv` / git history, not deleted.

**Bug fixes made alongside the rerun** (not score-tuning): `app/graph.py`'s Critic node and `app/pipeline.py`'s Planner call only caught JSON-parsing exceptions, not actual API failures (rate-limit exhaustion, network errors) — either would have crashed an entire eval run instead of marking one query unresolved. Both broadened to catch any exception and degrade gracefully. `app/llm.py` also gained a disk cache (identical prompts return instantly on re-run, zero quota spent) and real exponential backoff (5s → 10s → 20s → 40s + jitter) on top of the existing proactive throttle and hard timeout.

**Why a custom eval harness instead of the `ragas` package's `evaluate()`:** in testing, `ragas` 0.2.10's internal async executor deadlocked against Groq's free-tier rate limits (its own progress timer showed jobs completing in seconds while real wall-clock time showed multi-minute stalls with zero CPU — a deadlock, not a slow call). The harness implements the same three metric definitions directly through our own Groq client instead. See `app/ragas_eval.py` docstring for the full story.

**Why per-query subprocesses instead of one long-running eval script:** long-running processes making many sequential Groq calls degraded and eventually hung, even with proactive rate-limit throttling, a hard OS-level call timeout, and fresh-client-on-retry. Short-lived single-query processes have been reliable every time. See `app/ragas_eval_single.py` / `app/phase4_eval_single.py`.

## Project layout

- `app/` — all application code, added phase by phase:
  - `planner.py`, `retrieval.py`, `hybrid_retrieval.py`, `bm25_index.py`, `critic.py`, `graph.py`, `pipeline.py` — Phase 1-2 core loop
  - `synthesis.py`, `ragas_eval.py`, `ragas_eval_single.py`, `ragas_smoketest.py` — Phase 3 eval harness
  - `citation_verifier.py`, `api.py`, `phase4_eval.py`, `phase4_eval_single.py` — Phase 4 production layer
  - `ingest.py` — arXiv corpus pull + chunk + embed into ChromaDB
  - `app/data/` — corpus (gitignored, regenerate via `python -m app.ingest`)
  - `app/eval/` — question sets, labeled sets, and all results (`app/eval/results/`)
- `.github/workflows/eval.yml` — CI smoke test on every push (fixed fixture, no corpus needed — see harness note above)
- `Dockerfile` — local FastAPI service

## Setup

```
uv sync
cp .env.example .env   # fill in HF_TOKEN and GROQ_API_KEY
```

Build the corpus once (takes a while — arXiv downloads + embedding):
```
uv run python -m app.ingest
```

Run a single query through the full pipeline:
```
uv run python -m app.main "How does BM25 differ from dense vector retrieval?"
```

Run the FastAPI service:
```
uv run uvicorn app.api:app --port 8811
curl -X POST http://127.0.0.1:8811/query -H "Content-Type: application/json" -d '{"query": "What is BM25?"}'
```

Run in Docker (pending — see Status above):
```
docker build -t thinkscopev2 .
docker run -p 8811:8000 --env-file .env thinkscopev2
```

Reproduce the checkpoint numbers:
```
uv run python -m app.checkpoint                     # Phase 1, hybrid retrieval (default)
RETRIEVAL_MODE=dense uv run python -m app.checkpoint # Phase 1, dense-only
uv run python -m app.phase2_eval                     # Phase 2
uv run python -m app.ragas_eval                      # Phase 3 (or app.ragas_eval_single for one query at a time)
uv run python -m app.phase4_eval                     # Phase 4 (or app.phase4_eval_single for one query at a time)
```

## Notes on the free-tier LLM

Qwen3-32B via Groq's free developer tier (30 RPM / 6K TPM) is fine for this scope but shapes some engineering decisions worth knowing about if you're reading the code: `app/llm.py` proactively throttles calls, disables Qwen3's `<think>` reasoning block (`reasoning_effort="none"`) since only structured JSON output is needed, has a hard SIGALRM-based timeout as a backstop since the SDK's own timeout intermittently failed to fire under sustained load, retries on 429 with real exponential backoff, and caches every response to disk (`app/data/llm_cache/`, keyed on the exact prompt + params) so re-running an eval script after a crash or a code fix doesn't re-spend quota on prompts already answered.
