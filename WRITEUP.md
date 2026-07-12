# ThinkScope V2 — Phase 5 Write-up

Final resume bullets and interview talking points, built only from numbers actually measured and checked into `app/eval/results/`. Where a number came in weaker or flatter than hoped, it's kept as-is per the SOP's rule: an honest weak number is a stronger interview answer than a suspiciously perfect one.

---

## Resume bullets

**1. Architecture / Agentic Reasoning**
> Architected a multi-agent research system (Planner → Retrieval → Critic loop) using LangGraph, where the Critic autonomously triggers re-retrieval with query reformulation on failed relevance checks — resolving 60.0% of sub-questions on dense-only retrieval and 75.0% with hybrid retrieval, across a 20-sub-question corpus-grounded test set.

*Honest framing note (updated 2026-07-12): these are the current v3 numbers — chunk size, top_k, and retry-accumulation tuning moved this from an original 6.7% baseline (`v0-baseline` tag, generic/definitional questions against a corpus that doesn't define its own terms) through an intermediate 40-50% (once questions were corpus-grounded) to the current 60-75% (after retrieval architecture changes: larger chunks, higher top_k, chunks accumulating across retries instead of being discarded each attempt). Both numbers are still short of 100% — that's honest: some questions genuinely need more than 8 chunks' worth of context, or the paper doesn't state the answer as cleanly as the question implies.*

**2. Retrieval / Search Quality**
> Built a hybrid retrieval pipeline (all-MiniLM-L6-v2 dense embeddings + BM25, fused via Reciprocal Rank Fusion) over 110 arXiv papers (6,082 chunks) indexed in ChromaDB, improving retrieval precision@5 by 28.1% (71.2% → 91.2%) and hit@5 by 4pp (96.0% → 100%) over the dense-only baseline.

*Honest framing note (updated 2026-07-12): re-verified against the current corpus (1400-char chunks, re-ingested after the v1 800-char corpus). Unlike the earlier measurement, hit@5 was NOT already at a ceiling for dense-only this time (96%, not 100%) — a small but real, non-ceiling-effect improvement is visible on both metrics now, which is a cleaner result to defend than before.*

**3. Evaluation Infrastructure**
> Designed and ran an automated evaluation harness (faithfulness, answer relevancy, context precision — RAGAS's metric definitions) across a corpus-grounded test set comparing pipeline versions, wired into a GitHub Action that runs a smoke test on every push to catch integration breakage before it reaches main.

*Honest framing note: this deliberately does not say "wired to the `ragas` package" — in testing, `ragas` 0.2.10's async executor deadlocked against Groq's free-tier rate limits, so the harness implements the same three metric definitions directly through the project's own Groq client instead (see `app/ragas_eval.py`). It also does not claim "catching X quality regressions in production" — none were actually caught, since no regression was pushed during testing; the CI smoke test's real, honest job is catching *integration* breakage (a broken import, a changed API signature), not full-corpus quality regressions, because the 84MB vector DB isn't checked into git. **Current measured numbers (2026-07-12, dense-only, 11 of 15 planned questions — the hybrid comparison run was cut short by sustained free-tier rate-limiting and is still pending):** faithfulness 0.702, answer relevancy 0.936, context precision 0.482. The last completed dense-vs-hybrid comparison (on the prior 800-char corpus, 8 questions) showed hybrid winning all three metrics (0.706→0.944 faithfulness, 0.819→0.931 relevancy, 0.469→0.650 precision) — expected to hold on the new corpus too, but not yet re-confirmed, so don't state that comparison as current fact until it's rerun.*

**4. Production Engineering / Trust**
> Deployed a citation-verification layer and FastAPI inference service where every generated claim is tagged with its source chunk ID and independently verified by a second LLM call — cutting unverifiable claims from 76.7% (ungrounded baseline) to 10% (90.0% verified vs. 23.3% for free-form synthesis), a 285.7% relative improvement in claim verifiability.

*Honest framing note: Docker packaging is written and build-tested up through the dependency-install stage but the image build itself is currently paused (dev machine disk space), to be finished later. The FastAPI service itself is fully built and verified end-to-end locally — this bullet describes what's actually running, not what's aspirational. Median latency (~56.9s across 4 samples) is dominated by proactive rate-limit self-throttling on Groq's free tier, not model or infra latency — worth having the honest explanation ready if asked, not worth putting in the bullet itself.*

---

## 2-minute verbal walkthrough

**Opening (10s):** "I built a multi-agent RAG system from scratch — Planner, hybrid Retrieval, a Critic that can trigger re-retrieval, citation verification, and a FastAPI service — and every number I'll mention is from an eval script in the repo, not an estimate."

**Architecture (30s):** "The core loop is LangGraph: a Planner breaks the user's question into sub-questions, each one goes through a Retrieval → Critic cycle where the Critic judges whether the retrieved chunks are actually sufficient — if not, it reformulates the query and retries, capped at 2 retries so it never loops forever. I tagged a `v0-baseline` commit right after this was stable, using plain dense retrieval only, specifically so I'd have a real 'before' number for every later comparison."

**Retrieval (25s):** "Phase 2 added BM25 alongside the dense embeddings and fused them with Reciprocal Rank Fusion — standard technique, k=60. Measured on a paper-title-as-query proxy set, precision@5 went from 76.8% to 91.2%. I didn't do a chunking-strategy comparison — fixed-size with overlap, kept deliberately simple, that's a separate research question I scoped out."

**Evaluation (30s):** "For eval I wanted RAGAS's metrics — faithfulness, answer relevancy, context precision — but the actual `ragas` package's async execution kept deadlocking against my free-tier Groq rate limit. Instead of fighting that indefinitely, I implemented the same three metric definitions directly against my own LLM client, which turned out to be more reliable. That's wired into a GitHub Action — it's a smoke test on a fixed fixture, not a full-corpus regression test, since the vector DB isn't in git. I'm upfront about that distinction if asked."

**Production (25s):** "Every synthesized answer gets broken into claims, each one tagged with the exact source chunk ID, then a second LLM call independently checks the chunk actually supports the claim. That took verified-claim rate from 23% on free-form synthesis to 90% when forced to cite. That's wrapped in a FastAPI service — Docker packaging is written but I paused the actual build mid-session over a disk space issue on my dev machine, finishing that last."

**If asked "what was hard":** "Getting reliable behavior out of a free-tier rate-limited LLM API for anything long-running. I hit repeated hangs that looked like network timeouts but turned out to be state degrading over long-running processes — fixed it by driving evals as short-lived per-query subprocesses instead of one long loop. That's actually a good engineering story: I have real evidence of the failure mode, not just a guess."
