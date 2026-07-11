# ThinkScope V2 — Project SOP
### Autonomous Multi-Agent Research Intelligence Platform

> **Ground rule:** every feature below exists because it earns a specific word in a specific bullet. If you're tempted to add something not traceable to a bullet, don't — it dilutes build time and gives an interviewer a thread to pull that you can't defend.

---

## Part 1 — Target Resume Bullets (reverse-engineered first)

These are the 4 bullets you're building toward. Numbers marked `[X]` are **targets to hit, not to fabricate** — you run the eval, you get a real number, you plug it in. If the real number is worse than the target, the bullet still works with the honest number; it just won't be as punchy. Never write a number you haven't measured.

**1. Architecture / Agentic Reasoning**
> Architected a multi-agent research system (Planner → Retrieval → Critic loop) using LangGraph, where the Critic autonomously triggers re-retrieval on failed relevance checks — reducing unresolved queries from `[baseline]` to `[X]`% across a `[N]`-question test set.

**2. Retrieval / Search Quality**
> Built a hybrid retrieval pipeline (`[embedding model, e.g. all-MiniLM-L6-v2]` dense embeddings + BM25, fused via Reciprocal Rank Fusion) over `[N]`+ research papers indexed in ChromaDB, improving context precision by `[X]`% and recall@5 by `[X]`% over the single-method baseline.
>
> *Naming the model/DB is free — you already know these. Do NOT add a chunking-strategy comparison (semantic vs. fixed-size vs. parent-child) to earn an "optimized chunking" claim — that's a separate sub-project's worth of experiments. Use plain fixed-size chunking with overlap and say so honestly if asked; "optimized" is only earned if you actually ran and compared alternatives.*

**3. Evaluation Infrastructure**
> Designed and ran an automated RAGAS-based evaluation harness (faithfulness, answer relevance, context precision) across `[N]` test queries, wired into a GitHub Action that reruns on every push — catching `[X]` quality regressions across pipeline versions before they reached the main branch.
>
> *The GitHub Action is the one upgrade worth the extra 2-3 days: cheap, small, and directly answers "did you just run this once in a notebook?" — a question you will get asked.*

**4. Production Engineering / Trust**
> Deployed a citation-verification layer and FastAPI/Docker inference service achieving `[X]`ms median latency, where every generated claim is checked against its source chunk via LLM-tagged chunk-ID verification — cutting unverifiable claims by `[X]`% versus the ungrounded baseline.
>
> *Deliberately NOT semantic span-matching (token-to-chunk embedding alignment) — that's a standalone NLP problem with its own tuning surface. LLM-tagged chunk-ID + verify-the-chunk-supports-it gets ~80% of the credibility for ~20% of the effort. Also deliberately NOT Redis or cloud deployment (AWS/GCP) — local Docker + FastAPI is sufficient proof of "I can ship this as a service," and every extra infra piece is a shallow-depth topic you'll own in an interview instead of a deep one.*

Together these 4 bullets prove: **systems design, retrieval engineering, evaluation rigor, and production trust** — the exact four things the 2026 hiring signal keeps pointing to (agent reliability, evaluation discipline, not-just-another-RAG-chatbot).

---

## Part 2 — Architecture Overview

```
                        ┌─────────────┐
      User Query  ───▶  │   PLANNER   │  breaks query into sub-questions
                        └──────┬──────┘
                               ▼
                     ┌───────────────────┐
                     │   RETRIEVAL AGENT │  hybrid search (dense + BM25)
                     └────────┬──────────┘
                               ▼
                     ┌───────────────────┐
                     │   CRITIC AGENT    │  checks relevance/sufficiency
                     └────────┬──────────┘
                        pass? │  fail? → loop back to Retrieval (max 2 retries)
                               ▼
                     ┌───────────────────┐
                     │ CITATION VERIFIER │  grounds every claim to a chunk
                     └────────┬──────────┘
                               ▼
                     ┌───────────────────┐
                     │  ANSWER SYNTHESIS │  final grounded response
                     └───────────────────┘

     Parallel: EVAL HARNESS (RAGAS) runs against every pipeline version
     Parallel: FastAPI + Docker wraps the whole graph as a service
```

**Tech stack:**
- Orchestration: LangGraph
- Retrieval: SentenceTransformers (dense) + rank-bm25 (sparse) + RRF fusion
- Vector store: ChromaDB (keep — no need to switch)
- LLM: Groq/LLaMA-3 (keep — already proven fast in V1) or swap to Claude API for critic/synthesis quality
- Evaluation: RAGAS + DeepEval (for regression suite)
- Serving: FastAPI + Docker
- Logging/observability: simple structured logs (JSON) — latency, retries, critic pass/fail per query

---

## Part 3 — Build Plan (Start to End)

### Phase 0 — Setup (Days 1-2)
- Fresh repo, built from scratch (not forked from V1) — fine, but this means your "before" numbers now have to be generated by you, not inherited
- Define your test corpus (100+ papers, can reuse the same set V1 used)
- Set up your eval question set skeleton now (even if empty) so it's ready the moment Phase 1 is stable

**Important since you're not forking V1:** at the end of Phase 1, once the Planner→Retrieval→Critic loop works with plain dense retrieval only (no hybrid search yet, no citation verification yet), **stop and tag that commit as your baseline** (e.g. `git tag v0-baseline`). Run your eval question set against it once. This tagged commit is now your only source of "before" numbers for Bullet 2 (hybrid search improvement) and Bullet 3 (first RAGAS run to compare later versions against). Skip this and you'll have nothing to compare your later numbers to — don't skip it even though it feels like a detour from forward progress.

### Phase 1 — Multi-Agent Core (Days 3-14) → earns Bullet 1
- Implement Planner node: given a query, output 1-3 sub-questions (simple prompt-based decomposition, no need for anything fancy)
- Implement Retrieval node: wraps your existing ChromaDB search
- Implement Critic node: given retrieved chunks + sub-question, output pass/fail + reason (structured output, e.g. JSON)
- Wire the retry loop: on fail, Critic reformulates the query once, max 2 retries, then falls through with best-effort chunks (never infinite-loop)
- **Checkpoint metric:** % of queries resolved without hitting max retries vs. % that needed retrieval help — this is your "reduced unresolved queries" number

### Phase 2 — Hybrid Search (Days 15-24) → earns Bullet 2
- Add BM25 index (rank-bm25 library) alongside existing dense retrieval
- Implement Reciprocal Rank Fusion to merge dense + sparse rankings
- Use fixed-size chunking with overlap — do not build a chunking-strategy comparison, it's out of scope
- **Checkpoint metric:** build a small labeled set (20-30 queries with manually judged "relevant chunk" ground truth) and compute recall@5 / precision for dense-only vs. hybrid — this gives you the real % improvement number

### Phase 3 — Evaluation Harness (Days 25-40) → earns Bullet 3
- Build a fixed test set: 20-30 questions spanning your paper corpus (mix of easy factual + multi-hop)
- Run RAGAS: faithfulness, answer relevancy, context precision
- Run this harness against **V1 baseline, post-Phase-1, and post-Phase-2** — this is what gives you the "regression benchmark caught X quality drops" claim, because you'll actually see it happen across versions
- Wire it into a simple GitHub Action that reruns the harness on every push to main — this is the one approved scope addition, ~2-3 days, small and provable
- Store results as a simple CSV/dashboard so you have a paper trail, not just a final number

### Phase 4 — Citation Verification + Production (Days 41-53) → earns Bullet 4
- Citation verifier: ask the LLM to tag each claim with the source chunk ID it came from, then verify that chunk actually supports the claim (second lightweight LLM call or basic embedding similarity check) — do NOT build token-level semantic span-matching, it's a separate NLP problem
- Wrap the full graph in FastAPI (`/query` endpoint)
- Dockerize, run locally — do NOT add Redis caching or cloud deployment (AWS/GCP); it adds an interview topic with no connection to the RAG/agent skills you're proving
- Log latency per request, compute median across your test set
- **Checkpoint metric:** run 20-30 queries through the final pipeline, measure % of claims with a verified source vs. an ungrounded baseline (V1 or synthesis-without-verification) — this is your final trust number

### Phase 5 — Buffer + Write-up (Days 54-60)
- Fix whatever's fragile
- Write the actual bullets using your real numbers
- Prepare a 2-minute verbal walkthrough of the architecture — you will be asked to explain this live, so rehearse the "why LangGraph, why hybrid search, why RAGAS" reasoning, not just the what

---

## Part 4 — Hard Rules to Keep This Defensible

1. **Every bullet number must come from a script you ran, not an estimate.** Keep the eval scripts in the repo — if asked "how did you get 35%," you should be able to open the file.
2. **Don't add GraphRAG, long-term memory, or multi-document comparison unless Phases 1-4 are fully done with real numbers by day 45.** They're not in any bullet above — they're V1-doc scope creep that dilutes a tight, provable story.
3. **If a phase metric comes back weak or flat, keep it — don't hide it.** "I found hybrid search only improved recall by 4% on my corpus, here's why I think that is" is a *stronger* interview answer than a suspiciously perfect number.
4. **Tag your own baseline commit right after Phase 1 is stable** (plain dense retrieval, before hybrid search or citation verification) — since you're building from scratch, this tagged commit is your only source of "before" numbers for every later comparison claim. Don't skip it.
5. **Depth over breadth is the whole game.** 2026 interview research is consistent: interviewers pick 3-5 topics from your project and drill into failure modes and trade-offs rather than scanning for feature count. Every scope item cut from this SOP (chunking experiments, span-matching, Redis, cloud deploy) was cut because it would add a shallow topic instead of deepening one you can already defend. If you're ever tempted to add a feature mid-build, ask: "can I go two follow-up questions deep on this?" — if not, it's not worth the build time.