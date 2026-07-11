# ThinkScope V2

Autonomous multi-agent research intelligence RAG system. Built from scratch (not forked from V1) as a placement project — see [SOP.md](./SOP.md) for the full build plan, target resume bullets, and phase-by-phase scope.

## Status

**Phase 0 — Setup.** Repo scaffolded, no pipeline code yet.

## Architecture (target)

```
Planner -> Retrieval Agent (hybrid dense + BM25) -> Critic (retry loop) -> Citation Verifier -> Answer Synthesis
```

Parallel: RAGAS eval harness, FastAPI + Docker service. Full details in [SOP.md](./SOP.md).

## Project layout

- `app/` — everything: application code (Planner, Retrieval, Critic, etc. — added phase by phase), `app/data/` (research paper corpus, gitignored, regenerate/download locally), `app/eval/` (eval question sets and RAGAS harness)

## Setup

```
uv sync
uv run python app/main.py
```

## Reminder

Per SOP Part 4 rule 4: once Phase 1 (Planner -> Retrieval -> Critic loop, plain dense retrieval) is stable, tag that commit as `v0-baseline` before starting Phase 2 (hybrid search). Do not skip this — it's the only source of "before" numbers for later comparisons.
