"""LangGraph Retrieval <-> Critic loop for a single sub-question.

On critic failure the query is reformulated and retrieval retries,
up to MAX_RETRIEVAL_RETRIES times, then falls through with best-effort chunks.
Never loops forever.

Chunks accumulate across retries (deduplicated by chunk id, capped) instead
of being replaced each attempt -- a reformulated query finding 2 new useful
chunks used to throw away everything found on the previous attempt.
"""

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app import config, critic
from app.hybrid_retrieval import retrieve_hybrid
from app.llm import get_client
from app.retrieval import retrieve_dense

MAX_ACCUMULATED_CHUNKS = 12


class LoopState(TypedDict):
    sub_question: str
    search_query: str
    chunks: list[dict]
    resolved: bool
    reason: str
    retries: int


def _retrieve(query: str) -> list[dict]:
    if config.RETRIEVAL_MODE == "dense":
        return retrieve_dense(query)
    return retrieve_hybrid(query)


def _retrieve_node(state: LoopState) -> LoopState:
    t0 = time.time()
    print(f"    [retrieve:{config.RETRIEVAL_MODE}] start query={state['search_query'][:40]!r}", flush=True)
    new_chunks = _retrieve(state["search_query"])

    seen_ids = {c["id"] for c in state["chunks"]}
    merged = list(state["chunks"])
    for c in new_chunks:
        if c["id"] not in seen_ids:
            merged.append(c)
            seen_ids.add(c["id"])
    merged = merged[:MAX_ACCUMULATED_CHUNKS]

    print(f"    [retrieve] done in {time.time()-t0:.2f}s, {len(new_chunks)} new, {len(merged)} accumulated", flush=True)
    return {**state, "chunks": merged}


def _critic_node(state: LoopState) -> LoopState:
    t0 = time.time()
    print("    [critic] start", flush=True)
    client = get_client()
    try:
        verdict = critic.critique(client, state["sub_question"], state["chunks"])
    except Exception as exc:
        # Any critic failure (unparseable JSON, exhausted rate-limit retries,
        # network error) degrades to "insufficient" so one bad LLM call fails
        # a single sub-question instead of crashing the whole eval run.
        verdict = {
            "sufficient": False,
            "reason": f"critic call failed: {type(exc).__name__}: {exc}",
            "reformulated_query": state["search_query"],
        }
    print(f"    [critic] done in {time.time()-t0:.2f}s, sufficient={verdict.get('sufficient')}", flush=True)
    return {
        **state,
        "resolved": verdict.get("sufficient", False),
        "reason": verdict.get("reason", ""),
        "search_query": verdict.get("reformulated_query", state["search_query"]),
    }


def _should_retry(state: LoopState) -> str:
    if state["resolved"]:
        return "end"
    if state["retries"] >= config.MAX_RETRIEVAL_RETRIES:
        return "end"
    return "retry"


def _increment_retries(state: LoopState) -> LoopState:
    return {**state, "retries": state["retries"] + 1}


def build_graph():
    graph = StateGraph(LoopState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("increment", _increment_retries)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "critic")
    graph.add_conditional_edges("critic", _should_retry, {"retry": "increment", "end": END})
    graph.add_edge("increment", "retrieve")

    return graph.compile()


_compiled_graph = None


def run_retrieval_critic_loop(sub_question: str) -> LoopState:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: LoopState = {
        "sub_question": sub_question,
        "search_query": sub_question,
        "chunks": [],
        "resolved": False,
        "reason": "",
        "retries": 0,
    }
    return _compiled_graph.invoke(initial_state)
