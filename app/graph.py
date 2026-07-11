"""LangGraph Retrieval <-> Critic loop for a single sub-question.

On critic failure the query is reformulated and retrieval retries,
up to MAX_RETRIEVAL_RETRIES times, then falls through with best-effort chunks.
Never loops forever.
"""

import json
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app import config, critic
from app.llm import get_client
from app.retrieval import retrieve_dense


class LoopState(TypedDict):
    sub_question: str
    search_query: str
    chunks: list[dict]
    resolved: bool
    reason: str
    retries: int


def _retrieve_node(state: LoopState) -> LoopState:
    t0 = time.time()
    print(f"    [retrieve] start query={state['search_query'][:40]!r}", flush=True)
    chunks = retrieve_dense(state["search_query"])
    print(f"    [retrieve] done in {time.time()-t0:.2f}s, {len(chunks)} chunks", flush=True)
    return {**state, "chunks": chunks}


def _critic_node(state: LoopState) -> LoopState:
    t0 = time.time()
    print("    [critic] start", flush=True)
    client = get_client()
    try:
        verdict = critic.critique(client, state["sub_question"], state["chunks"])
    except (json.JSONDecodeError, KeyError):
        verdict = {"sufficient": False, "reason": "critic returned unparseable output", "reformulated_query": state["search_query"]}
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
