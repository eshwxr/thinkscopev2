"""Sparse (BM25) index over the same chunk corpus stored in ChromaDB.

Built once per process from the persisted collection's documents,
kept in memory (rank-bm25 is a plain in-memory index, no separate store).
"""

import re

from rank_bm25 import BM25Okapi

from app.retrieval import get_collection

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self._bm25 = BM25Okapi([tokenize(doc) for doc in documents])

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self.ids[i], "text": self.documents[i], "metadata": self.metadatas[i], "score": float(scores[i])}
            for i in ranked
        ]


_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        collection = get_collection()
        data = collection.get(include=["documents", "metadatas"])
        _index = BM25Index(data["ids"], data["documents"], data["metadatas"])
    return _index
