import chromadb
from chromadb.utils import embedding_functions

from app import config

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
        _collection = client.get_collection(name="papers", embedding_function=embedder)
    return _collection


def retrieve_dense(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    """Plain dense-vector search. This is the Phase 1 baseline retrieval method,
    kept as-is (not modified) so Phase 2's hybrid comparison has a stable reference."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    for doc, metadata, distance, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        chunks.append({"id": chunk_id, "text": doc, "metadata": metadata, "distance": distance})
    return chunks
