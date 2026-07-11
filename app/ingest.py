"""Pull an arXiv paper corpus, chunk it (fixed-size + overlap), and index into ChromaDB.

Chunking strategy is deliberately plain fixed-size-with-overlap, per SOP:
no semantic/parent-child chunking comparison, that's out of scope.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import arxiv
import chromadb
import requests
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from app import config

DOWNLOAD_WORKERS = 8

SEARCH_QUERIES = [
    "retrieval augmented generation",
    "large language model agents",
    "RAG evaluation faithfulness",
]
CATEGORIES = {"cs.CL", "cs.AI", "cs.LG"}
TARGET_PAPER_COUNT = 110


def fetch_papers(target_count: int = TARGET_PAPER_COUNT) -> list[arxiv.Result]:
    client = arxiv.Client(page_size=50, delay_seconds=3)
    seen_ids: set[str] = set()
    papers: list[arxiv.Result] = []

    for query in SEARCH_QUERIES:
        search = arxiv.Search(
            query=query,
            max_results=target_count,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            if not (CATEGORIES & set(result.categories)):
                continue
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)
            papers.append(result)
            if len(papers) >= target_count:
                return papers
    return papers


def download_and_extract_text(paper: arxiv.Result) -> str:
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    filename = paper.get_short_id().replace("/", "_") + ".pdf"
    pdf_path = config.PAPERS_DIR / filename

    if not pdf_path.exists():
        response = requests.get(paper.pdf_url, timeout=30)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_corpus(target_count: int = TARGET_PAPER_COUNT) -> None:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
    collection = chroma_client.get_or_create_collection(name="papers", embedding_function=embedder)

    papers = fetch_papers(target_count)
    print(f"Fetched {len(papers)} paper records from arXiv")

    texts: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        future_to_paper = {pool.submit(download_and_extract_text, p): p for p in papers}
        done = 0
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            paper_id = paper.get_short_id()
            done += 1
            try:
                texts[paper_id] = future.result()
                print(f"[{done}/{len(papers)}] downloaded {paper_id} - {paper.title[:60]}")
            except Exception as exc:
                print(f"[{done}/{len(papers)}] skip {paper_id}: {exc}")

    for paper in papers:
        paper_id = paper.get_short_id()
        text = texts.get(paper_id)
        if not text:
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"skip {paper_id}: no extractable text")
            continue

        ids = [f"{paper_id}_chunk{j}" for j in range(len(chunks))]
        metadatas = [
            {"paper_id": paper_id, "title": paper.title, "chunk_index": j}
            for j in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        print(f"indexed {paper_id} ({len(chunks)} chunks) - {paper.title[:60]}")

    print(f"Done. Collection now has {collection.count()} chunks.")


if __name__ == "__main__":
    build_corpus()
