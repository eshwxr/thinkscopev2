import sys

from app.pipeline import answer_query


def main():
    query = " ".join(sys.argv[1:]) or "What techniques improve retrieval precision in RAG systems?"

    result = answer_query(query)
    print(f"Query: {result['query']}\n")
    for r in result["results"]:
        status = "RESOLVED" if r["resolved"] else "UNRESOLVED"
        print(f"[{status}] sub-question: {r['sub_question']}")
        print(f"  retries used: {r['retries_used']}, reason: {r['reason']}")
        for c in r["chunks"][:2]:
            print(f"  - {c['metadata']['title'][:60]} (chunk {c['metadata']['chunk_index']})")
        print()


if __name__ == "__main__":
    main()
