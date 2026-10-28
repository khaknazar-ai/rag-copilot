import argparse

from app.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = Retriever()
    results = retriever.retrieve(
        question=args.question,
        top_k=args.top_k,
    )

    print(f"Question: {args.question}")
    print(f"Results: {len(results)}")
    print("")

    for index, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Lexical overlap count: {result.get('lexical_overlap_count', 0)}")
        print(f"Lexical overlap terms: {result.get('lexical_overlap_terms', [])}")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("")
        print(result["text"][:600])
        print("")


if __name__ == "__main__":
    main()