import argparse

from app.generation.answer_generator import AnswerGenerator
from app.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = Retriever()
    chunks = retriever.retrieve(
        question=args.question,
        top_k=args.top_k,
    )

    generator = AnswerGenerator()
    answer = generator.generate_answer(
        question=args.question,
        chunks=chunks,
    )

    print("")
    print("QUESTION")
    print(args.question)
    print("")
    print("ANSWER")
    print(answer)
    print("")
    print("CITATIONS")

    for index, chunk in enumerate(chunks, start=1):
        print("-" * 80)
        print(f"[{index}] Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Lexical overlap count: {chunk.get('lexical_overlap_count', 0)}")
        print(f"Lexical overlap terms: {chunk.get('lexical_overlap_terms', [])}")
        print(chunk["text"][:400])


if __name__ == "__main__":
    main()