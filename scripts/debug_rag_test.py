import argparse
import os
import sys
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import get_rag_index


DEFAULT_QUERIES = [
    "Ossigenoterapia mercoledì alle 11:00",
    "Camomilla mercoledì di sera",
    "Diabete di tipo 1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick RAG retrieval smoke test.")
    parser.add_argument("--query", action="append", help="Query to run (repeatable).")
    parser.add_argument("--top-k", type=int, default=3, help="Top K results to show.")
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES

    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    index = get_rag_index()
    retriever = index.as_retriever(similarity_top_k=args.top_k)

    for query in queries:
        print(f"\n=== QUERY: {query} ===")
        results = retriever.retrieve(query)
        if not results:
            print("No results.")
            continue
        for i, res in enumerate(results, 1):
            text = res.node.text.replace("\n", " ").strip()
            meta = res.node.metadata or {}
            meta_bits = []
            for key in ("type", "source", "patient_id", "caregiver_id", "activity_id", "category"):
                if key in meta:
                    meta_bits.append(f"{key}={meta[key]}")
            meta_str = f" | meta: {', '.join(meta_bits)}" if meta_bits else ""
            print(f"- ({i}) {text}{meta_str}")


if __name__ == "__main__":
    main()
