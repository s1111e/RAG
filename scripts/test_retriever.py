import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import retrieve


def main():
    query = "How do computers learn from data?"
    
    print("Query:", query)
    print("\nRunning retriever...\n")

    try:
        results = retrieve(query)

        print(f"Number of results: {len(results)}\n")

        for i, r in enumerate(results):
            print(f"--- Result {i+1} ---")
            print("Title:", r["title"])
            print("Similarity:", r["similarity"])
            print("Text preview:", r["text"][:150])
            print()

    except Exception as e:
        print("ERROR OCCURRED:")
        print(e)


if __name__ == "__main__":
    main()
