import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generator import answer_query


def main():
    query = "Artificial intelligence learning methods"

    print("Query:", query)
    print("\nRunning generator...\n")

    result = answer_query(query)

    print("ANSWER:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()