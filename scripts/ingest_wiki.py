import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
import chromadb

from src.embedder import embed_texts
from config import CHROMA_PATH, WIKI_COLLECTION, WIKI_MAX_DOCS


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # reset
    try:
        client.delete_collection(WIKI_COLLECTION)
    except:
        pass

    collection = client.get_or_create_collection(WIKI_COLLECTION)

    print("Loading Wikipedia dataset...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)

    texts = []
    metadatas = []
    ids = []

    print("Collecting data...")

    for i, row in enumerate(ds):
        if i >= WIKI_MAX_DOCS:
            break

        text = row["text"]

        if not text or len(text.strip()) < 50:
            continue

        texts.append(text)
        metadatas.append({
            "title": row.get("title", "unknown"),
            "url": ""
        })
        ids.append(f"id_{i}")

    print(f"Collected {len(texts)} documents")

    print("Embedding with MiniLM...")
    embeddings = embed_texts(texts)

    print("Adding to Chroma...")
    BATCH_SIZE = 500

    print("Adding to Chroma in batches...")

    for i in range(0, len(texts), BATCH_SIZE):
        collection.add(
            ids=ids[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE],
            documents=texts[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
        )

        print(f"Inserted {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    print("DONE: Wiki ingested!")


if __name__ == "__main__":
    main()