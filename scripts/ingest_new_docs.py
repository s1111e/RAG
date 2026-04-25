import argparse
import os
import sys
from pathlib import Path

import chromadb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CHROMA_PATH, NEW_COLLECTION, NEW_DOCS_DIR
from src.embedder import embed_texts
from src.utils import clean_whitespace, ensure_directory


def _reset_collection(client, collection_name):
	try:
		client.delete_collection(collection_name)
	except Exception:
		pass


def load_new_docs(directory=NEW_DOCS_DIR):
	docs_dir = Path(directory)
	rows = []
	for file_path in sorted(docs_dir.glob("*.txt")):
		text = clean_whitespace(file_path.read_text(encoding="utf-8"))
		if len(text) < 50:
			continue
		resolved_path = file_path.resolve()
		rows.append(
			{
				"title": file_path.stem.replace("_", " ").title(),
				"text": text,
				"url": resolved_path.as_uri(),
				"source": "local_new_docs",
			}
		)
	return rows


def main(batch_size=32):
	client = chromadb.PersistentClient(path=CHROMA_PATH)
	_reset_collection(client, NEW_COLLECTION)
	collection = client.get_or_create_collection(
		name=NEW_COLLECTION,
		metadata={"hnsw:space": "cosine"},
	)

	rows = load_new_docs()
	if not rows:
		raise RuntimeError(f"No .txt files with content found under {NEW_DOCS_DIR}.")

	texts = []
	metadatas = []
	ids = []

	for index, row in enumerate(rows):
		texts.append(row["text"])
		metadatas.append(
			{
				"title": row["title"],
				"url": row["url"],
				"source": row["source"],
			}
		)
		ids.append(f"new-{index}")

	for start in range(0, len(texts), batch_size):
		end = start + batch_size
		batch_texts = texts[start:end]
		batch_embeddings = embed_texts(batch_texts)
		collection.add(
			ids=ids[start:end],
			embeddings=batch_embeddings,
			documents=batch_texts,
			metadatas=metadatas[start:end],
		)

	print(f"DONE: ingested {len(texts)} local documents into {NEW_COLLECTION}.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Ingest local new-doc .txt files into Chroma.")
	parser.add_argument("--batch-size", type=int, default=32)
	args = parser.parse_args()
	ensure_directory(CHROMA_PATH)
	main(batch_size=args.batch_size)
