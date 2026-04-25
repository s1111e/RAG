import chromadb

from config import CHROMA_PATH, TOP_K, WIKI_COLLECTION
from src.embedder import embed_query


def retrieve(query, top_k=TOP_K, collection_name=WIKI_COLLECTION):
	client = chromadb.PersistentClient(path=CHROMA_PATH)

	try:
		collection = client.get_collection(collection_name)
	except Exception:
		return []

	results = collection.query(
		query_texts=[query],
		n_results=top_k,
		include=["documents", "metadatas", "distances"],
	)

	items = []
	ids = results.get("ids", [[]])[0] if results else []
	documents = results.get("documents", [[]])[0] if results else []
	metadatas = results.get("metadatas", [[]])[0] if results else []
	distances = results.get("distances", [[]])[0] if results else []

	for index, doc in enumerate(documents):

		query_words = query.lower().split()

		if not any(word in doc.lower() for word in query_words):
			continue

		metadata = metadatas[index] if index < len(metadatas) else {}

		metadata = metadatas[index] if index < len(metadatas) else {}
		distance = float(distances[index]) if index < len(distances) else None
		similarity = None if distance is None else round(1.0 - distance, 4)
		items.append(
			{
				"id": ids[index] if index < len(ids) else None,
				"title": metadata.get("title", "unknown"),
				"url": metadata.get("url", ""),
				"text": doc,
				"distance": distance,
				"similarity": similarity,
				"metadata": metadata,
			}
		)

	return items
