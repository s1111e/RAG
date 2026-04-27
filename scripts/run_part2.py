import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ARTIFACTS_DIR, NEW_COLLECTION, TOP_K, WIKI_COLLECTION
from src.generator import answer_query
from src.retriever import retrieve
from src.utils import ensure_directory, first_sentences, write_json, write_text


TARGETED_QUERIES = [
	"What is retrieval augmented generation?",
	"Why is cosine similarity useful in vector search?",
	"What are sentence embeddings?",
	"How does a vector database store embeddings?",
	"What is Chroma used for?",
]


CROSS_CORPUS_QUERIES = [
	"How can Wikipedia pages and retrieval augmented generation work together?",
	"What is the difference between cosine similarity and encyclopedia search?",
	"How do sentence embeddings help retrieve Wikipedia articles?",
	"When is a vector database more useful than a general encyclopedia?",
	"How could local documents and Wikipedia sources answer the same question?",
]


def _grounding_judgment(answer, contexts):
	if not contexts:
		return "ungrounded"
	if contexts and any(c["title"].lower() in answer.lower() for c in contexts):
		return "grounded"
	return "needs review"


def _choose_best_collection(query):
	wiki_contexts = retrieve(query, top_k=TOP_K, collection_name=WIKI_COLLECTION)
	new_contexts = retrieve(query, top_k=TOP_K, collection_name=NEW_COLLECTION)

	wiki_score = wiki_contexts[0]["similarity"] if wiki_contexts else float("-inf")
	new_score = new_contexts[0]["similarity"] if new_contexts else float("-inf")

	if new_score > wiki_score + 0.05:
		return NEW_COLLECTION, new_contexts, wiki_contexts
	return WIKI_COLLECTION, wiki_contexts, new_contexts


def _format_table(rows):
	header = "| Mode | Query | Chosen Collection | Wiki Top Source | New Docs Top Source | Answer First Two Sentences | Grounding |"
	separator = "| --- | --- | --- | --- | --- | --- | --- |"
	lines = [header, separator]
	for row in rows:
		lines.append(
			"| {mode} | {query} | {chosen} | {wiki_source} | {new_source} | {answer} | {grounding} |".format(
				mode=row["mode"],
				query=row["query"].replace("|", "\\|"),
				chosen=row["chosen_collection"],
				wiki_source=row["wiki_top_source"].replace("|", "\\|"),
				new_source=row["new_top_source"].replace("|", "\\|"),
				answer=row["answer_first_two_sentences"].replace("|", "\\|"),
				grounding=row["grounding_judgment"],
			)
		)
	return "\n".join(lines)


def _summarize_result(query, mode, chosen_collection, chosen_contexts, other_contexts):
	answer_data = answer_query(query, collection_name=chosen_collection, top_k=TOP_K)
	answer = answer_data["answer"]
	answer_body = answer.split("\n\nSources:", 1)[0]
	return {
		"mode": mode,
		"query": query,
		"chosen_collection": chosen_collection,
		"wiki_top_source": (
			other_contexts[0]["title"]
			if chosen_collection == NEW_COLLECTION and other_contexts
			else chosen_contexts[0]["title"] if chosen_contexts else "n/a"
		),
		"new_top_source": (
			chosen_contexts[0]["title"]
			if chosen_collection == NEW_COLLECTION and chosen_contexts
			else other_contexts[0]["title"] if other_contexts else "n/a"
		),
		"answer_first_two_sentences": first_sentences(answer_body, 2),
		"grounding_judgment": _grounding_judgment(answer, answer_data["contexts"]),
	}


def main():
	ensure_directory(ARTIFACTS_DIR)

	rows = []

	for query in TARGETED_QUERIES:
		chosen_collection = NEW_COLLECTION
		chosen_contexts = retrieve(query, top_k=TOP_K, collection_name=chosen_collection)
		other_contexts = retrieve(query, top_k=TOP_K, collection_name=WIKI_COLLECTION)
		rows.append(_summarize_result(query, "targeted", chosen_collection, chosen_contexts, other_contexts))

	for query in CROSS_CORPUS_QUERIES:
		chosen_collection, chosen_contexts, other_contexts = _choose_best_collection(query)
		rows.append(_summarize_result(query, "cross-corpus", chosen_collection, chosen_contexts, other_contexts))

	payload = {"queries": rows}
	write_json(Path(ARTIFACTS_DIR) / "part2_results.json", payload)
	write_text(Path(ARTIFACTS_DIR) / "part2_results.md", _format_table(rows))

	print(_format_table(rows))


if __name__ == "__main__":
	main()
