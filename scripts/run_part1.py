import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ARTIFACTS_DIR, TOP_K, WIKI_COLLECTION
from src.generator import answer_query
from src.retriever import retrieve
from src.utils import ensure_directory, first_sentences, truncate_text, write_json, write_text


STARTER_QUERIES = [
	"What is the Moon?",
	"What is photosynthesis?",
	"Who was Albert Einstein?",
	"What is a volcano?",
	"What is an algorithm?",
	"What is the Taj Mahal?",
	"What is a rainforest?",
	"What is the Internet?",
	"What is a gene?",
	"What is the United Nations?",
]


def _grounding_judgment(answer, contexts):
	if not contexts:
		return "ungrounded"
	lowered_answer = answer.lower()
	cited_titles = [context.get("title", "").lower() for context in contexts[: min(3, len(contexts))]]
	if any(title and title in lowered_answer for title in cited_titles):
		return "grounded"
	if "sources:" in lowered_answer:
		return "mostly grounded"
	return "needs review"


def _format_table(rows):
	header = "| Query | Top Source | Top Score | Answer First Two Sentences | Grounding |"
	separator = "| --- | --- | --- | --- | --- |"
	lines = [header, separator]
	for row in rows:
		lines.append(
			"| {query} | {source} | {score} | {answer} | {grounding} |".format(
				query=row["query"].replace("|", "\\|"),
				source=row["top_source"].replace("|", "\\|"),
				score=row["top_score"],
				answer=row["answer_first_two_sentences"].replace("|", "\\|"),
				grounding=row["grounding_judgment"],
			)
		)
	return "\n".join(lines)


def main():
	ensure_directory(ARTIFACTS_DIR)

	rows = []
	for query in STARTER_QUERIES:
		retrieved = retrieve(query, top_k=TOP_K, collection_name=WIKI_COLLECTION)
		result = answer_query(query, collection_name=WIKI_COLLECTION, top_k=TOP_K)
		contexts = result["contexts"]
		answer = result["answer"]
		top = retrieved[0] if retrieved else {}
		answer_body = answer.split("\n\nSources:", 1)[0]
		rows.append(
			{
				"query": query,
				"top_source": f"{top.get('title', 'n/a')} - {top.get('url', '')} (score={round(top.get('similarity', 0), 4) if top.get('similarity') is not None else 'n/a'})",				"top_score": round(top.get("similarity", 0), 4) if top.get("similarity") is not None else "n/a",
				"answer_first_two_sentences": first_sentences(answer_body, 2),
				"grounding_judgment": _grounding_judgment(answer, contexts),
			}
		)

	payload = {"queries": rows}
	write_json(Path(ARTIFACTS_DIR) / "part1_results.json", payload)
	write_text(Path(ARTIFACTS_DIR) / "part1_results.md", _format_table(rows))

	print(_format_table(rows))


if __name__ == "__main__":
	main()
