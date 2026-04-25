from config import TOP_K, WIKI_COLLECTION
from src.llm_client import call_llm
from src.retriever import retrieve
from src.utils import first_sentences, make_source_label, truncate_text


SYSTEM_PROMPT = (
    "You are a strict RAG assistant. "
    "Answer ONLY using the provided context. "
    "If the answer is not clearly in the context, say 'I don't know'. "
    "Do NOT add any external knowledge. "
    "Write 2-3 sentences maximum. "
    "Cite every sentence using [Title](URL)."
)


def _build_sources_block(contexts):
	lines = []
	for index, context in enumerate(contexts, start=1):
		label = make_source_label({
			"title": context.get("title"),
			"url": context.get("url")
		})
		excerpt = truncate_text(context.get("text", ""), 120)
		lines.append(f"[{index}] {label}: {excerpt}")
	return "\n".join(lines)


def _build_fallback_answer(contexts):
	parts = []
	for context in contexts[:2]:
		excerpt = first_sentences(context.get("text", ""), 2)
		label = make_source_label({
			"title": context.get("title"),
			"url": context.get("url")
		})
		if excerpt:
			parts.append(f"{excerpt} {label}.")
	if not parts:
		return "I could not find enough supporting context to answer confidently."
	return " ".join(parts)


def answer_query(query, collection_name=WIKI_COLLECTION, top_k=TOP_K):
	contexts = retrieve(query, top_k=top_k, collection_name=collection_name)
	contexts = contexts[:2]
	source_block = _build_sources_block(contexts)

	user_prompt = (
		f"Question: {query}\n\n"
		"Use ONLY the sources below. "
		"If the answer is not clearly about the question, say 'I don't know'. "
		"Do NOT answer using unrelated sources."
		f"Sources:\n{source_block}"
	)

	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": user_prompt},
	]

	response = call_llm(messages)
	if not response or response == "Error":
		response = _build_fallback_answer(contexts)

	response = response.strip()
	if contexts and "[" not in response:
		citations = "; ".join(
			make_source_label({
				"title": context.get("title"),
				"url": context.get("url")
			}) for context in contexts[: min(3, len(contexts))]
		)
		response = f"{response}\n\nSources: {citations}"

	return {
		"query": query,
		"answer": response,
		"contexts": contexts,
	}
