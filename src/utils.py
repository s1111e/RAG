import json
import os
import re
from pathlib import Path


def ensure_directory(path):
	Path(path).mkdir(parents=True, exist_ok=True)


def clean_whitespace(text):
	return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text):
	text = clean_whitespace(text)
	if not text:
		return []
	parts = re.split(r"(?<=[.!?])\s+", text)
	return [part.strip() for part in parts if part.strip()]


def first_sentences(text, count=2):
	sentences = split_sentences(text)
	if not sentences:
		return ""
	return " ".join(sentences[:count])


def truncate_text(text, max_chars=280):
	text = clean_whitespace(text)
	if len(text) <= max_chars:
		return text
	return text[: max_chars - 1].rstrip() + "…"


def make_source_label(metadata):
	title = (metadata or {}).get("title") or "unknown"
	url = (metadata or {}).get("url") or ""
	if url:
		return f"[{title}]({url})"
	return title


def write_json(path, payload):
	ensure_directory(Path(path).parent)
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_text(path, content):
	ensure_directory(Path(path).parent)
	with open(path, "w", encoding="utf-8") as handle:
		handle.write(content)
