# HW4 – Retrieval Augmented Generation (RAG) Pipeline

## Track: Track A

This project implements a **Retrieval Augmented Generation (RAG) system** that combines information retrieval with grounded text generation.

## Overview

The pipeline consists of:

1. **Corpus Ingestion**
   - **Primary corpus**: Timescale pre-embedded Simple English Wikipedia dataset (`timescale/wikipedia-22-12-simple-embeddings`)
   - **Secondary corpus**: Local Wikipedia-style documents stored in `data/new_docs/`
   
2. **Embedding & Storage**
   - Embeddings from Timescale dataset are ingested as-is (pre-computed)
   - Local documents are embedded using `sentence-transformers/all-MiniLM-L6-v2`
   - Both stored in Chroma vector database with metadata (title, URL, source)

3. **Retrieval & Generation**
   - Query embedding via `sentence-transformers/all-MiniLM-L6-v2`
   - Top-k retrieval (k=4) from Chroma collections
   - Grounded answer generation via LLM API (with extractive fallback)
   - Inline citations with `[Title](URL)` format

## Datasets & Models Used

### Datasets
- `timescale/wikipedia-22-12-simple-embeddings` – Pre-embedded Simple English Wikipedia corpus
- Local supplementary documents in `data/new_docs/` (5 files):
  - doc1.txt: Retrieval Augmented Generation
  - doc2.txt: Cosine Similarity
  - doc3.txt: Sentence Embeddings
  - doc4.txt: Vector Database
  - doc5.txt: Simple English Wikipedia

### Models
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **LLM Model**: `llama-3.3-70b-instruct-awq` (OpenAI-compatible API at `http://10.246.100.230/v1`)

## Project Structure

```
HW4/
├── README.md                 (this file)
├── REPORT.md                 (results & reflection)
├── config.py                 (configuration: models, paths, API settings)
├── requirements.txt          (Python dependencies)
├── src/
│   ├── embedder.py          (embedding utility with fallback)
│   ├── retriever.py         (Chroma-based retrieval)
│   ├── generator.py         (grounded generation with citations)
│   ├── llm_client.py        (LLM API wrapper)
│   └── utils.py             (helpers: whitespace, sentences, file I/O)
├── scripts/
│   ├── ingest_wiki.py       (ingest Timescale dataset into wiki collection)
│   ├── ingest_new_docs.py   (ingest local .txt files into new_docs collection)
│   ├── run_part1.py         (execute 10 starter queries, generate Part1 results)
│   ├── run_part2.py         (execute targeted + cross-corpus queries, generate Part2 results)
│   └── test_llm.py          (test LLM API connectivity)
├── data/
│   └── new_docs/            (5 local Wikipedia-style documents)
├── chroma_db/               (persistent Chroma vector store)
└── artifacts/               (Part1 & Part2 results in JSON and Markdown)
```

## Running the Pipeline

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest the starter corpus (Timescale Wikipedia)
```bash
python scripts/ingest_wiki.py --max-docs 3000
```

### 3. Ingest local supplementary documents
```bash
python scripts/ingest_new_docs.py
```

### 4. Run Part 1 benchmark (10 starter-corpus queries)
```bash
python scripts/run_part1.py
```
Output: `artifacts/part1_results.md` and `artifacts/part1_results.json`

### 5. Run Part 2 benchmark (5 targeted + 5 cross-corpus queries)
```bash
python scripts/run_part2.py
```
Output: `artifacts/part2_results.md` and `artifacts/part2_results.json`

## Key Features

- **Robust Embedding**: Fallback from transformer to hashing vectorizer if model loading fails
- **Graceful LLM Fallback**: If LLM API is unavailable, extractive summarization is used
- **Citation-Aware Generation**: Answers include inline citations `[Title](URL)`
- **Two-Collection Retrieval**: Supports comparison between Wikipedia and supplementary corpora
- **Comprehensive Metadata**: Every document retains title, URL, and source for traceability

## Notes

- The LLM API at `http://10.246.100.230/v1` may not always be available; in such cases, the extractive fallback provides grounded answers from the top-k retrieved passages.
- Embedding dimensions: Timescale dataset vectors are ~768-dim (Cohere embeddings); MiniLM vectors are 384-dim. Chroma handles mixed dimensions.
- Chroma is configured for cosine similarity (hnsw:space = "cosine").
