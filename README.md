# HW4 – Retrieval Augmented Generation (RAG) Pipeline

## Track Declaration

**Track: A – Text RAG**

### Implementation Choice: MiniLM Alternative Path
Per the assignment specification, I implemented the simpler alternative path: using `sentence-transformers/all-MiniLM-L6-v2` for both the starter corpus embedding and new document embedding. This avoids external API keys while maintaining full reproducibility on CPU. The starter corpus is loaded from `wikimedia/wikipedia` (Simple English, ~10,000 passages) and embedded locally, then extended with 5 supplementary local documents, all in a consistent 384-dimensional vector space.

## Overview

The project implements a **Retrieval Augmented Generation (RAG) system** that combines information retrieval with grounded text generation.

The pipeline consists of:

1. **Corpus Ingestion**
   - **Primary corpus**: Simple English Wikipedia dataset (`wikimedia/wikipedia`, config: `20231101.simple`)  
   - **Secondary corpus**: Local Wikipedia-style documents stored in `data/new_docs/`
   - **Embedding Strategy**: All documents (starter and new) embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for consistency
   
2. **Embedding & Storage**
   - All documents embedded locally using MiniLM on CPU
   - Both stored in Chroma vector database with metadata (title, URL, source)
   - Consistent 384-dim vector space across both collections

3. **Retrieval & Generation**
   - Query embedding via `sentence-transformers/all-MiniLM-L6-v2`
   - Top-k retrieval (k=4) from Chroma collections
   - Grounded answer generation via LLM API (with extractive fallback)
   - Inline citations with `[Title](URL)` format

## Datasets & Models Used

### Datasets
- `wikimedia/wikipedia` (config: `20231101.simple`) – Simple English Wikipedia corpus (10,000 passages)
- Local supplementary documents in `data/new_docs/` (5 files):
  - doc1.txt: Retrieval Augmented Generation
  - doc2.txt: Cosine Similarity
  - doc3.txt: Sentence Embeddings
  - doc4.txt: Vector Database
  - doc5.txt: Simple English Wikipedia

### Models
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, used for all documents)
- **LLM Model**: `meta-llama/Llama-3.1-8B-Instruct` (OpenAI-compatible API at UTSA_MODEL)

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
│   ├── ingest_wiki.py       (load Wikipedia dataset and embed with MiniLM)
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

### 2. Ingest the starter corpus (Wikipedia dataset with local MiniLM embedding)
```bash
python scripts/ingest_wiki.py --max-docs 10000
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

- The LLM API at UTSA endpoint provides access to Llama-3.1-8B-Instruct; if unavailable, the extractive fallback provides grounded answers from top-k retrieved passages.
- All configuration loaded from `config.py` and `.env` (environment variables for API credentials)
- Embedding dimensions: All documents embedded with MiniLM at 384-dim for consistency.
- Chroma is configured for cosine similarity (hnsw:space = "cosine").
