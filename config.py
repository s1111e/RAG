# Embedding model (LOCAL)
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed; use os.getenv directly

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Vector DB
CHROMA_PATH = "./chroma_db"

# Corpus and outputs
WIKI_MAX_DOCS = 10000
NEW_DOCS_DIR = "./data/new_docs"
ARTIFACTS_DIR = "./artifacts"

# Collections
WIKI_COLLECTION = "wiki"
NEW_COLLECTION = "new_docs"

# Retrieval
TOP_K = 4

# LLM API (UTSA)
# Set these via environment variables: LLM_API_URL, LLM_API_KEY, LLM_MODEL
LLM_API_URL = os.getenv("LLM_API_URL", "http://149.165.173.247:8888/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY_HERE")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Generation
MAX_TOKENS = 300
TEMPERATURE = 0.2