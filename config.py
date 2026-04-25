# Embedding model (LOCAL)
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
LLM_API_URL = "http://10.100.1.213:8888/v1"
LLM_API_KEY = "utsa-08GdYYyq2lzmWc02fhfMSKzv3ACPwYgq6U02BozaaupZym1wGQzJBNC59dV4wFTi"
LLM_MODEL = "Qwen/Qwen3.5-27B"

# Generation
MAX_TOKENS = 300
TEMPERATURE = 0.2