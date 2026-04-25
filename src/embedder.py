import os

from config import EMBED_MODEL_NAME

for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
    env_value = os.environ.get(env_name)
    if env_value and not os.path.exists(env_value):
        os.environ.pop(env_name, None)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from sklearn.feature_extraction.text import HashingVectorizer

_model = None
_hash_vectorizer = HashingVectorizer(
    n_features=384,
    alternate_sign=False,
    norm="l2",
)

def get_model():
    global _model
    if SentenceTransformer is None:
        return None
    if _model is None:
        try:
            _model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception:
            return None
    return _model

def embed_texts(texts):
    model = get_model()
    if model is not None:
        return model.encode(texts).tolist()

    # Fallback path when transformer dependencies are unavailable.
    return _hash_vectorizer.transform(texts).toarray().tolist()

def embed_query(query):
    return embed_texts([query])[0]