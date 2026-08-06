from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

@lru_cache(maxsize=1)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embedding_model()
