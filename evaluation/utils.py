"""Shared factories for embedding & reranking models (loaded lazily)."""

from functools import lru_cache

from evaluation.config import EMBEDDING_MODEL, RERANKER_MODEL


@lru_cache(maxsize=1)
def embeddings_factory():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def reranker_factory():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)