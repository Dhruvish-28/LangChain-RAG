"""Loads the evaluation corpus and chunks it for a given configuration."""

import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from evaluation.config import CORPUS_PATH


def load_corpus_documents():
    """Return the corpus documents as list of dicts."""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["documents"]


def chunk_corpus(cfg):
    """
    Split every corpus document into LangChain Documents using the same
    splitter strategy as the app (RecursiveCharacterTextSplitter) but with
    the benchmark parameters from cfg.

    Each chunk carries metadata:
      - source   : file name (like the app)
      - doc_id   : corpus document id
      - chunk_id : 0-based index within that document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    all_chunks = []

    for doc in load_corpus_documents():
        pieces = splitter.split_text(doc["content"])
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": doc["path"],
                        "doc_id": doc["id"],
                        "chunk_id": i,
                    },
                )
            )

    return all_chunks


def docs_to_text(chunks):
    """Helper to convert chunk list to plain text list (for BM25)."""
    return [c.page_content for c in chunks]