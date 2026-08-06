"""
Parametrized retrieval pipeline used for benchmarking.

Mirrors the app's retrieval strategy (retrieval/ensemble_retriever.py +
retrieval/retriever.py) but parameterized by Config so we can compare
different retriever types, candidate pools, reranking, and chunking.

Every stage records its own wall-clock latency so the benchmark can report
vector search time, BM25 time, ensemble time, rerank time, and total
retrieval time separately.
"""

import time
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from evaluation.config import Config
from evaluation.core.corpus import chunk_corpus
from evaluation.utils import embeddings_factory, reranker_factory


class TimingResult:
    """Collects per-stage latency for a single query-retrieval call."""

    def __init__(self):
        self.index_build_ms = 0.0
        self.vector_search_ms = 0.0
        self.bm25_search_ms = 0.0
        self.ensemble_ms = 0.0
        self.rerank_ms = 0.0
        self.total_retrieval_ms = 0.0


class RetrievalEngine:
    """Builds index once per Config, then answers queries with instrumentation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chunks = chunk_corpus(cfg)
        self._vectorstore = None
        self._bm25 = None

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------
    def build_index(self, timing: TimingResult):
        """Build the FAISS vector store from the chunked corpus."""
        start = time.perf_counter()
        self._vectorstore = FAISS.from_documents(self.chunks, embeddings_factory())
        timing.index_build_ms = (time.perf_counter() - start) * 1000

    def _ensure_vectorstore(self, timing: TimingResult):
        if self._vectorstore is None:
            self.build_index(timing)
        return self._vectorstore

    def _ensure_bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Retriever.from_documents(self.chunks)
            self._bm25.k = self.cfg.ensemble_k
        return self._bm25

    # ------------------------------------------------------------------
    # Retrieval strategies
    # ------------------------------------------------------------------
    def _bm25_retrieve(self, question: str, timing: TimingResult):
        start = time.perf_counter()
        bm25 = self._ensure_bm25()
        docs = bm25.invoke(question)
        timing.bm25_search_ms = (time.perf_counter() - start) * 1000
        return docs[: self.cfg.top_k]

    def _vector_retrieve(self, question: str, timing: TimingResult):
        start = time.perf_counter()
        vs = self._ensure_vectorstore(timing)
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.cfg.top_k,
                "fetch_k": self.cfg.fetch_k,
                "lambda_mult": self.cfg.lambda_mult,
            },
        )
        docs = retriever.invoke(question)
        timing.vector_search_ms = (time.perf_counter() - start) * 1000
        return docs

    def _ensemble_retrieve(self, question: str, timing: TimingResult):
        start = time.perf_counter()
        vs = self._ensure_vectorstore(timing)
        vec_retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.cfg.ensemble_k,
                "fetch_k": self.cfg.fetch_k,
                "lambda_mult": self.cfg.lambda_mult,
            },
        )
        bm25 = self._ensure_bm25()
        ens = EnsembleRetriever(
            retrievers=[bm25, vec_retriever],
            weights=[self.cfg.bm25_weight, 1.0 - self.cfg.bm25_weight],
        )
        docs = ens.invoke(question)
        timing.ensemble_ms = (time.perf_counter() - start) * 1000
        return docs

    def _rerank(self, question: str, docs: List, timing: TimingResult):
        if len(docs) == 0:
            timing.rerank_ms = 0.0
            return docs
        start = time.perf_counter()
        pairs = [(question, d.page_content) for d in docs]
        scores = reranker_factory().predict(pairs)
        ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        timing.rerank_ms = (time.perf_counter() - start) * 1000
        return ranked

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def retrieve(self, question: str, timing: TimingResult) -> List[Document]:
        rtype = self.cfg.retriever_type

        if rtype == "bm25":
            docs = self._bm25_retrieve(question, timing)
        elif rtype == "vector":
            docs = self._vector_retrieve(question, timing)
        elif rtype == "ensemble":
            docs = self._ensemble_retrieve(question, timing)
        elif rtype == "bm25_rerank":
            docs = self._bm25_retrieve(question, timing)
            docs = self._rerank(question, docs, timing)
        elif rtype == "vector_rerank":
            docs = self._vector_retrieve(question, timing)
            docs = self._rerank(question, docs, timing)
        elif rtype == "ensemble_rerank":
            docs = self._ensemble_retrieve(question, timing)
            docs = self._rerank(question, docs, timing)
        else:
            raise ValueError(f"Unknown retriever type: {rtype}")

        timing.total_retrieval_ms = (
            timing.vector_search_ms
            + timing.bm25_search_ms
            + timing.ensemble_ms
            + timing.rerank_ms
        )
        return docs[: self.cfg.top_k]