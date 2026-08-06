"""
Evaluation configuration.

Central place for all tunable parameters and the benchmark grid.
Nothing in this package modifies the existing app code.
"""

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Fixed model choices (mirror the app so comparisons are apples-to-apples)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Baseline configuration (matches the live app: UI.py -> retrieval/)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_type: str = "ensemble_rerank"    # see RETRIEVER_TYPES
    top_k: int = 6                             # final number of chunks used as context
    ensemble_k: int = 20                       # candidate pool before ensemble merging
    fetch_k: int = 30                          # FAISS MMR fetch pool
    lambda_mult: float = 0.5                   # FAISS MMR diversity
    bm25_weight: float = 0.4                   # ensemble weight (vector = 1 - bm25)
    rerank: bool = True


# Retriever strategies available for benchmarking
RETRIEVER_TYPES = [
    "bm25",                 # keyword-only
    "vector",               # FAISS MMR only
    "ensemble",             # BM25 + FAISS MMR (no rerank)
    "vector_rerank",        # FAISS MMR + cross-encoder rerank
    "bm25_rerank",          # BM25 + cross-encoder rerank
    "ensemble_rerank",      # BM25 + FAISS MMR + cross-encoder rerank  <-- app baseline
]


# ---------------------------------------------------------------------------
# Benchmark grid.
#
# The full sweep runs every configuration in COMBINATIONS below.
# A "focus run" (--quick) runs only BASELINE + the one-at-a-time
# comparisons that tell the best story for a resume.
# ---------------------------------------------------------------------------

QUICK_CONFIGS: List[Config] = [
    # 1) Baseline = exactly what the app ships with
    Config(),

    # 2) Retriever ablation: no rerank
    Config(retriever_type="ensemble", rerank=False),

    # 3) Retriever ablation: single retrievers
    Config(retriever_type="bm25", rerank=False),
    Config(retriever_type="vector", rerank=False),

    # 4) Rerank-only variants
    Config(retriever_type="vector_rerank", rerank=True),
    Config(retriever_type="bm25_rerank", rerank=True),

    # 5) Chunk size sweep (rerank ensemble kept)
    Config(chunk_size=250, chunk_overlap=25),
    Config(chunk_size=1000, chunk_overlap=100),

    # 6) Top-K sweep (final context length)
    Config(top_k=3),
    Config(top_k=10),
]

FULL_CONFIGS: List[Config] = [
    *QUICK_CONFIGS,

    # Chunk overlap sweep
    Config(chunk_size=500, chunk_overlap=0),
    Config(chunk_size=500, chunk_overlap=100),

    # Ensemble weight sweep
    Config(bm25_weight=0.3),
    Config(bm25_weight=0.5),

    # Candidate pool sweep
    Config(ensemble_k=10, fetch_k=15),
    Config(ensemble_k=35, fetch_k=50),
]

ALL_CONFIGS = FULL_CONFIGS


def config_id(cfg: Config) -> str:
    """Human-readable unique id for a config (used for file/dir names)."""
    parts = [
        f"chunk{cfg.chunk_size}x{cfg.chunk_overlap}",
        str(cfg.retriever_type),
        f"k{cfg.top_k}",
    ]
    if cfg.rerank:
        parts.append("rerank")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Dataset / corpus paths
# ---------------------------------------------------------------------------
CORPUS_PATH = "evaluation/data/corpus.json"
GROUND_TRUTH_PATH = "evaluation/data/ground_truth.json"

RESULTS_DIR = "evaluation/results"
VECTOR_CACHE_DIR = "evaluation/data/vector_cache"
REPORT_PATH = "evaluation/results/SUMMARY_REPORT.md"