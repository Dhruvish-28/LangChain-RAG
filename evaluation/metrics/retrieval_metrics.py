"""
Objective retrieval-quality metrics.

A retrieved chunk counts as a "hit" for a question if it contains *any* of
the question's gold excerpts (verbatim substrings from the corpus). Because
excerpts are lexical substrings, scoring is fully deterministic and requires
no LLM calls — results are identical every run.
"""

from typing import List


def recall_at_k(retrieved, gold_excerpts, k):
    """Recall@K: did any retrieved chunk (in top-k) contain a gold excerpt?"""
    top = retrieved[:k]
    for doc in top:
        if any(excerpt in doc.page_content for excerpt in gold_excerpts):
            return 1.0
    return 0.0


def precision_at_k(retrieved, gold_excerpts, k):
    """Precision@K = fraction of top-k chunks that contain a gold excerpt."""
    top = retrieved[:k]
    if len(top) == 0:
        return 0.0
    hits = sum(
        1 for doc in top if any(excerpt in doc.page_content for excerpt in gold_excerpts)
    )
    return hits / len(top)


def reciprocal_rank(retrieved, gold_excerpts):
    """RR = 1 / rank of the first relevant chunk (0 if none retrieved)."""
    for i, doc in enumerate(retrieved, start=1):
        if any(excerpt in doc.page_content for excerpt in gold_excerpts):
            return 1.0 / i
    return 0.0