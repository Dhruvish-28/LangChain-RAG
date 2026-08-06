"""
Deterministic, LLM-free generation-quality metrics.

Instead of calling an LLM to generate an answer and another LLM to judge it,
the harness scores the *retrieved context* directly against the ground-truth
gold excerpts:

  - answer_present        : does any retrieved chunk contain the gold answer text?
  - evidence_coverage     : fraction of the gold excerpts present in the context
  - top1_contains_gold    : is the top-ranked chunk a hit?
  - full_evidence_topk    : are ALL gold excerpts present in the top-k context?

All signals are lexical (character-exact substrings from the corpus), fully
deterministic, require zero API calls, and run in microseconds.
"""

from typing import List


def answer_present(retrieved, gold_answer: str) -> float:
    """1.0 if the gold answer appears verbatim in any retrieved chunk."""
    if not gold_answer:
        return 0.0
    return 1.0 if any(gold_answer in d.page_content for d in retrieved) else 0.0


def evidence_coverage(retrieved, gold_excerpts: List[str]) -> float:
    """Fraction of gold excerpts found in the retrieved context (0..1)."""
    if not gold_excerpts:
        return 0.0
    context = "\n".join(d.page_content for d in retrieved)
    hits = sum(1 for e in gold_excerpts if e in context)
    return hits / len(gold_excerpts)


def top1_contains_gold(retrieved, gold_excerpts: List[str]) -> float:
    """1.0 if the top-ranked chunk contains any gold excerpt."""
    if not retrieved:
        return 0.0
    return 1.0 if any(e in retrieved[0].page_content for e in gold_excerpts) else 0.0


def full_evidence_topk(retrieved, gold_excerpts: List[str], k: int) -> float:
    """1.0 if ALL gold excerpts appear across the top-k context."""
    if not gold_excerpts:
        return 0.0
    context = "\n".join(d.page_content for d in retrieved[:k])
    return 1.0 if all(e in context for e in gold_excerpts) else 0.0