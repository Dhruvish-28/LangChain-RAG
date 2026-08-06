"""
Benchmark runner.

For every Config in the grid:
  1. Build the FAISS index (timed).
  2. For every ground-truth question:
       - retrieve top-k chunks (timed per stage)
       - score objective retrieval metrics (Recall@k, Precision@k, MRR)
       - score deterministic evidence metrics from the retrieved context
  3. Aggregate + save results to evaluation/results/ as CSVs.

Everything is deterministic and LLM-free — no API key needed, runs offline.
"""

import csv
import os
from statistics import mean
from typing import List

from evaluation.config import Config, config_id, RESULTS_DIR
from evaluation.core.retrievers import RetrievalEngine, TimingResult
from evaluation.data_builder.ground_truth_builder import QUESTIONS
from evaluation.metrics.generation_metrics import (
    answer_present,
    evidence_coverage,
    full_evidence_topk,
    top1_contains_gold,
)
from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else 0.0


def run_config(cfg: Config, questions):
    """Run a single configuration against every ground-truth question."""
    print(f"\n=== Running config: {config_id(cfg)} ===")

    engine = RetrievalEngine(cfg)
    index_timing = TimingResult()
    engine.build_index(index_timing)

    retrieval_rows = []
    generation_rows = []

    for q in questions:
        question = q["question"]
        gold = q["gold_excerpts"]
        gold_answer = q["answer"]

        t = TimingResult()
        docs = engine.retrieve(question, t)

        # Objective retrieval metrics (deterministic, no LLM calls)
        r3 = recall_at_k(docs, gold, 3)
        r5 = recall_at_k(docs, gold, 5)
        rk = recall_at_k(docs, gold, cfg.top_k)
        p3 = precision_at_k(docs, gold, 3)
        pk = precision_at_k(docs, gold, cfg.top_k)
        mrr = reciprocal_rank(docs, gold)

        retrieval_rows.append(
            {
                "question": question,
                "gold": gold[0][:60],
                "recall@3": r3,
                "recall@5": r5,
                "recall@top_k": rk,
                "precision@3": p3,
                "precision@top_k": pk,
                "mrr": mrr,
                "vector_ms": round(t.vector_search_ms, 2),
                "bm25_ms": round(t.bm25_search_ms, 2),
                "ensemble_ms": round(t.ensemble_ms, 2),
                "rerank_ms": round(t.rerank_ms, 2),
                "total_retrieval_ms": round(t.total_retrieval_ms, 2),
            }
        )

        # Deterministic evidence metrics (run for every config, no API cost)
        generation_rows.append(
            {
                "question": question,
                "answer_present": answer_present(docs, gold_answer),
                "evidence_coverage": evidence_coverage(docs, gold),
                "top1_contains_gold": top1_contains_gold(docs, gold),
                "full_evidence_topk": full_evidence_topk(docs, gold, cfg.top_k),
            }
        )

    agg = {
        "config": config_id(cfg),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "retriever_type": cfg.retriever_type,
        "top_k": cfg.top_k,
        "ensemble_k": cfg.ensemble_k,
        "fetch_k": cfg.fetch_k,
        "bm25_weight": cfg.bm25_weight,
        "rerank": cfg.rerank,
        "index_build_ms": round(index_timing.index_build_ms, 2),
        "total_retrieval_avg_ms": round(
            _safe_mean([r["total_retrieval_ms"] for r in retrieval_rows]), 2
        ),
        "recall@3": round(_safe_mean([r["recall@3"] for r in retrieval_rows]), 3),
        "recall@5": round(_safe_mean([r["recall@5"] for r in retrieval_rows]), 3),
        "recall@top_k": round(_safe_mean([r["recall@top_k"] for r in retrieval_rows]), 3),
        "precision@3": round(_safe_mean([r["precision@3"] for r in retrieval_rows]), 3),
        "precision@top_k": round(_safe_mean([r["precision@top_k"] for r in retrieval_rows]), 3),
        "mrr": round(_safe_mean([r["mrr"] for r in retrieval_rows]), 3),
        "answer_present": round(
            _safe_mean([r["answer_present"] for r in generation_rows]), 3
        ),
        "evidence_coverage": round(
            _safe_mean([r["evidence_coverage"] for r in generation_rows]), 3
        ),
        "top1_contains_gold": round(
            _safe_mean([r["top1_contains_gold"] for r in generation_rows]), 3
        ),
        "full_evidence_topk": round(
            _safe_mean([r["full_evidence_topk"] for r in generation_rows]), 3
        ),
    }

    return {
        "aggregate": agg,
        "retrieval_rows": retrieval_rows,
        "generation_rows": generation_rows,
    }


def run_all(configs: List[Config], out_dir=RESULTS_DIR, questions=None):
    """Run all configs and write per-config retrieval/evidence CSVs + a master table."""
    os.makedirs(out_dir, exist_ok=True)

    if questions is None:
        questions = QUESTIONS

    all_rows, all_gen_rows = [], []

    for cfg in configs:
        result = run_config(cfg, questions)

        # Per-config detail CSVs
        ret_path = os.path.join(out_dir, f"retrieval_{config_id(cfg)}.csv")
        _write_csv(ret_path, result["retrieval_rows"])
        gen_path = os.path.join(out_dir, f"evidence_{config_id(cfg)}.csv")
        _write_csv(gen_path, result["generation_rows"])

        all_rows.append(result["aggregate"])
        all_gen_rows.append(result["generation_rows"])

    # Master aggregate table
    master_path = os.path.join(out_dir, "master_summary.csv")
    _write_csv(master_path, all_rows)

    return all_rows


def _write_csv(path, rows, fields=None):
    if not rows:
        return
    if fields is None:
        fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
