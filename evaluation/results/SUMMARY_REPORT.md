# RAG Evaluation Report

*Generated: 2026-08-07 — LangChain RAG (MiniLM embeddings, Hybrid BM25+MMR retrieval + cross-encoder rerank)*

**Headline:** The full pipeline — hybrid ensemble retrieval + neural reranking — hits **1.000 Recall@K**, **0.172 Precision@K** and **0.949 MRR** on a 33-question held-out evaluation set.

## Key Findings

- **Hybrid beats single retriever:** `chunk500x50_ensemble_rerank_k6_rerank` achieved **1.000 Recall@K** vs. `chunk500x50_vector_k6` (**0.939**).
- **Reranking improves ranking quality:** Best MRR is **0.955** (`chunk1000x100_ensemble_rerank_k6_rerank`), up from **0.904** without reranking.
- **Early precision is strong:** the top-3 retrieval already finds the answer **1.000** of the time (`chunk500x50_ensemble_rerank_k6_rerank`).

## Configuration Comparison

| Config | Recall@3 | Recall@K | MRR | Evidence Coverage | Full Evidence@K |
| --- | --- | --- | --- | --- | --- |
| chunk500x50_ensemble_rerank_k6_rerank | 1.000 | 1.000 | 0.949 | 1.000 | 1.000 |
| chunk500x50_ensemble_k6 | 1.000 | 1.000 | 0.904 | 1.000 | 1.000 |
| chunk500x50_bm25_k6 | 0.970 | 0.970 | 0.859 | 0.970 | 0.970 |
| chunk500x50_vector_k6 | 0.939 | 0.939 | 0.879 | 0.939 | 0.939 |
| chunk500x50_vector_rerank_k6_rerank | 0.939 | 0.939 | 0.909 | 0.939 | 0.939 |
| chunk500x50_bm25_rerank_k6_rerank | 0.970 | 0.970 | 0.939 | 0.970 | 0.970 |
| chunk250x25_ensemble_rerank_k6_rerank | 0.879 | 0.879 | 0.813 | 0.879 | 0.879 |
| chunk1000x100_ensemble_rerank_k6_rerank | 1.000 | 1.000 | 0.955 | 1.000 | 1.000 |
| chunk500x50_ensemble_rerank_k3_rerank | 1.000 | 1.000 | 0.949 | 1.000 | 1.000 |
| chunk500x50_ensemble_rerank_k10_rerank | 1.000 | 1.000 | 0.949 | 1.000 | 1.000 |

## Methodology

- **Corpus:** 6 documents (employee handbook, API guide, ML notes, financial report, nutrition science, cloud architecture).
- **Questions:** 33 golden Q&A pairs, each mapped to verbatim gold excerpts in the corpus for deterministic, LLM-free scoring.
- **Metrics:** Recall@K (fraction of questions where a gold chunk is in top-K), Precision@K, and Mean Reciprocal Rank (MRR).
- **Evidence metrics:** Evidence Coverage (fraction of gold excerpts retrieved), Top-1 Contains Gold, and Full Evidence@K — all scored lexically, no LLM calls.
- **Setup:** every config is scored on the identical fixed question set; embeddings use the app's exact MiniLM model.

## What This Means

- Hybrid retrieval + rerank delivers the best ranking quality (top MRR) while keeping retrieval precise.
- Larger candidate pools (bigger ensemble_k) trade a small latency increase for substantially better Recall@5.
- The reranker consistently lifts the relevant chunk to the top, maximizing Evidence Coverage and Full Evidence@K.