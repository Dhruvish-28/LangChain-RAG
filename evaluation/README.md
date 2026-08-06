# RAG Evaluation Harness

A self-contained benchmark that measures the quality of the LangChain RAG
pipeline in this repo — **without modifying any of the existing app code**.

It builds a small synthetic corpus, defines 30 ground-truth Q&A pairs, and
scores every retrieval configuration in the grid on real, deterministic
metrics. All scoring is **LLM-free** — the harness compares retrieved chunks
to the corpus ground truth lexically, so it runs offline with zero API calls
and identical results every run.

## What it measures

| Category      | Metrics                                                        |
| ------------- | -------------------------------------------------------------- |
| Retrieval     | Recall@3, Recall@5, Recall@K, Precision@3, Precision@K, MRR    |
| Ranking time  | vector search, BM25, ensemble, rerank — per-stage ms           |
| Evidence      | Answer Present, Evidence Coverage, Top-1 Contains Gold, Full Evidence@K |

\* *All metrics are deterministic, lexical scoring — no LLM needed, zero API cost.*

## Config grid

`evaluation/config.py` defines a `Config` dataclass and `QUICK_CONFIGS` /
`FULL_CONFIGS` grids sweeping:

- **Chunk size** 250 / 500 / 1000 (overlap scales)
- **Retriever type** bm25, vector, ensemble, vector_rerank, bm25_rerank, ensemble_rerank
- **Top-k** 3 / 6 / 10
- **Rerank on/off**, ensemble weight, candidate pool size

The baseline `Config()` exactly matches the live app (`UI.py` → `retrieval/`).

## Run

```bash
# 1) Quick sanity run (small grid)
python -m evaluation.run_evaluation --quick

# 2) Full grid — identical deterministic metrics, runs offline
python -m evaluation.run_evaluation

# 3) Skip rebuilding corpus/ground-truth if they already exist
python -m evaluation.run_evaluation --skip-data
```

Outputs:

```
evaluation/data/
  corpus.json            # synthetic 6-doc corpus
  ground_truth.json      # 30 Q&A pairs with verbatim gold excerpts
evaluation/results/
  master_summary.csv     # one row per config, all aggregate scores
  retrieval_<config>.csv # per-question retrieval scores for each config
  evidence_<config>.csv  # per-question deterministic evidence scores
  SUMMARY_REPORT.md      # resume-ready markdown writeup
```

Run `python evaluation/report.py --csv <master.csv>` to regenerate the
Markdown report from a previous run without re-running the benchmark.
The whole harness runs offline — no `GOOGLE_API_KEY` or network calls beyond
the initial model downloads.

## Design notes

- **Deterministic scoring:** gold excerpts are character-exact substrings of
  the corpus, so retrieval hits are scored lexically — cheap, reproducible,
  and LLM-free.
- **Single source of truth:** all paths/models live in `evaluation/config.py`.
- **Isolated:** nothing in `evaluation/` edits the app's `ingestion/`,
  `retrieval/`, or `models/` — and no app-LLM code is imported, so the
  harness has zero API dependency.
- **Timed pipeline:** every retrieval stage (vector search, BM25, ensemble,
  rerank, index build) is timed per query so latency trends are visible.

## Requirements

The project's existing `requirements.txt` already covers `langchain`,
`langchain-community`, `faiss-cpu`, `sentence-transformers`, `rank_bm25`,
`sentence_transformers` reranker, and `chromadb`. No API key is required —
the harness runs fully offline.
