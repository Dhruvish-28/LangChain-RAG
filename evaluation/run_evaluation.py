"""
Entry point for the RAG evaluation harness.

Usage:
    python -m evaluation.run_evaluation --quick     # small grid, fast check
    python -m evaluation.run_evaluation             # full grid

Flow:
    1. (Re)builds the synthetic corpus  -> evaluation/data/corpus.json
    2. (Re)builds ground-truth Q&A      -> evaluation/data/ground_truth.json
    3. Runs the benchmark grid          -> evaluation/results/*.csv
    4. Generates the Markdown report    -> evaluation/results/SUMMARY_REPORT.md

All scoring is deterministic and LLM-free: retrieval hits and evidence
coverage are scored by exact lexical matching against the corpus ground
truth. No API key is required — the benchmark runs completely offline.
"""

import argparse
import os
import sys

# Make the project root importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config import FULL_CONFIGS, QUICK_CONFIGS
from evaluation.data_builder.corpus_builder import build_corpus
from evaluation.data_builder.ground_truth_builder import build_ground_truth
from evaluation.core.benchmark import run_all
from evaluation.report import generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run only the quick config grid (baseline + key ablations).",
    )
    ap.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip rebuilding corpus + ground truth (use existing JSON).",
    )
    args = ap.parse_args()

    if not args.skip_data:
        build_corpus()
        build_ground_truth()

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    print(f"\n[run_evaluation] Running {len(configs)} configurations...")

    rows = run_all(configs)
    print(f"\n[run_evaluation] Completed {len(rows)} configs -> evaluation/results/master_summary.csv")

    from evaluation.config import RESULTS_DIR

    generate(os.path.join(RESULTS_DIR, "master_summary.csv"))
    print("\nDone. Open evaluation/results/SUMMARY_REPORT.md")


if __name__ == "__main__":
    main()