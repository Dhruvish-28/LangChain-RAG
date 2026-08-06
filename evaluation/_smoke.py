"""Quick smoke test: run the benchmark pipeline on 2 questions for 1 config."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config import Config
from evaluation.core.benchmark import run_config
from evaluation.data_builder.ground_truth_builder import QUESTIONS

subset = QUESTIONS[:1]
cfg = Config(retriever_type="ensemble_rerank", rerank=True, top_k=3)
result = run_config(cfg, subset)
agg = result["aggregate"]
print("config:", agg["config"])
print("recall@3:", agg["recall@3"], "mrr:", agg["mrr"])
print("gen_rows:", len(result["generation_rows"]))
print("SMOKE_OK")
