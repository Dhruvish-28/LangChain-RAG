"""
Builds the ground-truth question dataset used for retrieval evaluation.

Each entry contains:
  - question      : the query given to the retrieval pipeline
  - answer        : the exact, concise answer found in the corpus (objective)
  - document_id   : which corpus document the fact belongs to
  - gold_excerpts : verbatim sentences/snippets from the corpus that must be
                    present in a retrieved chunk for that chunk to count as
                    a "hit" (ground truth is chunk-agnostic, so it works for
                    every chunking configuration in the benchmark grid).

The excerpts are copied character-for-character from the corpus text, so
matching is purely lexical — no LLM or human judgement is required to
score retrieval quality.
"""

import json
import os

from evaluation.config import GROUND_TRUTH_PATH
from evaluation.data_builder.corpus_builder import DOCUMENTS

# ---------------------------------------------------------------------------
# Ground-truth questions.
#
# The `gold_excerpts` strings below are substrings of the documents defined in
# corpus_builder.py. They are intentionally self-contained, factual sentences.
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ---- Employee handbook ----
    {
        "question": "How many paid vacation days do full-time employees at Atlas Technologies accrue per year?",
        "document_id": "employee_handbook",
        "answer": "24 paid vacation days per calendar year.",
        "gold_excerpts": [
            "Full-time employees accrue twenty-four paid vacation days per calendar year.",
        ],
    },
    {
        "question": "What happens with unused sick days at Atlas?",
        "document_id": "employee_handbook",
        "answer": "Unused sick days do not roll over.",
        "gold_excerpts": [
            "Unused sick days do not roll over.",
        ],
    },
    {
        "question": "How many weeks of paid parental leave does Atlas provide?",
        "document_id": "employee_handbook",
        "answer": "16 weeks of fully paid time off.",
        "gold_excerpts": [
            "Parental leave provides sixteen weeks of fully paid time off for the birth, adoption, or fostering of a child.",
        ],
    },
    {
        "question": "How much is the learning and development budget per year at Atlas?",
        "document_id": "employee_handbook",
        "answer": "$1,500 per year.",
        "gold_excerpts": [
            "learning and development budget of fifteen hundred dollars per year",
        ],
    },
    {
        "question": "By when must expense reports be submitted at Atlas?",
        "document_id": "employee_handbook",
        "answer": "Within 30 days of the expense date.",
        "gold_excerpts": [
            "All expense reports must be submitted within thirty days of the expense date.",
        ],
    },
    {
        "question": "What is the vesting schedule for restricted stock units at Atlas?",
        "document_id": "employee_handbook",
        "answer": "RSUs vest over four years with a one-year cliff; 25% vests annually after the first year.",
        "gold_excerpts": [
            "restricted stock units that vest over four years with a one-year cliff",
            "twenty-five percent of the grant vests annually",
        ],
    },
    {
        "question": "What rating qualifies an employee for the top bonus tier at Atlas?",
        "document_id": "employee_handbook",
        "answer": "An overall rating of 4 or higher.",
        "gold_excerpts": [
            "An overall rating of four or higher qualifies an employee for the top bonus tier.",
        ],
    },

    # ---- API guide ----
    {
        "question": "What HTTP status code does the Atlas Payments API return for rate limit exceeded?",
        "document_id": "api_guide",
        "answer": "HTTP 429 with a Retry-After header.",
        "gold_excerpts": [
            "the API returns HTTP 429 with a Retry-After header indicating the number of seconds to wait",
        ],
    },
    {
        "question": "What is the default page size for Atlas Payments list endpoints?",
        "document_id": "api_guide",
        "answer": "50 results per page (page_size can be set 1-200).",
        "gold_excerpts": [
            "The page_size parameter controls the number of results per page and can be set between one and two hundred, with a default of fifty.",
        ],
    },
    {
        "question": "How is the webhook signature generated in the Atlas Payments API?",
        "document_id": "api_guide",
        "answer": "HMAC-SHA256 computed over the raw payload and the webhook secret.",
        "gold_excerpts": [
            "Atlas-Signature header containing an HMAC-SHA256 signature computed over the raw payload and the webhook secret",
        ],
    },
    {
        "question": "How long is a webhook signature valid?",
        "document_id": "api_guide",
        "answer": "Five minutes after generation.",
        "gold_excerpts": [
            "Signatures are valid for five minutes after generation.",
        ],
    },
    {
        "question": "What is the idempotency window for replaying POST requests on the Atlas API?",
        "document_id": "api_guide",
        "answer": "24 hours.",
        "gold_excerpts": [
            "Replaying a request with the same idempotency key within twenty-four hours returns the original response without creating a duplicate resource.",
        ],
    },
    {
        "question": "Which request header does the Atlas Payments API require for authentication?",
        "document_id": "api_guide",
        "answer": "Authorization header using the Bearer scheme.",
        "gold_excerpts": [
            "Every request must include an API key in the Authorization header using the Bearer scheme.",
        ],
    },

    # ---- ML fundamentals ----
    {
        "question": "What is the standard dropout probability for hidden layers in neural networks?",
        "document_id": "ml_fundamentals",
        "answer": "0.5 (0.1 for inputs).",
        "gold_excerpts": [
            "The default dropout probability is commonly set to 0.5 for hidden layers and 0.1 for inputs.",
        ],
    },
    {
        "question": "Which regularization technique drives weights exactly to zero?",
        "document_id": "ml_fundamentals",
        "answer": "L1 regularization.",
        "gold_excerpts": [
            "L1 regularization drives many weights exactly to zero, producing sparse models",
        ],
    },
    {
        "question": "What are the standard Adam optimizer hyperparameters?",
        "document_id": "ml_fundamentals",
        "answer": "Learning rate 0.001, beta1 0.9, beta2 0.999, epsilon 1e-8.",
        "gold_excerpts": [
            "Standard Adam hyperparameters are a learning rate of 0.001, beta1 of 0.9, and beta2 of 0.999 with a small epsilon of 1e-8.",
        ],
    },
    {
        "question": "What is a common dataset split for training, validation, and test sets?",
        "document_id": "ml_fundamentals",
        "answer": "80% training, 10% validation, 10% test.",
        "gold_excerpts": [
            "A common split is eighty percent training, ten percent validation, and ten percent test.",
        ],
    },
    {
        "question": "Which metric is preferred over accuracy for imbalanced classification?",
        "document_id": "ml_fundamentals",
        "answer": "F1 score (harmonic mean of precision and recall).",
        "gold_excerpts": [
            "The F1 score is the harmonic mean of precision and recall and is preferred over accuracy when classes are imbalanced.",
        ],
    },

    # ---- Financial report ----
    {
        "question": "What was Atlas Technologies total revenue for fiscal year 2025?",
        "document_id": "financial_report",
        "answer": "$842 million, up 18% year over year.",
        "gold_excerpts": [
            "total revenue of eight hundred forty-two million dollars, representing year-over-year growth of eighteen percent",
        ],
    },
    {
        "question": "What was the gross margin for fiscal 2025?",
        "document_id": "financial_report",
        "answer": "78%, up 2 percentage points from the prior year.",
        "gold_excerpts": [
            "Gross margin for fiscal 2025 was seventy-eight percent, up two percentage points from the prior year",
        ],
    },
    {
        "question": "How many enterprise customers did Atlas close fiscal 2025 with?",
        "document_id": "financial_report",
        "answer": "3,400 enterprise customers (up from 2,900).",
        "gold_excerpts": [
            "The company closed the year with thirty-four hundred enterprise customers, up from twenty-nine hundred at the end of the prior year",
        ],
    },
    {
        "question": "What is the fiscal 2026 revenue guidance range?",
        "document_id": "financial_report",
        "answer": "$985 million to $1.02 billion (growth of ~17-21%).",
        "gold_excerpts": [
            "Management guided fiscal 2026 revenue in the range of nine hundred eighty-five million to one point zero two billion dollars",
        ],
    },
    {
        "question": "What was the dollar-based net revenue retention rate at the end of fiscal 2025?",
        "document_id": "financial_report",
        "answer": "121%.",
        "gold_excerpts": [
            "net revenue retention rate of one hundred twenty-one percent",
        ],
    },

    # ---- Nutrition science ----
    {
        "question": "How many kilocalories per gram does fat provide?",
        "document_id": "nutrition_science",
        "answer": "9 kilocalories per gram.",
        "gold_excerpts": [
            "fat provides nine kilocalories per gram",
        ],
    },
    {
        "question": "What is the recommended dietary allowance of vitamin D for adults over 70?",
        "document_id": "nutrition_science",
        "answer": "800 international units per day.",
        "gold_excerpts": [
            "eight hundred international units per day for adults over seventy",
        ],
    },
    {
        "question": "What is the daily fiber recommendation for adult women?",
        "document_id": "nutrition_science",
        "answer": "25 grams per day.",
        "gold_excerpts": [
            "Total fiber recommendations are twenty-five grams per day for adult women",
        ],
    },
    {
        "question": "What is the protein recommendation for athletic populations?",
        "document_id": "nutrition_science",
        "answer": "1.6 to 2.2 grams per kilogram of body weight per day.",
        "gold_excerpts": [
            "Athletic populations benefit from higher intakes between 1.6 and 2.2 grams per kilogram per day",
        ],
    },
    {
        "question": "How much water per day is recommended for adult men?",
        "document_id": "nutrition_science",
        "answer": "Approximately 3.7 liters per day.",
        "gold_excerpts": [
            "Total water intake recommendations are approximately 3.7 liters per day for adult men",
        ],
    },

    # ---- Cloud architecture ----
    {
        "question": "What is a common CPU utilization target for autoscaling policies?",
        "document_id": "cloud_architecture",
        "answer": "Around 60% average CPU utilization, with a 5-minute cooldown.",
        "gold_excerpts": [
            "average CPU utilization around sixty percent, with a cooldown period of five minutes between scaling actions",
        ],
    },
    {
        "question": "What is the most common caching pattern in cloud applications?",
        "document_id": "cloud_architecture",
        "answer": "Cache-aside.",
        "gold_excerpts": [
            "Cache-aside is the most common caching pattern",
        ],
    },
    {
        "question": "What are the three pillars of observability?",
        "document_id": "cloud_architecture",
        "answer": "Logs, metrics, and traces.",
        "gold_excerpts": [
            "Observability rests on three pillars: logs, metrics, and traces.",
        ],
    },
    {
        "question": "What is a common default availability target (SLO) for production APIs?",
        "document_id": "cloud_architecture",
        "answer": "99.9% uptime.",
        "gold_excerpts": [
            "ninety-nine point nine percent uptime a common default for production APIs",
        ],
    },
    {
        "question": "What is the naming of schema-per-tenant isolation in SaaS?",
        "document_id": "cloud_architecture",
        "answer": "Schema-per-tenant or row-level tenancy, rather than a dedicated database per tenant.",
        "gold_excerpts": [
            "Software-as-a-service tenants should be isolated using schema-per-tenant or row-level tenancy rather than a dedicated database per tenant",
        ],
    },
]


def build_ground_truth():
    """Validate excerpts against the corpus, then write ground truth JSON."""
    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)

    corpus_text = {doc["id"]: doc["content"] for doc in DOCUMENTS}

    missing = []

    for q in QUESTIONS:
        doc_id = q["document_id"]
        text = corpus_text.get(doc_id, "")
        for snippet in q["gold_excerpts"]:
            if snippet not in text:
                missing.append((q["question"], doc_id, snippet))

    if missing:
        raise ValueError(
            f"{len(missing)} gold excerpt(s) do not exist in the corpus:\n"
            + "\n".join(f"  - Q: {q[0]} | doc={q[1]}: {q[2][:80]}..." for q in missing)
        )

    payload = {
        "description": (
            f"Ground-truth retrieval dataset: {len(QUESTIONS)} questions "
            "over 6 documents. Excerpts are verbatim corpus substrings, so "
            "hits are scored lexically."
        ),
        "questions": QUESTIONS,
    }

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[ground_truth_builder] Wrote {len(QUESTIONS)} questions to {GROUND_TRUTH_PATH}")
    return payload


if __name__ == "__main__":
    build_ground_truth()