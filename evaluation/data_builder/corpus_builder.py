"""
Builds the evaluation corpus: a deterministic, self-contained set of
realistic documents spanning multiple domains.

The corpus is written to evaluation/data/corpus.json and is used by every
benchmark. It intentionally contains dense, factual content (numbers,
procedures, specifications) so that ground-truth questions have objective
answers and retrieval quality can be measured without human annotation.

Nothing in this module imports or modifies the existing app code.
"""

import json
import os

from evaluation.config import CORPUS_PATH

# ---------------------------------------------------------------------------
# Documents
#
# Each document is a long, structured text. Paragraphs are separated by
# double newlines. Facts are deliberately specific (exact numbers, dates,
# procedures) so chunk-level ground truth is unambiguous.
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": "employee_handbook",
        "title": "Atlas Technologies Employee Handbook 2025",
        "path": "employee_handbook.md",
        "content": """
Atlas Technologies Employee Handbook 2025

Remote Work Policy

Atlas operates a hybrid working model. All full-time employees are expected to work from the office at least three days per week, typically Tuesday through Thursday. Remote-only arrangements require written approval from both the department head and the people operations team. Employees who wish to work remotely for more than two consecutive weeks must submit a formal request at least fifteen business days in advance.

The company provides a remote work allowance of one hundred dollars per month to cover home internet and utilities for employees who work remotely at least eight days per month. Laptops, monitors, and ergonomic chairs are provided for both office and home setups. Employees are responsible for maintaining a secure home network and must use the company VPN whenever accessing internal systems.

Vacation and Leave Policy

Full-time employees accrue twenty-four paid vacation days per calendar year. Vacation days can be carried over up to five days into the following year. In addition to vacation, employees receive twelve paid sick days and five paid personal days annually. Unused sick days do not roll over.

Parental leave provides sixteen weeks of fully paid time off for the birth, adoption, or fostering of a child. Employees returning from parental leave receive a phased return of half time at full pay for the first four weeks back at work. Caregiver leave for the serious illness of a family member provides up to six weeks of paid leave per year.

Performance Review Process

Atlas runs quarterly check-in meetings between employees and their managers, followed by a formal annual review in December. Each employee is rated on a scale from one to five across five core competencies: technical execution, collaboration, communication, delivery, and growth mindset. An overall rating of four or higher qualifies an employee for the top bonus tier.

Ratings are finalized through a calibration process where managers meet to compare scores and ensure consistency across teams. Performance improvement plans are offered to employees who receive an overall rating of two or below, with a structured ninety-day review period.

Compensation and Benefits

The compensation review cycle runs annually in January with merit increases and equity refreshes. New hires receive equity in the form of restricted stock units that vest over four years with a one-year cliff. This means no shares vest before the first anniversary, after which twenty-five percent of the grant vests annually.

Employees receive a health and wellness stipend of seven hundred fifty dollars per year and a learning and development budget of fifteen hundred dollars per year. The learning budget can be used for courses, certifications, conferences, and books. Tuition reimbursement covers up to five thousand dollars per year for approved degree programs related to the employee's role.

Expense Policy

Business travel must be booked at least two weeks in advance where possible to receive the preferred airline rates. Hotel stays are reimbursed up to four hundred dollars per night in major cities and two hundred fifty dollars per night elsewhere. Meals are reimbursed at a per diem rate of seventy-five dollars for full travel days and forty dollars for partial days.

All expense reports must be submitted within thirty days of the expense date. Receipts are required for any single expense above twenty-five dollars. The company issues corporate cards to employees who travel more than six times per year.

Security and Data Policies

Employees must complete security awareness training within the first two weeks of employment and every six months afterwards. Passwords must be at least fourteen characters and multi-factor authentication is mandatory for all accounts. Access to sensitive customer data is granted on a least-privilege basis and reviewed quarterly.

Personal devices used for work must be registered with the IT department and enrolled in the mobile device management system. Lost devices must be reported within two hours to the security operations center. Violations of data handling policies are subject to disciplinary action up to and including termination.
""".strip(),
    },
    {
        "id": "api_guide",
        "title": "Atlas Payments REST API Reference v3",
        "path": "api_guide.md",
        "content": """
Atlas Payments REST API Reference v3

Getting Started

The Atlas Payments API is a RESTful JSON API. All requests must be made over HTTPS using the base URL https://api.atlaspay.io. The current version is v3 and is reached through the versioned path /v3. Breaking changes are released at most once per year, and deprecated endpoints remain available for twelve months with a Deprecation header returned on every call.

Authentication

Every request must include an API key in the Authorization header using the Bearer scheme. API keys are issued in the dashboard and can be scoped to read-only or read-write access. OAuth2 client credentials flow is supported for server-to-server integrations and returns an access token valid for one hour.

Production API keys are restricted by IP allowlist. Requests without valid credentials return HTTP 401 with a problem detail body. Requests from an allowlisted IP but with an invalid scope return HTTP 403.

Rate Limits

Free tier accounts are limited to one hundred requests per minute. Pro accounts receive one thousand requests per minute, and enterprise accounts receive ten thousand requests per minute with the ability to purchase additional capacity. Rate limits are enforced per API key using a token bucket algorithm.

When a rate limit is exceeded, the API returns HTTP 429 with a Retry-After header indicating the number of seconds to wait. Clients should implement exponential backoff starting at one second and doubling each attempt, capped at sixty seconds, with jitter to avoid thundering herd effects.

Core Endpoints

The orders resource is the central object for payment processing. Create an order with POST /v3/orders, list orders with GET /v3/orders, retrieve a single order with GET /v3/orders/{order_id}, and update billable fields with PATCH /v3/orders/{order_id}. Refunds are issued with POST /v3/orders/{order_id}/refunds and can be full or partial.

Order amounts are expressed in minor currency units. For example, an amount of one thousand cents with currency code USD represents ten US dollars. Each order carries a unique order_id generated by the platform, and clients may attach an external reference via the idempotency_key field.

Pagination and Filtering

List endpoints use cursor-based pagination. The response includes a next_cursor value pointing to the next page. Cursor values are opaque and expire after one hour. The page_size parameter controls the number of results per page and can be set between one and two hundred, with a default of fifty.

Filtering is supported through query parameters such as created_at[gte], created_at[lte], status, and currency. Filters can be combined, and list responses always return items ordered by creation time descending.

Error Handling

Errors follow the RFC 7807 problem details format with fields type, title, status, detail, and instance. The type field is a URI that uniquely identifies the error category. Common categories include invalid_request, authentication_error, rate_limit_error, and processing_error.

Idempotency and Retries

POST requests that create resources accept an Idempotency-Key header. Replaying a request with the same idempotency key within twenty-four hours returns the original response without creating a duplicate resource. The key must be a UUID string of fifty characters or less. Idempotency keys are recommended for all mutation requests, especially refunds and order creation.

Webhooks

Webhook notifications are sent to configured HTTPS endpoints for events such as order.completed, refund.succeeded, and dispute.created. Each delivery includes an Atlas-Signature header containing an HMAC-SHA256 signature computed over the raw payload and the webhook secret. Signatures are valid for five minutes after generation.

Deliveries are retried up to three times using exponential backoff with delays of five minutes, thirty minutes, and two hours. Delivery attempts that fail four times are dropped, and the failing event is added to the dashboard's webhook log for manual inspection.
""".strip(),
    },
    {
        "id": "ml_fundamentals",
        "title": "Machine Learning Fundamentals — Technical Notes",
        "path": "ml_fundamentals.md",
        "content": """
Machine Learning Fundamentals — Technical Notes

Overfitting and Underfitting

Overfitting occurs when a model memorizes the training data instead of learning general patterns. It is characterized by very low training error but high validation error, often a symptom of high variance. Underfitting is the opposite: the model is too simple to capture the underlying structure, resulting in high error on both training and validation sets, a symptom of high bias.

Classic techniques to combat overfitting include collecting more data, reducing model capacity, applying regularization, using dropout, and early stopping. The capacity of a model should be matched to the complexity of the task and the amount of available data.

Regularization

L1 regularization adds the sum of the absolute values of the weights to the loss function. Because the derivative of the absolute value is constant, L1 regularization drives many weights exactly to zero, producing sparse models that are useful for feature selection. L2 regularization adds the sum of squared weights and instead shrinks weights toward zero without making them exactly zero, which improves generalization when many features carry small signal.

Dropout is another effective regularizer for neural networks. During training, a fraction of neurons are randomly deactivated each forward pass. The default dropout probability is commonly set to 0.5 for hidden layers and 0.1 for inputs. Dropout forces the network to learn redundant representations and reduces co-adaptation between neurons.

Gradient Descent and Optimizers

Gradient descent updates model parameters in the direction of steepest descent of the loss function. The learning rate controls the step size; too large a learning rate causes divergence, while too small a learning rate slows convergence dramatically. A common practice is to use learning rate schedules that warm up over the first few epochs and then decay.

Momentum accelerates gradient descent by accumulating a running average of past gradients, with the momentum coefficient usually set to 0.9. The Adam optimizer combines momentum with per-parameter adaptive learning rates. Standard Adam hyperparameters are a learning rate of 0.001, beta1 of 0.9, and beta2 of 0.999 with a small epsilon of 1e-8.

Data Splitting and Validation

A dataset should be split into training, validation, and test sets. A common split is eighty percent training, ten percent validation, and ten percent test. The test set must be used only once at the very end to estimate generalization; repeated use of the test set for tuning causes information leakage.

Cross-validation is preferred for small datasets. In k-fold cross-validation the data is partitioned into k folds, and the model is trained k times, each time holding out a different fold for validation. The final score is the average across folds. Five-fold and ten-fold are the most common choices.

Bias-Variance Tradeoff

Total expected error decomposes into bias, variance, and irreducible noise. Bias measures how far average predictions deviate from the true values; variance measures the spread of predictions across different training sets. Simple models tend to have high bias and low variance, while complex models tend to have low bias and high variance.

The goal of model selection is to find the sweet spot that minimizes total error. Regularization, ensembling methods such as bagging, and careful feature engineering all shift the balance between bias and variance.

Evaluation Metrics

For classification tasks, precision is the fraction of positive predictions that are correct, while recall is the fraction of actual positives that the model finds. The F1 score is the harmonic mean of precision and recall and is preferred over accuracy when classes are imbalanced. The ROC curve plots the true positive rate against the false positive rate, and the area under it, AUC, summarizes ranking quality across thresholds.

Feature Scaling

Features with very different scales can slow convergence and bias distance-based algorithms. Standardization rescales features to zero mean and unit variance using the z-score transformation. Normalization rescales values into the range zero to one using the min-max transformation. Tree-based models are invariant to monotonic feature scaling, but linear models and neural networks benefit significantly from standardization.
""".strip(),
    },
    {
        "id": "financial_report",
        "title": "Atlas Technologies Annual Financial Report 2025",
        "path": "financial_report_2025.md",
        "content": """
Atlas Technologies Annual Financial Report 2025

Executive Summary

Atlas Technologies finished fiscal year 2025 with total revenue of eight hundred forty-two million dollars, representing year-over-year growth of eighteen percent. The company ended the year with a net revenue retention rate of one hundred twenty-one percent among enterprise customers and achieved positive free cash flow for the third consecutive year.

Fourth quarter revenue was two hundred thirty-five million dollars, exceeding the guidance range of two hundred twenty to two hundred twenty-five million dollars. The outperform was driven by stronger-than-expected expansion in the small business segment and the early success of the AI-assisted analytics add-on, which contributed eleven million dollars of annualized recurring revenue by year end.

Revenue Breakdown by Segment

Software as a service remains the largest revenue stream, contributing sixty-four percent of total revenue or five hundred thirty-nine million dollars. Professional services contributed twenty-one percent, or one hundred seventy-seven million dollars, and perpetual licensing contributed the remaining fifteen percent, or one hundred twenty-six million dollars.

Geographically, the Americas accounted for fifty-two percent of revenue, EMEA for thirty percent, and the Asia-Pacific region for eighteen percent. International revenue grew twenty-six percent year over year, outpacing the domestic growth rate of thirteen percent.

Profitability and Margins

Gross margin for fiscal 2025 was seventy-eight percent, up two percentage points from the prior year, driven by continued optimization of cloud infrastructure spend and favorable mix shift toward higher-margin products. Non-GAAP operating margin reached twenty-three percent, an improvement of five percentage points year over year.

Research and development expenses totaled one hundred ninety-one million dollars, representing twenty-two point seven percent of revenue. Sales and marketing expenses were forty-one percent of revenue, and general and administrative expenses were thirteen percent of revenue.

Customer Metrics

The company closed the year with thirty-four hundred enterprise customers, up from twenty-nine hundred at the end of the prior year, a net addition of five hundred customers. The average contract value for new enterprise deals was eighty-four thousand dollars per year.

Gross revenue churn for the enterprise segment was four point two percent, down from five point one percent in the prior year. The dollar-based net revenue retention rate of one hundred twenty-one percent indicates that the existing customer base expanded revenue by twenty-one percent through upsells and cross-sells.

Balance Sheet and Liquidity

Atlas ended the year with one point two billion dollars in cash, cash equivalents, and marketable securities. The company has no outstanding debt and maintains an undrawn revolving credit facility of five hundred million dollars. Days sales outstanding improved to thirty-one days, down from thirty-eight days in the prior year.

Capital expenditures for the year were one hundred twelve million dollars, primarily related to data center expansion in the Frankfurt and Singapore regions to support European and Asia-Pacific growth.

Outlook for Fiscal 2026

Management guided fiscal 2026 revenue in the range of nine hundred eighty-five million to one point zero two billion dollars, representing growth of approximately seventeen to twenty-one percent over fiscal 2025. Non-GAAP operating margin is expected to expand to twenty-five percent.

Second quarter 2026 guidance calls for revenue between two hundred forty-eight million and two hundred fifty-two million dollars. The company expects to add between six hundred and seven hundred net new enterprise customers during fiscal 2026 and to maintain a net revenue retention rate above one hundred eighteen percent.
""".strip(),
    },
    {
        "id": "nutrition_science",
        "title": "Applied Nutrition Science — Clinical Reference",
        "path": "nutrition_science.md",
        "content": """
Applied Nutrition Science — Clinical Reference

Energy and Macronutrients

Carbohydrates and protein each provide four kilocalories per gram, while fat provides nine kilocalories per gram and alcohol provides seven kilocalories per gram. The Acceptable Macronutrient Distribution Range for adults is forty-five to sixty-five percent of calories from carbohydrates, ten to thirty-five percent from protein, and twenty to thirty-five percent from fat.

Protein requirements depend on activity level. Sedentary adults are advised to consume at least 0.8 grams of protein per kilogram of body weight per day. Athletic populations benefit from higher intakes between 1.6 and 2.2 grams per kilogram per day, with intakes spread across four to six meals to maximize muscle protein synthesis.

Dietary Fiber

Total fiber recommendations are twenty-five grams per day for adult women and thirty-eight grams per day for adult men. Insoluble fiber, found in whole grains and vegetables, adds bulk to stool and promotes regularity. Soluble fiber, found in oats, beans, and certain fruits, forms a gel in the digestive tract, slows glucose absorption, and helps lower LDL cholesterol.

Gradually increasing fiber intake and ensuring adequate fluid consumption reduces the risk of bloating and gastrointestinal discomfort when fiber intake is raised.

Micronutrients and Vitamins

Vitamin D is unique in that the body can synthesize it from sunlight, but dietary intake remains important, particularly during winter months. The recommended dietary allowance is six hundred international units per day for adults up to age seventy and eight hundred international units per day for adults over seventy.

Iron requirements differ substantially by population. Adult men require eight milligrams per day, while adult women aged nineteen to fifty require eighteen milligrams per day. Heme iron from animal sources is absorbed more efficiently than non-heme iron from plants, and vitamin C consumed in the same meal enhances non-heme iron absorption.

Healthy Fats

Unsaturated fats, including monounsaturated and polyunsaturated fatty acids, should make up the majority of dietary fat. The omega-3 fatty acids eicosapentaenoic acid and docosahexaenoic acid, known as EPA and DHA, support cardiovascular and cognitive health. The recommended intake of EPA and DHA combined is two hundred fifty to five hundred milligrams per day, achievable through two servings of fatty fish per week.

Trans fats, produced by partial hydrogenation of vegetable oils, should be avoided entirely. The World Health Organization recommends reducing saturated fat intake to below ten percent of total daily energy.

Hydration

Total water intake recommendations are approximately 3.7 liters per day for adult men and 2.7 liters per day for adult women, including water from food. Requirements increase with physical activity, hot environments, and during pregnancy and lactation. Urine color in the pale yellow range is a practical indicator of adequate hydration.

Meal Timing and Blood Glucose

The glycemic index ranks carbohydrate-containing foods by how quickly they raise blood glucose. Low-glycemic-index foods such as legumes and intact whole grains produce smaller post-meal glucose excursions than refined grains and added sugars. Pairing carbohydrates with protein and fat slows gastric emptying and blunts postprandial glucose spikes.

For individuals with insulin resistance, distributing carbohydrate intake evenly across the day rather than concentrating it in large meals supports better glycemic control.
""".strip(),
    },
    {
        "id": "cloud_architecture",
        "title": "Cloud Architecture Patterns and Best Practices",
        "path": "cloud_architecture.md",
        "content": """
Cloud Architecture Patterns and Best Practices

Microservices and Monoliths

Monolithic applications run all functionality in a single deployable unit, which simplifies development and operations early on but becomes difficult to scale and evolve as teams grow. Microservices decompose the system into independently deployable services, each owning its own data and exposing an API.

The decomposition should follow business capabilities rather than technical layers. Services that change for the same business reason should stay together, while services that change for different reasons should be split. Event-driven communication through a message broker decouples services and allows independent scaling, at the cost of added operational complexity and eventual consistency.

Horizontal Autoscaling

Horizontal scaling adds or removes instances to match demand. Autoscaling policies commonly target a metric such as average CPU utilization around sixty percent, with a cooldown period of five minutes between scaling actions to prevent thrashing. Scaling policies should be based on application-specific metrics such as queue depth or request latency rather than CPU alone for latency-sensitive services.

Pre-warming capacity ahead of known traffic peaks avoids cold start failures. Scale-in should be more conservative than scale-out because removing capacity too aggressively can result in cascading overload.

Data Storage and Databases

Relational databases support ACID transactions and strong consistency, making them the right choice for billing, inventory, and other transactional workloads. Read-heavy applications should add read replicas and use connection pooling to survive connection storms.

As data grows, sharding partitions the database by a shard key such as customer identifier. Shard keys must be chosen to distribute load evenly and avoid hot partitions. NoSQL options such as key-value stores excel at high-throughput reads and writes with eventual consistency, but do not provide multi-row transactions. Software-as-a-service tenants should be isolated using schema-per-tenant or row-level tenancy rather than a dedicated database per tenant, to avoid management overhead.

Caching Strategies

Cache-aside is the most common caching pattern: the application checks the cache first, and on a miss loads from the database and populates the cache. Read-through caches simplify this by loading from the underlying store automatically, while write-through caches update the cache on every write, which increases write latency but keeps the cache fresh.

Time-to-live values depend on data volatility; a common default is ten minutes for reference data such as product catalogs. Cache invalidation must be explicit for authoritative updates. Redis is a popular choice for in-memory caching, and cache clusters should be sized to keep the working set resident in memory with at least twenty percent headroom.

Observability and SLOs

Observability rests on three pillars: logs, metrics, and traces. Logs record discrete events, metrics are aggregated numeric time series, and traces follow requests across distributed services. Distributed tracing implements the W3C trace context standard to propagate a trace id across service boundaries.

Meaningful service level objectives should be expressed as availability targets, with ninety-nine point nine percent uptime a common default for production APIs. Burn rate alerts should page the on-call engineer when error budget is being consumed faster than projected.

Cost Optimization

Spot instances can reduce compute costs by sixty to ninety percent for fault-tolerant and interruptible workloads such as batch processing and CI runners. Right-sizing involves matching instance types to actual utilization patterns; unattended development environments are a leading source of wasted spend.

Reserved capacity or savings plans provide steep discounts for predictable, always-on workloads. Tagging resources by team and environment enables chargeback and cost anomaly detection.
""".strip(),
    },
]


def build_corpus():
    """Write the corpus to evaluation/data/corpus.json (idempotent)."""
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)

    payload = {
        "description": (
            "Deterministic evaluation corpus for RAG-Langchain benchmarks. "
            "Six documents across HR, API, ML, finance, nutrition, and cloud domains."
        ),
        "documents": DOCUMENTS,
    }

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[corpus_builder] Wrote {len(DOCUMENTS)} documents to {CORPUS_PATH}")
    return payload


if __name__ == "__main__":
    build_corpus()