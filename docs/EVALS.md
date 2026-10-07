# Evals on system health

Run `python -m evals.run_all` (offline, writes `evals/reports/report.md`). `--gates` exits 1 on regression (used in CI);
`--calibrate` prints abstain/novelty thresholds for the chosen embedder (similarity scales differ per model; re-calibrate after every re-index).

| Section | What is measured |
|---|---|
| Hand-written set | the same retrieval, parsing and step metrics on 60 complaints not produced by the generator's templates |
| Retrieval | recall@1/5, MRR, nDCG@5 for dense, **hybrid (used)** and a TF-IDF **keyword baseline** (today's workflow); relevant = same `issue_key`. KB recall@3 |
| Parsing | category accuracy/macro-F1, product, severity (exact, ±1), sentiment |
| RAG | citation validity (cited ids ⊆ retrieved sources), grounding score, step precision / gold-step recall |
| Abstention | out-of-domain complaints must escalate |
| Evolving classes | unseen class → flagged → added at runtime (no retraining) → accuracy; base classes must not regress |
| Lifecycle | idempotent re-ingest, edit-not-duplicate, blue/green re-index, KB versioning/retirement, PII redaction, cache |
| Latency | p50/p95 in-process (excludes network and LLM) |

## Measured (hashing embedder, synthetic data and hand-written sets)
Full tables: `evals/reports/report.md`. Headline numbers:

| retrieval | dense | hybrid (used) | keyword |
|---|---|---|---|
| synthetic (144, unseen wording) recall@1 / recall@5 / MRR | 0.26 / 0.44 / 0.35 | 0.29 / 0.56 / 0.40 | 0.31 / 0.54 / 0.40 |
| hand-written (60) recall@1 / recall@5 / MRR | 0.78 / 0.95 / 0.84 | 0.80 / 0.93 / 0.86 | 0.78 / 0.92 / 0.84 |

| | synthetic | hand-written |
|---|---|---|
| KB recall@3 | 0.74 | 0.82 |
| category accuracy | 0.38 | 0.78 |
| product accuracy | 0.57 | 0.93 |
| sentiment accuracy | 0.77 | 0.68 |
| severity within one level | 0.94 | 0.88 |
| step precision / gold-step recall | 0.33 / 0.35 | 0.79 / 0.84 |

Short vague queries (29): on-topic answers 0.86, top-1 on topic 0.86. Free-text fix agreement: same fix worded differently 0.69, different fixes 0.25.
Calibration (204 complaints): match ECE 0.06, category ECE 0.08, monotonic; off-topic mean confidence 0.20. Out-of-domain escalated 0.88, citation validity 1.00, p95 57 ms.

**Read this honestly.**
* The hashing embedder is a lexical CI fallback; on the hard split it ties keyword search. These numbers show the plumbing works, not that semantic search wins. Run `python -m evals.run_all --backend sentence-transformers --rerank --gates`; with `--rerank` the report adds a hybrid vs hybrid + cross-encoder table.
* The hand-written and short-query sets are easier, written by one author, and their labels are subjective.
* Category accuracy on held-out synthetic data fell as the taxonomy grew from 8 to 12 classes with overlapping vocabulary (voice vs mobile network, fraud vs SIM); a semantic embedder is expected to recover most of this.
* Step precision and recall are for the extractive generator. `python -m evals.llm_judge` grades the LLM path (faithfulness of each step to its citations, 1-5 relevance) but needs an API key and has only been tested against a mock.
* Known weak spots: novel-class flagging before the class exists is low; the medium confidence band is over-confident; complaints that resemble a known problem in surface wording but not intent can be answered confidently.

## Online health (Prometheus `/metrics`, rules in `monitoring/alerts.yml`)
| Signal | Meaning |
|---|---|
| `http_request_duration_seconds{route}`, `pipeline_stage_duration_seconds{stage}` | latency, per stage |
| `retrieval_top_score` | top-1 similarity distribution = embedding/data **drift** |
| `resolve_escalations_total`, `resolve_unknown_category_total` | no grounded answer / **new problem types** |
| `llm_calls_total`, `llm_fallback_total`, `llm_circuit_breaker_open` | provider health |
| `ungrounded_steps_dropped_total`, `resolution_grounding_score` | hallucination-guard activity |
| `feedback_total{helpful}` | agent-perceived quality |
| `unindexed_documents`, `celery_queue_depth`, `index_documents` | pipeline backlog |

`GET /v1/admin/drift` summarises the same signals from the query log; `GET /v1/admin/emerging` clusters unmatched complaints.
