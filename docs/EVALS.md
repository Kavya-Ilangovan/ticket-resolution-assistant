# Eval report (hashing, 96 held-out complaints)

## Retrieval (relevant = same underlying issue)

| metric | dense | hybrid (used) | keyword baseline |
|---|---|---|---|
| recall@1 | 0.281 | 0.323 | 0.344 |
| recall@5 | 0.552 | 0.604 | 0.562 |
| mrr | 0.394 | 0.44 | 0.445 |
| ndcg@5 | 0.267 | 0.312 | 0.349 |

## Short / vague complaints (29, e.g. 'wifi not working')

What the agent *sees* must be on topic. `off_topic_free` = share of queries with no result from a forbidden department.

## Short queries

| metric | value |
|---|---|
| precision@5 | 0.739 |
| top1 | 0.862 |
| off_topic_free | 0.862 |
| avg_sources_shown | 6.207 |

## Are the confidence scores honest? (132 complaints)

Match confidence = probability that the leading fix group is the right issue. Calibration error (ECE) is the average gap between stated confidence and observed accuracy.

## Calibration

| metric | value |
|---|---|
| match_ece | 0.06 |
| category_ece | 0.041 |
| high_conf_precision | 0.92 |
| low_conf_precision | 0.19 |
| monotonic | True |
| ood_mean_confidence | 0.195 |

Free-text answers (same fix worded differently vs genuinely different fixes):

## Agreement on free-text answers

| metric | value |
|---|---|
| same_fix_agreement | 0.688 |
| different_fix_agreement | 0.25 |

**Match confidence vs correctness of the leading fix**

| stated confidence | n | mean stated | actually right |
|---|---|---|---|
| 0.0-0.4 | 59 | 0.26 | 0.19 |
| 0.4-0.7 | 23 | 0.56 | 0.35 |
| 0.7-1.0 | 50 | 0.87 | 0.92 |

**Category confidence vs correctness of the category (known classes only)**

| stated confidence | n | mean stated | actually right |
|---|---|---|---|
| 0.0-0.4 | 39 | 0.26 | 0.28 |
| 0.4-0.7 | 27 | 0.56 | 0.48 |
| 0.7-1.0 | 57 | 0.92 | 0.93 |


## Hand-written complaints (36, not template-generated)

| metric | dense | hybrid (used) | keyword baseline |
|---|---|---|---|
| recall@1 | 0.861 | 0.861 | 0.861 |
| recall@5 | 0.972 | 0.972 | 0.917 |
| mrr | 0.912 | 0.903 | 0.886 |
| ndcg@5 | 0.809 | 0.845 | 0.803 |

## Hand-written: parsing and KB

| metric | value |
|---|---|
| kb_recall@3 | 0.861 |
| category_acc | 0.861 |
| product_acc | 0.917 |
| severity_within1 | 0.861 |
| sentiment_acc | 0.667 |
| step_precision | 0.821 |
| gold_step_recall | 0.883 |

## Parsing, RAG, abstention, evolving classes, latency

| metric | value |
|---|---|
| kb_recall@3 | 0.75 |
| category_acc | 0.479 |
| category_macro_f1 | 0.434 |
| product_acc | 0.552 |
| severity_acc | 0.417 |
| severity_within1 | 0.938 |
| sentiment_acc | 0.729 |
| citation_validity | 1.0 |
| mean_grounding | 1.0 |
| step_precision | 0.338 |
| gold_step_recall | 0.362 |
| escalated_in_domain | 0.0 |
| ood_escalated | 0.875 |
| novel_flagged_before | 0.167 |
| emerging_clusters | 0 |
| novel_after_acc | 0.333 |
| base_acc_before | 0.396 |
| base_acc_after | 0.375 |
| seconds_to_learn | 0.14 |
| p50_ms | 70.7 |
| p95_ms | 93.0 |
| throughput_rps_1thread | 13.6 |

## Gates

| gate | value | threshold | |
|---|---|---|---|
| hybrid.recall@5 | 0.604 | 0.42 | PASS |
| hybrid.mrr | 0.44 | 0.35 | PASS |
| kb_recall@3 | 0.75 | 0.6 | PASS |
| category_acc | 0.479 | 0.38 | PASS |
| product_acc | 0.552 | 0.48 | PASS |
| sentiment_acc | 0.729 | 0.6 | PASS |
| citation_validity | 1.0 | 1.0 | PASS |
| step_precision | 0.338 | 0.25 | PASS |
| gold_step_recall | 0.362 | 0.3 | PASS |
| ood_escalated | 0.875 | 0.75 | PASS |
| novel_after_acc | 0.333 | 0.25 | PASS |
| p95_ms | 93.0 | 500 | PASS |
| hand.hybrid.recall@5 | 0.972 | 0.9 | PASS |
| hand.hybrid.mrr | 0.903 | 0.85 | PASS |
| hand.kb_recall@3 | 0.861 | 0.8 | PASS |
| hand.category_acc | 0.861 | 0.75 | PASS |
| hand.product_acc | 0.917 | 0.85 | PASS |
| hand.step_precision | 0.821 | 0.65 | PASS |
| hand.gold_step_recall | 0.883 | 0.75 | PASS |
| short.precision@5 | 0.739 | 0.7 | PASS |
| short.top1 | 0.862 | 0.8 | PASS |
| short.off_topic_free | 0.862 | 0.8 | PASS |
| calib.match_ece | 0.06 | 0.12 | PASS |
| calib.category_ece | 0.041 | 0.1 | PASS |
| calib.high_conf_precision | 0.92 | 0.8 | PASS |
| calib.ood_mean_confidence | 0.195 | 0.2 | PASS |
| freetext.same_fix_agreement | 0.688 | 0.6 | PASS |
| freetext.different_fix_agreement | 0.25 | 0.35 | PASS |
| lifecycle.reingest_is_idempotent | True | True | PASS |
| lifecycle.edit_updates_not_duplicates | True | True | PASS |
| lifecycle.reindex_swaps_and_preserves_results | True | True | PASS |
| lifecycle.kb_update_replaces_chunks | True | True | PASS |
| lifecycle.retired_kb_not_retrievable | True | True | PASS |
| lifecycle.pii_redacted_in_logs | True | True | PASS |
| lifecycle.cache_hit_on_repeat | True | True | PASS |
| calib.monotonic | True | True | PASS |

gates: 36/36 passed