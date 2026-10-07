# Eval report (hashing, 144 held-out complaints)

## Retrieval (relevant = same underlying issue)

| metric | dense | hybrid (used) | keyword baseline |
|---|---|---|---|
| recall@1 | 0.264 | 0.285 | 0.306 |
| recall@5 | 0.444 | 0.562 | 0.542 |
| mrr | 0.353 | 0.404 | 0.4 |
| ndcg@5 | 0.215 | 0.275 | 0.306 |

## Short / vague complaints (29, e.g. 'wifi not working')

What the agent *sees* must be on topic. `off_topic_free` = share of queries with no result from a forbidden department.

## Short queries

| metric | value |
|---|---|
| precision@5 | 0.737 |
| top1 | 0.862 |
| off_topic_free | 0.862 |
| avg_sources_shown | 5.897 |

## Are the confidence scores honest? (204 complaints)

Match confidence = probability that the leading fix group is the right issue. Calibration error (ECE) is the average gap between stated confidence and observed accuracy.

## Calibration

| metric | value |
|---|---|
| match_ece | 0.059 |
| category_ece | 0.075 |
| high_conf_precision | 0.77 |
| low_conf_precision | 0.2 |
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
| 0.0-0.4 | 96 | 0.26 | 0.2 |
| 0.4-0.7 | 34 | 0.52 | 0.56 |
| 0.7-1.0 | 74 | 0.86 | 0.77 |

**Category confidence vs correctness of the category (known classes only)**

| stated confidence | n | mean stated | actually right |
|---|---|---|---|
| 0.0-0.4 | 76 | 0.2 | 0.3 |
| 0.4-0.7 | 32 | 0.52 | 0.41 |
| 0.7-1.0 | 71 | 0.92 | 0.92 |


## Hand-written complaints (60, not template-generated)

| metric | dense | hybrid (used) | keyword baseline |
|---|---|---|---|
| recall@1 | 0.783 | 0.8 | 0.783 |
| recall@5 | 0.95 | 0.933 | 0.917 |
| mrr | 0.842 | 0.858 | 0.837 |
| ndcg@5 | 0.725 | 0.796 | 0.746 |

## Hand-written: parsing and KB

| metric | value |
|---|---|
| kb_recall@3 | 0.817 |
| category_acc | 0.783 |
| product_acc | 0.933 |
| severity_within1 | 0.883 |
| sentiment_acc | 0.683 |
| step_precision | 0.793 |
| gold_step_recall | 0.842 |

## Parsing, RAG, abstention, evolving classes, latency

| metric | value |
|---|---|
| kb_recall@3 | 0.743 |
| category_acc | 0.375 |
| category_macro_f1 | 0.365 |
| product_acc | 0.569 |
| severity_acc | 0.396 |
| severity_within1 | 0.944 |
| sentiment_acc | 0.771 |
| citation_validity | 1.0 |
| mean_grounding | 0.979 |
| step_precision | 0.329 |
| gold_step_recall | 0.347 |
| escalated_in_domain | 0.021 |
| ood_escalated | 0.875 |
| novel_flagged_before | 0.167 |
| emerging_clusters | 0 |
| novel_after_acc | 0.222 |
| base_acc_before | 0.417 |
| base_acc_after | 0.396 |
| seconds_to_learn | 0.09 |
| p50_ms | 47.2 |
| p95_ms | 55.7 |
| throughput_rps_1thread | 20.5 |

## Gates

| gate | value | threshold | |
|---|---|---|---|
| hybrid.recall@5 | 0.562 | 0.42 | PASS |
| hybrid.mrr | 0.404 | 0.35 | PASS |
| kb_recall@3 | 0.743 | 0.6 | PASS |
| category_acc | 0.375 | 0.33 | PASS |
| product_acc | 0.569 | 0.48 | PASS |
| sentiment_acc | 0.771 | 0.6 | PASS |
| citation_validity | 1.0 | 1.0 | PASS |
| step_precision | 0.329 | 0.25 | PASS |
| gold_step_recall | 0.347 | 0.3 | PASS |
| ood_escalated | 0.875 | 0.75 | PASS |
| novel_after_acc | 0.222 | 0.2 | PASS |
| p95_ms | 55.7 | 500 | PASS |
| hand.hybrid.recall@5 | 0.933 | 0.9 | PASS |
| hand.hybrid.mrr | 0.858 | 0.85 | PASS |
| hand.kb_recall@3 | 0.817 | 0.8 | PASS |
| hand.category_acc | 0.783 | 0.75 | PASS |
| hand.product_acc | 0.933 | 0.85 | PASS |
| hand.step_precision | 0.793 | 0.65 | PASS |
| hand.gold_step_recall | 0.842 | 0.75 | PASS |
| short.precision@5 | 0.737 | 0.7 | PASS |
| short.top1 | 0.862 | 0.8 | PASS |
| short.off_topic_free | 0.862 | 0.8 | PASS |
| calib.match_ece | 0.059 | 0.12 | PASS |
| calib.category_ece | 0.075 | 0.1 | PASS |
| calib.high_conf_precision | 0.77 | 0.75 | PASS |
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
