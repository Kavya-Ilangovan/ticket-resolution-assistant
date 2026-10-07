# Exploration (EDA)

Data: **780** synthetic resolved tickets, **36** underlying issues,
**12** categories, 36 KB articles, 96 held-out eval complaints, 8 out-of-domain probes and a
held-out *unseen* class ("5G Home Internet"). 

**How the data is built** (`data/generate_synthetic.py`, seeded). The public datasets are not telecom-specific and have no
"same underlying problem" ground truth, which retrieval evals need, so each synthetic ticket carries an `issue_key`.
Complaints = opener + symptom phrasing + issue-specific detail + optional attempts/impact/outage/duration + sentiment closer +
optional PII. The last two symptom phrasings and the 2nd detail sentence of every issue appear **only** in the eval complaints, so
an eval query describes a known problem in words the index has never seen. `data/load_hf.py` loads the public Hugging Face dataset
instead (not run offline).

![overview](img/eda_overview.png)

## Findings

1. **Vocabulary gap is the core problem.** Held-out complaints describing issue X share on average only
   **0.20** token-Jaccard with their *closest* ticket about the same issue
   (vs 0.16 for a random ticket of another issue). 49% of held-out complaints share
   <= 8 content tokens with *all* training tickets of their issue. Keyword search therefore plateaus
   (recall@5 of the TF-IDF baseline is only ~0.56, see `evals/reports/report.md`).
2. **Boilerplate dominates raw similarity.** Openers ("Hi team"), closers ("Please advise") and impact sentences
   ("I work from home...") appear in *every* category and make bag-of-words/mean-pooled vectors look alike.
   This motivated (a) hybrid dense+sparse retrieval with server-side IDF, (b) category-aware re-ranking and
   (c) kNN voting weighted by similarity squared rather than a plain majority.
3. **Ticket length:** median 27 words (p90 38); resolutions have
   4.9 steps on average, so chunking tickets is unnecessary but KB articles are chunked (<=140 words).
4. **Severity is mostly driven by impact language, not by the technical issue.** The same fault ranges from
   low to critical depending on "work from home", "costing me", duration and repetition cues. This is why severity is
   a transparent cue-scorer rather than a function of category.
5. **Similarity scales differ per embedder.** In-domain top-1 median 0.43, unseen-class median
   0.46, out-of-domain median 0.16 (hashing). The overlap between *unseen class* and
   *in-domain* is large: a similarity threshold alone cannot reliably detect a new class. Hence novelty detection is a
   *workflow* (flag low-confidence complaints -> cluster them in `/v1/admin/emerging` -> an admin adds the class with a few
   seed tickets), not a single cutoff. Thresholds must be re-calibrated whenever the embedding model changes
   (`python -m evals.run_all --calibrate`).
6. **Out-of-domain separation is partial.** Median top-1 similarity is 0.16 for OOD probes vs
   0.43 in-domain, but the *max* OOD score (0.33) exceeds the in-domain p5
   (0.20): probes that share surface words with telecom content ("reset password" -> router admin
   article) slip through a pure threshold. That is why abstention = similarity threshold **plus** an LLM-side
   "sources are insufficient" escape hatch **plus** agent feedback; with a real sentence encoder this gap should shrink
   (to be verified with the sentence-transformers eval run).

## Caveats
* Synthetic text is cleaner and more templated than real tickets; absolute numbers are optimistic for severity/sentiment
  (labels are generated from cues similar to the heuristics) and pessimistic for retrieval on the *held-out wording* split
  (deliberately adversarial). Treat the numbers as a regression harness, then re-measure on real data
  (`python -m data.load_hf`).
