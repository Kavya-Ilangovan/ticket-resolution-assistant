"""Exploratory data analysis -> docs/EXPLORATION.md + docs/img/*.png

    python -m scripts.explore
"""
from __future__ import annotations

import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.core.text import jaccard, tokenize  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.services import indexer, retriever  # noqa: E402
from evals.run_all import stack  # noqa: E402
from scripts.seed import read, seed  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
IMG = DOCS / "img"


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    tickets, queries = read("tickets"), read("eval_queries")
    cats = collections.Counter(t["category"] for t in tickets)
    sev = collections.defaultdict(collections.Counter)
    for t in tickets:
        sev[t["category"]][t["severity"]] += 1
    lens = [len(t["body"].split()) for t in tickets]
    steps = [len(t["resolution_steps"]) for t in tickets]

    # --- vocabulary gap: how much do held-out complaints share with their *own* issue's training tickets?
    by_issue = collections.defaultdict(list)
    for t in tickets:
        by_issue[t["issue_key"]].append(t["body"])
    own, other = [], []
    keys = list(by_issue)
    rng = np.random.default_rng(0)
    for q in queries:
        own.append(max(jaccard(q["text"], b) for b in by_issue[q["issue_key"]]))
        k2 = rng.choice([k for k in keys if k != q["issue_key"]])
        other.append(max(jaccard(q["text"], b) for b in by_issue[k2]))
    zero_overlap = np.mean([len(set(tokenize(q["text"])) & set(tokenize(" ".join(by_issue[q["issue_key"]])))) <= 8 for q in queries])

    # --- similarity distributions (in-domain vs out-of-domain vs unseen class)
    ctx = stack("hashing")
    with session_scope() as db:
        seed(ctx, db)
        info = indexer.get_active(ctx, db)

        def top1(texts):
            return [max(h.score for h in retriever.search_hits(ctx, info, t, 5, {"doc_type": "ticket"}, "dense")) for t in texts]
        ind, ood, nov = (top1([q["text"] for q in read(n)]) for n in ("eval_queries", "ood_queries", "novel_queries"))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = sorted(cats)
    ax[0].barh(names, [cats[n] for n in names], color="#3b6ea5")
    ax[0].set_title("Tickets per category")
    bottom = np.zeros(len(names))
    for s, c in zip(["low", "medium", "high", "critical"], ["#9ccc9c", "#f2d16b", "#f09a4a", "#d9534f"]):
        v = np.array([sev[n][s] for n in names])
        ax[1].barh(names, v, left=bottom, color=c, label=s)
        bottom += v
    ax[1].set_title("Severity mix per category")
    ax[1].legend(fontsize=8)
    ax[1].set_yticklabels([])
    ax[2].hist([ind, nov, ood], bins=15, label=["in-domain", "unseen class", "out-of-domain"], color=["#3b6ea5", "#f2a33b", "#b0b0b0"])
    ax[2].set_title("Top-1 dense similarity (hashing embedder)")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG / "eda_overview.png", dpi=110)

    md = f"""# Exploration (EDA)

Data: **{len(tickets)}** synthetic resolved tickets, **{len(set(t['issue_key'] for t in tickets))}** underlying issues,
**{len(cats)}** categories, 24 KB articles, {len(queries)} held-out eval complaints, 8 out-of-domain probes and a
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
   **{np.mean(own):.2f}** token-Jaccard with their *closest* ticket about the same issue
   (vs {np.mean(other):.2f} for a random ticket of another issue). {zero_overlap:.0%} of held-out complaints share
   <= 8 content tokens with *all* training tickets of their issue. Keyword search therefore plateaus
   (recall@5 of the TF-IDF baseline is only ~0.56, see `evals/reports/report.md`).
2. **Boilerplate dominates raw similarity.** Openers ("Hi team"), closers ("Please advise") and impact sentences
   ("I work from home...") appear in *every* category and make bag-of-words/mean-pooled vectors look alike.
   This motivated (a) hybrid dense+sparse retrieval with server-side IDF, (b) category-aware re-ranking and
   (c) kNN voting weighted by similarity squared rather than a plain majority.
3. **Ticket length:** median {int(np.median(lens))} words (p90 {int(np.percentile(lens, 90))}); resolutions have
   {np.mean(steps):.1f} steps on average, so chunking tickets is unnecessary but KB articles are chunked (<=140 words).
4. **Severity is mostly driven by impact language, not by the technical issue.** The same fault ranges from
   low to critical depending on "work from home", "costing me", duration and repetition cues. This is why severity is
   a transparent cue-scorer rather than a function of category.
5. **Similarity scales differ per embedder.** In-domain top-1 median {np.median(ind):.2f}, unseen-class median
   {np.median(nov):.2f}, out-of-domain median {np.median(ood):.2f} (hashing). The overlap between *unseen class* and
   *in-domain* is large: a similarity threshold alone cannot reliably detect a new class. Hence novelty detection is a
   *workflow* (flag low-confidence complaints -> cluster them in `/v1/admin/emerging` -> an admin adds the class with a few
   seed tickets), not a single cutoff. Thresholds must be re-calibrated whenever the embedding model changes
   (`python -m evals.run_all --calibrate`).
6. **Out-of-domain separation is partial.** Median top-1 similarity is {np.median(ood):.2f} for OOD probes vs
   {np.median(ind):.2f} in-domain, but the *max* OOD score ({max(ood):.2f}) exceeds the in-domain p5
   ({np.percentile(ind, 5):.2f}): probes that share surface words with telecom content ("reset password" -> router admin
   article) slip through a pure threshold. That is why abstention = similarity threshold **plus** an LLM-side
   "sources are insufficient" escape hatch **plus** agent feedback; with a real sentence encoder this gap should shrink
   (to be verified with the sentence-transformers eval run).

## Caveats
* Synthetic text is cleaner and more templated than real tickets; absolute numbers are optimistic for severity/sentiment
  (labels are generated from cues similar to the heuristics) and pessimistic for retrieval on the *held-out wording* split
  (deliberately adversarial). Treat the numbers as a regression harness, then re-measure on real data
  (`python -m data.load_hf`).
"""
    (DOCS / "EXPLORATION.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
