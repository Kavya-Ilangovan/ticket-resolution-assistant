"""Offline evals and quality gates.

  python -m evals.run_all                      # hashing embedder (offline), writes evals/reports/report.md
  python -m evals.run_all --backend sentence-transformers --gates
  python -m evals.run_all --calibrate          # print abstain/novelty thresholds for the chosen embedder
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from app.config import Settings
from app.core.context import build_context, set_context
from app.core.reranker import load_reranker, rerank_hits
from app.core.text import jaccard
from app.db.models import QueryLog
from app.db.session import init_engine, session_scope
from app.schemas import KBArticleIn, TicketIn
from app.services import drift, indexer, ingestion, resolver, retriever, taxonomy
from scripts.seed import read, seed

REPORTS = Path(__file__).parent / "reports"   # one report per embedding backend, so they never overwrite each other
SEV = ["low", "medium", "high", "critical"]
MODELS = {"sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2", "hashing": "hashing"}

# hashing: regression floors (measured minus a margin) for the offline lexical embedder.
# sentence-transformers: quality gates; hybrid retrieval must beat the keyword baseline by a margin.
GATES = {
    "hashing": {"hybrid.recall@5": 0.42, "hybrid.mrr": 0.35, "kb_recall@3": 0.60, "category_acc": 0.33, "product_acc": 0.48,
                "sentiment_acc": 0.60, "citation_validity": 1.0, "step_precision": 0.25, "gold_step_recall": 0.30,
                "ood_escalated": 0.75, "novel_after_acc": 0.20, "p95_ms": 500,
                "hand.hybrid.recall@5": 0.90, "hand.hybrid.mrr": 0.85, "hand.kb_recall@3": 0.80, "hand.category_acc": 0.75,
                "hand.product_acc": 0.85, "hand.step_precision": 0.65, "hand.gold_step_recall": 0.75,
                "short.precision@5": 0.70, "short.top1": 0.80, "short.off_topic_free": 0.80,
                "calib.match_ece": 0.12, "calib.category_ece": 0.10, "calib.high_conf_precision": 0.75,
                "calib.ood_mean_confidence": 0.20, "freetext.same_fix_agreement": 0.60, "freetext.different_fix_agreement": 0.35},
    "sentence-transformers": {"hybrid_vs_keyword.recall@5": 0.03, "hybrid_vs_keyword.mrr": 0.03,
                              "hand.hybrid_vs_keyword.recall@5": 0.03, "hand.hybrid_vs_keyword.mrr": 0.03,
                              "kb_recall@3": 0.80, "category_acc": 0.60, "hand.category_acc": 0.60,
                              "citation_validity": 1.0, "sentiment_acc": 0.60, "ood_escalated": 0.75, "p95_ms": 800,
                              "short.precision@5": 0.70, "short.off_topic_free": 0.80, "calib.match_ece": 0.15,
                              "calib.high_conf_precision": 0.80, "calib.ood_mean_confidence": 0.20,
                              "freetext.same_fix_agreement": 0.60, "freetext.different_fix_agreement": 0.35},
}


LOWER_IS_BETTER = {"p95_ms", "calib.match_ece", "calib.category_ece", "calib.ood_mean_confidence", "freetext.different_fix_agreement"}


def stack(backend: str):
    s = Settings(database_url="sqlite:///:memory:", qdrant_url=":memory:", embedding_backend=backend,
                 embedding_model=MODELS[backend], celery_eager=True, openrouter_api_key=None)
    init_engine(s)
    ctx = build_context(s)
    set_context(ctx)
    return ctx


def avg(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def rank_metrics(rows):  # rows: list of boolean relevance lists, best first
    rr = lambda r: next((1 / (i + 1) for i, x in enumerate(r) if x), 0.0)  # noqa: E731
    ndcg = lambda r: sum(x / math.log2(i + 2) for i, x in enumerate(r[:5])) / sum(1 / math.log2(i + 2) for i in range(5))  # noqa: E731
    return {"recall@1": avg(r[0] for r in rows), "recall@5": avg(any(r[:5]) for r in rows), "mrr": avg(rr(r) for r in rows),
            "ndcg@5": avg(ndcg(r) for r in rows)}


def retrieval(ctx, db, queries, reranker=None):
    info = indexer.get_active(ctx, db)
    rel = {"dense": [], "hybrid": []}
    kb = []
    for q in queries:
        for mode in ("dense", "hybrid"):
            hits = retriever.search_hits(ctx, info, q["text"], 10, {"doc_type": "ticket"}, mode)
            rel[mode].append([h.payload["issue_key"] == q["issue_key"] for h in hits])
            if reranker and mode == "hybrid":
                reordered = rerank_hits(reranker, q["text"], hits)
                rel.setdefault("hybrid+rerank", []).append([h.payload["issue_key"] == q["issue_key"] for h in reordered])
        top = retriever.search_hits(ctx, info, q["text"], 3, {"doc_type": "kb"})
        kb.append(any(h.payload["external_id"] == f"kb-{q['issue_key']}" for h in top))
    tickets = read("tickets")  # keyword baseline = today's agent workflow (lexical TF-IDF)
    tf = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = tf.fit_transform([t["body"] for t in tickets])
    sims = (tf.transform([q["text"] for q in queries]) @ X.T).toarray()
    rel["keyword"] = [[tickets[j]["issue_key"] == q["issue_key"] for j in np.argsort(-sims[i])[:10]] for i, q in enumerate(queries)]
    out = {k: rank_metrics(v) for k, v in rel.items()}
    out["hybrid_vs_keyword"] = {k: round(out["hybrid"][k] - out["keyword"][k], 3) for k in out["hybrid"]}
    return out, avg(kb)


def parsing(queries, outs):
    t = lambda k: [q[k] for q in queries]  # noqa: E731
    sev = [(SEV.index(q["severity"]), SEV.index(o.analysis.severity)) for q, o in zip(queries, outs)]
    cats = [o.analysis.category for o in outs]
    return {"category_acc": avg(a == b for a, b in zip(t("category"), cats)),
            "category_macro_f1": round(float(f1_score(t("category"), cats, average="macro")), 3),
            "product_acc": avg(q["product"] == o.analysis.product for q, o in zip(queries, outs)),
            "severity_acc": avg(a == b for a, b in sev), "severity_within1": avg(abs(a - b) <= 1 for a, b in sev),
            "sentiment_acc": avg(q["sentiment"] == o.analysis.sentiment for q, o in zip(queries, outs))}


def rag(queries, outs):
    valid, prec, rec = [], [], []
    for q, o in zip(queries, outs):
        steps, ids = o.resolution.steps, {s.id for s in o.sources}
        valid.append(all(st.citations and set(st.citations) <= ids for st in steps))
        match = lambda a, b: jaccard(a, b) >= 0.6  # noqa: E731
        prec.append(avg(any(match(st.text, g) for g in q["gold_steps"]) for st in steps) if steps else 0.0)
        rec.append(avg(any(match(st.text, g) for st in steps) for g in q["gold_steps"]) if steps else 0.0)
    return {"citation_validity": avg(valid), "mean_grounding": avg(o.resolution.grounding_score for o in outs),
            "step_precision": avg(prec), "gold_step_recall": avg(rec), "escalated_in_domain": avg(o.resolution.escalate for o in outs)}


def evolving(backend, base_queries):
    """Hold a whole class out -> flag it -> add it at runtime (no retraining) -> classify it."""
    ctx = stack(backend)
    novel = read("novel_queries")
    res = lambda db, qs: [resolver.resolve(ctx, db, q["text"], "eval", use_cache=False) for q in qs]  # noqa: E731
    with session_scope() as db:
        seed(ctx, db)
        before = res(db, novel)
        base_before = avg(o.analysis.category == q["category"] for q, o in zip(base_queries[:48], res(db, base_queries[:48])))
        clusters = drift.emerging_topics(ctx, db, min_size=3)
        taxonomy.ensure_category(db, "5G Home Internet", created_by="admin")
        t0 = time.perf_counter()
        ingestion.ingest_tickets(ctx, db, [TicketIn(**r) for r in read("novel_tickets")])
        ingestion.ingest_kb(ctx, db, [KBArticleIn(**r) for r in read("novel_kb")])
        learn_s = time.perf_counter() - t0
        after = res(db, novel)
        base_after = avg(o.analysis.category == q["category"] for q, o in zip(base_queries[:48], res(db, base_queries[:48])))
    return {"novel_flagged_before": avg(not o.analysis.is_known_category for o in before), "emerging_clusters": len(clusters),
            "novel_after_acc": avg(o.analysis.category == q["category"] for q, o in zip(novel, after)),
            "base_acc_before": base_before, "base_acc_after": base_after, "seconds_to_learn": round(learn_s, 2)}


def lifecycle(ctx, db):
    checks, tix = {}, [TicketIn(**r) for r in read("tickets")[:50]]
    checks["reingest_is_idempotent"] = ingestion.ingest_tickets(ctx, db, tix) == {"inserted": 0, "updated": 0, "skipped": 50}
    edited = tix[0].model_copy(update={"resolution_steps": tix[0].resolution_steps + ["Extra step."]})
    r = ingestion.ingest_tickets(ctx, db, [edited])
    checks["edit_updates_not_duplicates"] = (r["updated"], r["inserted"]) == (1, 0)
    q = read("eval_queries")[0]["text"]
    pre = {x.external_id for x in resolver.search(ctx, db, q, 5, "ticket").results}
    swap = indexer.reindex(ctx, db)
    post = {x.external_id for x in resolver.search(ctx, db, q, 5, "ticket").results}
    checks["reindex_swaps_and_preserves_results"] = swap["old"] != swap["new"] and len(pre & post) >= 3
    kb = KBArticleIn(**read("kb")[0])
    kb.version, kb.body = 2, kb.body + "\n\nUpdate: new firmware 3.2 fixes this."
    ingestion.ingest_kb(ctx, db, [kb])
    checks["kb_update_replaces_chunks"] = resolver.search(ctx, db, "firmware 3.2 fixes this", 3, "kb").results[0].external_id == kb.external_id
    kb.active = False
    ingestion.ingest_kb(ctx, db, [kb])
    checks["retired_kb_not_retrievable"] = all(x.external_id != kb.external_id for x in resolver.search(ctx, db, kb.title, 5, "kb").results)
    p = resolver.resolve(ctx, db, "Router keeps rebooting, call 98765 43210 or mail raj@example.com", "eval")
    checks["pii_redacted_in_logs"] = not any(x in db.get(QueryLog, p.query_id).text for x in ("98765", "@example.com"))
    for cq in read("eval_queries")[:20]:  # escalations are deliberately not cached
        a = resolver.resolve(ctx, db, cq["text"], "eval")
        if not a.resolution.escalate:
            checks["cache_hit_on_repeat"] = not a.cached and resolver.resolve(ctx, db, cq["text"], "eval").cached
            break
    return checks


def calibrate(ctx, db, queries):
    info = indexer.get_active(ctx, db)
    top1 = lambda texts: np.array([max(h.score for h in retriever.search_hits(ctx, info, t, 5, {"doc_type": "ticket"}, "dense")) for t in texts])  # noqa: E731
    ind, ood, nov = (top1([q["text"] for q in qs]) for qs in (queries, read("ood_queries"), read("novel_queries")))
    p = lambda x, q: round(float(np.percentile(x, q)), 3)  # noqa: E731
    print(f"in-domain top1 p5={p(ind, 5)} p25={p(ind, 25)} p50={p(ind, 50)} | out-of-domain max={ood.max():.3f} p50={p(ood, 50)} "
          f"| unseen class p50={p(nov, 50)}")
    _refit(ctx, db)
    print(f"suggested: ABSTAIN_THRESHOLD={(ood.max() + p(ind, 5)) / 2 if p(ind, 5) > ood.max() else p(ood, 75):.3f} "
          f"NOVELTY_THRESHOLD={(p(nov, 50) + p(ind, 25)) / 2:.3f}  (review the overlap before adopting)")


def handwritten(ctx, db):
    """Same metrics on 36 complaints written by hand (data/handwritten): not produced by the synthetic generator's
    templates, so this is the check against 'the system only learned my own templates'."""
    qs = read("realistic_queries", "handwritten")
    ret, kb_recall = retrieval(ctx, db, qs)
    outs = [resolver.resolve(ctx, db, q["text"], "eval", use_cache=False) for q in qs]
    p = parsing(qs, outs)
    gold: dict[str, list[str]] = {}   # gold steps of an issue = every distinct resolution step used by its resolved tickets
    for t in read("tickets"):
        gold.setdefault(t["issue_key"], [])
        gold[t["issue_key"]] += [x for x in t["resolution_steps"] if x not in gold[t["issue_key"]]]
    r = rag([{**q, "gold_steps": gold[q["issue_key"]]} for q in qs], outs)
    return {"n": len(qs), "ret": ret, "kb_recall@3": kb_recall,
            **{k: p[k] for k in ("category_acc", "product_acc", "severity_within1", "sentiment_acc")},
            "step_precision": r["step_precision"], "gold_step_recall": r["gold_step_recall"]}


def short_queries(ctx, db):
    """Vague 2-4 word complaints ("wifi not working"). What the agent SEES must be on topic: the share of results from an
    unrelated department is the failure the project had (billing articles under a Wi-Fi complaint)."""
    qs, issue = read("short_queries", "handwritten"), {t["external_id"]: t["issue_key"] for t in read("tickets")}
    p5, top1, clean, n_src = [], [], [], []
    for q in qs:
        o = resolver.resolve(ctx, db, q["text"], "eval", use_cache=False)
        tix, ok = [x for x in o.sources if x.doc_type == "ticket"], set(q["acceptable_issues"])
        p5.append(avg(issue.get(x.external_id) in ok for x in tix) if tix else 0.0)
        top1.append(bool(tix) and issue.get(tix[0].external_id) in ok)
        clean.append(all(x.category not in set(q["forbidden_categories"]) for x in o.sources))
        n_src.append(len(o.sources))
    return {"n": len(qs), "precision@5": avg(p5), "top1": avg(top1), "off_topic_free": avg(clean), "avg_sources_shown": avg(n_src)}


def free_text_agreement(ctx):
    """Real agents word the same fix differently ("restart the router" / "power cycle the modem"). Agreement must be HIGH
    for answers that mean the same and LOW for answers that prescribe different fixes (data/handwritten/paraphrased_fixes.json,
    4 problems x 4 differently worded answers)."""
    from app.core.vectorstore import Hit
    from app.services import confidence as conf
    fixes = json.loads((Path(__file__).resolve().parent.parent / "data" / "handwritten" / "paraphrased_fixes.json").read_text(encoding="utf-8"))
    emb = ctx.embedders.get(ctx.settings.embedding_backend, ctx.settings.embedding_model)
    mk = lambda lst: [Hit(id=str(i), score=0.6, payload={"resolution_steps": a, "category": "X", "title": "t", "doc_type": "ticket"}) for i, a in enumerate(lst)]  # noqa: E731
    same = [conf.assess(mk(v), [], ctx.settings, embed=emb.embed)[0].agreement for v in fixes.values()]
    diff = conf.assess(mk([v[0] for v in fixes.values()]), [], ctx.settings, embed=emb.embed)[0].agreement
    return {"same_fix_agreement": round(float(np.mean(same)), 3), "different_fix_agreement": round(diff, 3)}


def _ece(p, y, bins=5):
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & (p < lo + 1 / bins + 1e-9)
        e += m.mean() * abs(p[m].mean() - y[m].mean()) if m.any() else 0.0
    return round(float(e), 3)


def _bins(p, y, edges=(0, 0.4, 0.7, 1.01)):
    rows = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        m = [(a, b) for a, b in zip(p, y, strict=True) if lo <= a < hi]
        rows.append((f"{lo:.1f}-{min(hi, 1):.1f}", len(m), round(float(np.mean([a for a, _ in m])), 2) if m else None,
                     round(float(np.mean([b for _, b in m])), 2) if m else None))
    return rows


def calibration(ctx, db, outputs=None):
    """Are the confidence numbers honest? Compare stated confidence with how often the answer was actually right.
    match confidence -> was the leading fix group the right issue?   category confidence -> was the category right?"""
    issue = {t["external_id"]: t["issue_key"] for t in read("tickets")}
    pm, ym, pc, yc = [], [], [], []
    for qs in (read("eval_queries"), read("realistic_queries", "handwritten")):
        for q in qs:
            o = resolver.resolve(ctx, db, q["text"], "eval", use_cache=False)
            lead = next((x for x in o.sources if x.id in o.confidence.source_ids and x.doc_type == "ticket"), None)
            pm.append(o.confidence.score)
            ym.append(lead is not None and issue.get(lead.external_id) == q["issue_key"])
            if o.analysis.is_known_category:
                pc.append(o.analysis.category_confidence)
                yc.append(o.analysis.category == q["category"])
    ood = [resolver.resolve(ctx, db, q["text"], "eval", use_cache=False).confidence.score for q in read("ood_queries")]
    b = _bins(pm, ym)
    hi, lo = b[-1], b[0]
    mono = all((b[i][3] or 0) <= (b[i + 1][3] or 1) + 1e-9 for i in range(len(b) - 1) if b[i][1] and b[i + 1][1])
    return {"match_ece": _ece(pm, ym), "category_ece": _ece(pc, yc), "high_conf_precision": hi[3] or 0.0, "low_conf_precision": lo[3] or 0.0,
            "monotonic": mono, "ood_mean_confidence": round(float(np.mean(ood)), 3), "n": len(pm),
            "_bins_match": b, "_bins_category": _bins(pc, yc), "_raw": (pm, ym, pc, yc)}


def _refit(ctx, db):
    """Print logistic refits of the two confidence calibrations for the CURRENT embedder."""
    from sklearn.linear_model import LogisticRegression
    issue = {t["external_id"]: t["issue_key"] for t in read("tickets")}
    ag, ym, raw, yc = [], [], [], []
    sl, bi = ctx.settings.cat_conf_slope, ctx.settings.cat_conf_bias
    for qs in (read("eval_queries"), read("realistic_queries", "handwritten")):
        for q in qs:
            o = resolver.resolve(ctx, db, q["text"], "eval", use_cache=False)
            lead = next((x for x in o.sources if x.id in o.confidence.source_ids and x.doc_type == "ticket"), None)
            ag.append([o.confidence.agreement])
            ym.append(lead is not None and issue.get(lead.external_id) == q["issue_key"])
            if o.analysis.is_known_category:
                p = min(max(o.analysis.category_confidence, 1e-4), 1 - 1e-4)
                raw.append([(math.log(p / (1 - p)) - bi) / sl])
                yc.append(o.analysis.category == q["category"])
    original, rows = ctx.settings, []
    try:
        for tau in (1.01, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2):
            ctx.settings = original.model_copy(update={"fix_link_threshold": tau})
            ft = free_text_agreement(ctx)
            rows.append((tau, calibration(ctx, db)["match_ece"], ft["same_fix_agreement"], ft["different_fix_agreement"]))
    finally:
        ctx.settings = original
    print("\nFIX_LINK_THRESHOLD sweep (telecom match ECE must stay near the no-semantic-link baseline; same-fix agreement should be high; different-fix low):")
    for tau, ece, same, diff in rows:
        print(f"   {tau:>4}: match_ece={ece:.3f}  same_fix_agreement={same:.2f}  different_fix_agreement={diff:.2f}")
    base = rows[0][1]   # threshold > 1 = no semantic linking = the baseline
    ok = [r for r in rows if r[0] < 1 and r[1] <= base + 0.02 and r[3] <= 0.35]
    if ok:
        print(f"suggested: FIX_LINK_THRESHOLD={min(ok, key=lambda r: r[0])[0]}   (lowest threshold that keeps calibration within 0.02 of baseline {base:.3f})")
    for name, X, y, keys in (("match confidence", ag, ym, ("CONF_SLOPE", "CONF_BIAS")),
                             ("category confidence", raw, yc, ("CAT_CONF_SLOPE", "CAT_CONF_BIAS"))):
        lr = LogisticRegression(C=100).fit(X, y)
        print(f"suggested ({name}): {keys[0]}={float(lr.coef_[0][0]):.2f} {keys[1]}={float(lr.intercept_[0]):.2f}   (n={len(y)})")


def table(title, d):
    return f"## {title}\n\n| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in d.items()) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hashing", choices=list(MODELS))
    ap.add_argument("--gates", action="store_true", help="exit 1 if a quality gate fails")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="also report a cross-encoder reranking arm (needs requirements-ml.txt)")
    a = ap.parse_args()
    ctx, queries = stack(a.backend), read("eval_queries")
    with session_scope() as db:
        seed(ctx, db)
        if a.calibrate:
            return calibrate(ctx, db, queries)
        ret, kb_recall = retrieval(ctx, db, queries, load_reranker("cross-encoder", ctx.settings.reranker_model) if a.rerank else None)
        outs = [resolver.resolve(ctx, db, q["text"], "eval", use_cache=False) for q in queries]
        ood = [resolver.resolve(ctx, db, q["text"], "eval", use_cache=False) for q in read("ood_queries")]
        ts = []
        for q in (queries * 2)[:150]:
            t0 = time.perf_counter()
            resolver.resolve(ctx, db, q["text"], "eval", use_cache=False)
            ts.append((time.perf_counter() - t0) * 1000)
        hand = handwritten(ctx, db)
        short = short_queries(ctx, db)
        calib = calibration(ctx, db)
        freetext = free_text_agreement(ctx)
        life = lifecycle(ctx, db)   # last: it edits and retires KB articles
    res = {"kb_recall@3": kb_recall, **parsing(queries, outs), **rag(queries, outs), "ood_escalated": avg(o.resolution.escalate for o in ood),
           **evolving(a.backend, queries), "p50_ms": round(float(np.percentile(ts, 50)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
           "throughput_rps_1thread": round(1000 / statistics.mean(ts), 1)}
    flat = {f"{m}.{k}": v for m, d in ret.items() for k, v in d.items()} | res | {"kb_recall@3": kb_recall}
    flat |= {f"hand.{m}.{k}": v for m, d in hand["ret"].items() for k, v in d.items()}
    flat |= {f"hand.{k}": v for k, v in hand.items() if k not in ("ret", "n")}
    flat |= {f"freetext.{k}": v for k, v in freetext.items()}
    flat |= {f"short.{k}": v for k, v in short.items()} | {f"calib.{k}": v for k, v in calib.items() if not k.startswith("_")}
    gates = [(m, flat.get(m), t, flat.get(m) is not None and (flat[m] <= t if m in LOWER_IS_BETTER else flat[m] >= t)) for m, t in GATES[a.backend].items()]
    gates += [(f"lifecycle.{k}", v, True, bool(v)) for k, v in life.items()]
    gates.append(("calib.monotonic", flat.get("calib.monotonic"), True, bool(flat.get("calib.monotonic"))))
    md = [f"# Eval report ({a.backend}, {len(queries)} held-out complaints)\n", "## Retrieval (relevant = same underlying issue)\n",
          "| metric | dense | hybrid (used) | keyword baseline |\n|---|---|---|---|"]
    md += [f"| {k} | {ret['dense'][k]} | {ret['hybrid'][k]} | {ret['keyword'][k]} |" for k in ret["dense"]]
    if "hybrid+rerank" in ret:
        md += ["", "Cross-encoder reranking of the hybrid top-10:", "", "| metric | hybrid | hybrid + rerank |", "|---|---|---|"]
        md += [f"| {k} | {ret['hybrid'][k]} | {ret['hybrid+rerank'][k]} |" for k in ret["hybrid"]]
    bins = lambda rows: ["| stated confidence | n | mean stated | actually right |", "|---|---|---|---|"] + [f"| {a} | {n} | {b} | {c} |" for a, n, b, c in rows]  # noqa: E731
    md += ["", f"## Short / vague complaints ({short['n']}, e.g. 'wifi not working')\n",
           "What the agent *sees* must be on topic. `off_topic_free` = share of queries with no result from a forbidden department.\n",
           table("Short queries", {k: v for k, v in short.items() if k != "n"}),
           f"## Are the confidence scores honest? ({calib['n']} complaints)\n",
           "Match confidence = probability that the leading fix group is the right issue. Calibration error (ECE) is the average gap "
           "between stated confidence and observed accuracy.\n",
           table("Calibration", {k: v for k, v in calib.items() if not k.startswith("_") and k != "n"}),
           "Free-text answers (same fix worded differently vs genuinely different fixes):\n",
           table("Agreement on free-text answers", freetext),
           "**Match confidence vs correctness of the leading fix**\n", *bins(calib["_bins_match"]), "",
           "**Category confidence vs correctness of the category (known classes only)**\n", *bins(calib["_bins_category"]), ""]
    md += ["", f"## Hand-written complaints ({hand['n']}, not template-generated)\n",
           "| metric | dense | hybrid (used) | keyword baseline |\n|---|---|---|---|"]
    md += [f"| {k} | {hand['ret']['dense'][k]} | {hand['ret']['hybrid'][k]} | {hand['ret']['keyword'][k]} |" for k in hand["ret"]["dense"]]
    md += ["", table("Hand-written: parsing and KB", {k: v for k, v in hand.items() if k not in ("ret", "n")}),
           table("Parsing, RAG, abstention, evolving classes, latency", res), "## Gates\n", "| gate | value | threshold | |\n|---|---|---|---|"]
    md += [f"| {m} | {v} | {t} | {'PASS' if ok else 'FAIL'} |" for m, v, t, ok in gates]
    REPORTS.mkdir(exist_ok=True)
    report = REPORTS / ("report.md" if a.backend == "hashing" else f"report_{a.backend}.md")
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    print("\n".join(md))
    print(f"\nreport written to {report}")
    failed = [g for g in gates if not g[3]]
    print(f"\ngates: {len(gates) - len(failed)}/{len(gates)} passed")
    if a.gates and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
