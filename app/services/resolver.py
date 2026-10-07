"""End-to-end /resolve pipeline: redact -> embed -> retrieve -> parse -> rerank -> generate -> log."""
from __future__ import annotations

import json
import time
import uuid

from sqlalchemy.orm import Session

from app.core import metrics as m
from app.core.context import Context
from app.core.pii import redact
from app.core.reranker import load_reranker, rerank_hits
from app.core.text import content_hash, strip_boilerplate
from app.db.models import QueryLog
from app.schemas import Analysis, ResolveOut, SearchOut, Source
from app.services import analyzer, indexer, rag, retriever, taxonomy
from app.services import confidence as conf
from app.services.docs import origin_of


def _clean(ctx: Context, text: str) -> str:
    return redact(" ".join(text.split())) if ctx.settings.redact_pii else " ".join(text.split())


def resolve(ctx: Context, db: Session, text: str, user_id: str, *, top_k_tickets: int | None = None,
            top_k_kb: int | None = None, use_cache: bool = True, request_id: str | None = None) -> ResolveOut:
    t_start = time.perf_counter()
    s = ctx.settings
    request_id = request_id or uuid.uuid4().hex
    clean = _clean(ctx, text)
    info = indexer.get_active(ctx, db)
    kt, kk = top_k_tickets or s.top_k_tickets, s.top_k_kb if top_k_kb is None else top_k_kb

    ckey = "resolve:v5:" + content_hash(info.collection, clean.lower(), kt, kk, s.llm_enabled, s.llm_analysis, s.reranker)
    if use_cache and (hit := ctx.kv.get(ckey)):
        m.CACHE.labels("hit").inc()
        data = json.loads(hit)
        data.update(request_id=request_id, query_id=uuid.uuid4().hex, cached=True, latency_ms=round((time.perf_counter() - t_start) * 1000, 2))
        out = ResolveOut(**data)
        _log_query(db, user_id, clean, out, top_score=max((x.score for x in out.sources), default=0.0))
        return out
    m.CACHE.labels("miss").inc()

    candidate_k = max(s.knn_k, kt)
    tickets, kb = retriever.retrieve(
        ctx, info, clean,
        knn_k=candidate_k,
        kb_k=max(kk, 5)
    )
    tickets = rerank_hits(load_reranker(s.reranker, s.reranker_model), clean, tickets)

    t0 = time.perf_counter()
    cats = taxonomy.active_categories(db)
    analysis: Analysis = analyzer.analyze(ctx, clean, tickets, cats)
    m.STAGE_LATENCY.labels("analyze").observe(time.perf_counter() - t0)
    if not analysis.is_known_category:
        m.UNKNOWN_CATEGORY.inc()

    cat = analysis.category if analysis.is_known_category else None
    # confidence uses every candidate above the abstain floor; only results near the best match are shown
    cand_t = _above_floor(retriever.rerank(tickets, cat, analysis.product)[:kt], "ticket", s)
    kb_ranked = _on_topic(retriever.rerank(kb, cat, analysis.product)[:kk], "kb", s)
    top_score = max([h.score for h in tickets + kb], default=0.0)
    m.TOP_SCORE.observe(top_score)

    confidence, focus_idx = conf.assess(cand_t, kb_ranked, s, embed=info.embedder.embed)
    rel = [conf.relevance(h.score, "ticket", s) for h in cand_t]
    keep = [i for i, r in enumerate(rel) if i in focus_idx or r >= REL_KEEP * max(rel, default=0.0)]
    tickets_ranked = [cand_t[i] for i in keep]
    focus = [f"T{keep.index(i) + 1}" for i in focus_idx]
    confidence.source_ids = focus
    m.CONFIDENCE.observe(confidence.score)
    if confidence.level == "low":
        m.LOW_CONFIDENCE.inc()
    sources, resolution = rag.generate(ctx, clean, analysis, tickets_ranked, kb_ranked, focus=focus,
                                       low_confidence=confidence.level == "low", confidence_score=confidence.score)
    if resolution.escalate and analysis.is_known_category is False:
        resolution.escalation_reason = (resolution.escalation_reason or "") + " Issue type looks new to the taxonomy."

    out = ResolveOut(request_id=request_id, query_id=uuid.uuid4().hex, analysis=analysis, sources=sources,
                     resolution=resolution, confidence=confidence, index_version=info.collection, cached=False,
                     latency_ms=round((time.perf_counter() - t_start) * 1000, 2))
    _log_query(db, user_id, clean, out, top_score)
    if use_cache and not resolution.escalate:
        ctx.kv.set(ckey, out.model_dump_json(), s.cache_ttl_s)
    m.STAGE_LATENCY.labels("total").observe(time.perf_counter() - t_start)
    return out


REL_KEEP = 0.5   # show a hit only if its relevance is at least this fraction of the best one


def _above_floor(hits: list, doc_type: str, s) -> list:
    kept = [h for h in hits if conf.relevance(h.score, doc_type, s) > 0]
    return kept or hits[:1]


def _on_topic(hits: list, doc_type: str, s) -> list:
    """Keep only results that are genuinely close: above the abstain floor AND within reach of the best match.
    Showing the rest is how an unrelated billing article ends up under a Wi-Fi complaint. The best hit is always
    kept so the abstain logic still sees what was closest."""
    rel = [conf.relevance(h.score, doc_type, s) for h in hits]
    best = max(rel, default=0.0)
    kept = [h for h, r in zip(hits, rel, strict=True) if r > 0 and r >= REL_KEEP * best]
    return kept or hits[:1]


def _log_query(db: Session, user_id: str, text: str, out: ResolveOut, top_score: float) -> None:
    db.add(QueryLog(id=out.query_id, user_id=user_id, text=text,
                    analysis=out.analysis.model_dump(), top_score=top_score, category=out.analysis.category,
                    is_known=out.analysis.is_known_category, source_ids=[x.external_id for x in out.sources],
                    escalate=out.resolution.escalate, generation_mode=out.resolution.generation_mode,
                    latency_ms=out.latency_ms, cached=out.cached))
    db.commit()


def search(ctx: Context, db: Session, text: str, top_k: int, doc_type: str = "all", category: str | None = None,
           product: str | None = None) -> SearchOut:
    info = indexer.get_active(ctx, db)
    clean = _clean(ctx, text)
    where = {k: v for k, v in (("category", category), ("product", product)) if v}
    hits = []
    for dt in (["ticket", "kb"] if doc_type == "all" else [doc_type]):
        hits += retriever.search_hits(ctx, info, clean, top_k, {**where, "doc_type": dt})
    hits.sort(key=lambda h: (-h.rrf, -h.score, h.id))
    best: dict[str, float] = {}
    for h in hits:
        dt = h.payload["doc_type"]
        best[dt] = max(best.get(dt, 0.0), conf.relevance(h.score, dt, ctx.settings))
    ranked = [h for h in hits if (r := conf.relevance(h.score, h.payload["doc_type"], ctx.settings)) > 0
              and r >= REL_KEEP * best[h.payload["doc_type"]]]
    res = [Source(id=f"R{i}", doc_type=h.payload["doc_type"], external_id=h.payload["external_id"],
                  origin=origin_of(h.payload["external_id"]), title=strip_boilerplate(h.payload.get("title", "")),
                  score=round(h.score, 4), relevance=round(conf.relevance(h.score, h.payload["doc_type"], ctx.settings), 3),
                  category=h.payload.get("category"), product=h.payload.get("product"),
                  snippet=strip_boilerplate(h.payload.get("text") or "")[:240])
           for i, h in enumerate(ranked[:top_k], 1)]
    return SearchOut(results=res, index_version=info.collection)
