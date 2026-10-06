"""Semantic retrieval over resolved tickets + KB, with metadata-aware re-ranking."""
from __future__ import annotations

import time

from app.core import metrics as m
from app.core.context import Context
from app.core.embeddings import sparse_encode
from app.core.text import strip_boilerplate
from app.core.vectorstore import Hit
from app.services.indexer import IndexInfo


def search_hits(ctx: Context, info: IndexInfo, text: str, limit: int, where: dict | None = None,
                mode: str | None = None) -> list[Hit]:
    mode = mode or ("hybrid" if ctx.settings.hybrid_search else "dense")
    text = strip_boilerplate(text)
    return ctx.vectors.search(info.collection, info.embedder.embed([text])[0], sparse_encode(text), limit, where, mode)


def retrieve(ctx: Context, info: IndexInfo, text: str, *, knn_k: int, kb_k: int,
             where_extra: dict | None = None) -> tuple[list[Hit], list[Hit]]:
    text = strip_boilerplate(text)
    t0 = time.perf_counter()
    vec = info.embedder.embed([text])[0]
    m.STAGE_LATENCY.labels("embed").observe(time.perf_counter() - t0)
    t0 = time.perf_counter()
    base = dict(where_extra or {})
    mode = "hybrid" if ctx.settings.hybrid_search else "dense"
    sp = sparse_encode(text)
    tickets = ctx.vectors.search(info.collection, vec, sp, knn_k, {**base, "doc_type": "ticket"}, mode)
    kb = ctx.vectors.search(info.collection, vec, sp, kb_k, {**base, "doc_type": "kb"}, mode) if kb_k else []
    m.STAGE_LATENCY.labels("vector_search").observe(time.perf_counter() - t0)
    return tickets, kb


def rerank(hits: list[Hit], category: str | None, product: str | None) -> list[Hit]:
    """Small, explainable boost for agreement with the parsed category/product (never overrides semantics)."""
    def adj(h: Hit) -> float:
        b = 0.0
        if category and h.payload.get("category") == category:
            b += 0.05
        if product and h.payload.get("product") == product:
            b += 0.03
        return h.score + b
    return sorted(hits, key=adj, reverse=True)
