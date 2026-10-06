"""Data/concept drift monitoring and emerging-class discovery from live traffic."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.context import Context
from app.core.text import stem, tokenize
from app.db.models import Feedback, QueryLog, utcnow
from app.services import indexer


def _stats(rows: list[QueryLog]) -> dict:
    if not rows:
        return {"n": 0}
    top = np.array([r.top_score for r in rows])
    return {
        "n": len(rows),
        "unknown_rate": round(sum(not r.is_known for r in rows) / len(rows), 3),
        "escalation_rate": round(sum(r.escalate for r in rows) / len(rows), 3),
        "top_score_mean": round(float(top.mean()), 3),
        "top_score_p10": round(float(np.percentile(top, 10)), 3),
        "latency_p95_ms": round(float(np.percentile([r.latency_ms for r in rows], 95)), 1),
    }


def drift_report(db: Session, window_days: int = 7, min_n: int = 30) -> dict:
    """Compare the latest window with the one before it. Flags are intentionally simple and explainable."""
    now = utcnow()
    cur = list(db.execute(select(QueryLog).where(QueryLog.created_at >= now - timedelta(days=window_days))).scalars())
    prev = list(db.execute(select(QueryLog).where(QueryLog.created_at < now - timedelta(days=window_days),
                                                  QueryLog.created_at >= now - timedelta(days=2 * window_days))).scalars())
    a, b = _stats(cur), _stats(prev)
    flags = []
    if a["n"] >= min_n and b["n"] >= min_n:
        if a["unknown_rate"] - b["unknown_rate"] > 0.05:
            flags.append("unknown_category_rate_up: possible new problem class or product launch")
        if b["top_score_mean"] - a["top_score_mean"] > 0.05:
            flags.append("retrieval_similarity_down: index may be stale vs. incoming complaints")
        if a["escalation_rate"] - b["escalation_rate"] > 0.08:
            flags.append("escalation_rate_up")
    fb = list(db.execute(select(Feedback.helpful).where(Feedback.created_at >= now - timedelta(days=window_days))).scalars())
    hr = round(sum(fb) / len(fb), 3) if fb else None
    if hr is not None and len(fb) >= min_n and hr < 0.6:
        flags.append(f"helpful_ratio_low: {hr}")
    return {"window_days": window_days, "current": a, "previous": b, "helpful_ratio": hr, "flags": flags}


def emerging_topics(ctx: Context, db: Session, days: int = 14, threshold: float | None = None,
                    min_size: int = 3, limit: int = 500) -> list[dict]:
    """Greedy cosine clustering of complaints that matched no known class -> candidates for a new category."""
    info = indexer.get_active(ctx, db)
    rows = list(db.execute(select(QueryLog).where(QueryLog.is_known.is_(False),
                                                  QueryLog.created_at >= utcnow() - timedelta(days=days))
                           .order_by(QueryLog.created_at.desc()).limit(limit)).scalars())
    if not rows:
        return []
    thr = threshold or (0.55 if info.backend == "sentence-transformers" else 0.35)
    vecs = info.embedder.embed([r.text for r in rows])
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    for i, v in enumerate(vecs):
        if centroids:
            sims = np.array([c @ v / (np.linalg.norm(c) + 1e-9) for c in centroids])
            j = int(sims.argmax())
            if sims[j] >= thr:
                members[j].append(i)
                centroids[j] = centroids[j] + v
                continue
        centroids.append(v.copy())
        members.append([i])
    out = []
    for mem in members:
        if len(mem) < min_size:
            continue
        words = Counter(stem(t) for i in mem for t in tokenize(rows[i].text))
        out.append({"size": len(mem), "top_terms": [w for w, _ in words.most_common(8)],
                    "examples": [rows[i].text[:200] for i in mem[:3]],
                    "query_ids": [rows[i].id for i in mem[:20]]})
    return sorted(out, key=lambda c: -c["size"])
