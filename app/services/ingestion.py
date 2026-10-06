"""Idempotent, incremental ingestion of tickets and KB articles (Postgres first, then vector index)."""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import metrics as m
from app.core.context import Context
from app.core.pii import redact
from app.core.text import content_hash
from app.db.models import IngestJob, KBArticle, Ticket, utcnow
from app.schemas import KBArticleIn, TicketIn
from app.services import docs, indexer, taxonomy

log = logging.getLogger(__name__)
BATCH = 128


def _red(ctx: Context, text: str) -> str:
    return redact(text) if ctx.settings.redact_pii else text


def ingest_tickets(ctx: Context, db: Session, items: list[TicketIn], *, job: IngestJob | None = None,
                   source: str = "api") -> dict:
    stats = {"inserted": 0, "updated": 0, "skipped": 0}
    info = indexer.get_active(ctx, db)
    to_index: list[Ticket] = []

    def flush():
        if not to_index:
            return
        docs.index_tickets(ctx, info.embedder, info.collection, to_index)
        for t in to_index:
            t.indexed_collection = info.collection
        db.commit()
        to_index.clear()

    for n, it in enumerate(items, 1):
        subject, body = _red(ctx, it.subject), _red(ctx, it.body)
        steps = [_red(ctx, s) for s in it.resolution_steps]
        cat = taxonomy.ensure_category(db, it.category, created_by="ingest") if it.category else taxonomy.UNCATEGORIZED
        h = content_hash(subject, body, cat, it.product, it.severity, it.status, *steps)
        ext = it.external_id or f"auto-{content_hash(subject, body)[:24]}"
        row = db.execute(select(Ticket).where(Ticket.external_id == ext)).scalar_one_or_none()
        if row and row.content_hash == h and row.indexed_collection == info.collection:
            stats["skipped"] += 1
            m.INGESTED.labels("ticket", "skipped").inc()
        else:
            if row is None:
                row = Ticket(external_id=ext, body=body, content_hash=h)
                db.add(row)
                stats["inserted"] += 1
                m.INGESTED.labels("ticket", "inserted").inc()
            else:
                stats["updated"] += 1
                m.INGESTED.labels("ticket", "updated").inc()
            row.subject, row.body, row.category, row.product, row.severity = subject, body, cat, it.product, it.severity
            row.resolution_steps, row.status, row.issue_key, row.source = steps, it.status, it.issue_key, source
            row.content_hash, row.updated_at = h, utcnow()
            db.flush()
            # only resolved tickets with a resolution are useful as RAG evidence
            if row.status == "resolved" and steps:
                to_index.append(row)
            else:
                row.indexed_collection = None
        if len(to_index) >= BATCH:
            flush()
        if job is not None and n % BATCH == 0:
            _progress(db, job, n, stats)
    db.commit()
    flush()
    if job is not None:
        _progress(db, job, len(items), stats)
    return stats


def ingest_kb(ctx: Context, db: Session, items: list[KBArticleIn], *, job: IngestJob | None = None) -> dict:
    stats = {"inserted": 0, "updated": 0, "skipped": 0}
    info = indexer.get_active(ctx, db)
    for n, it in enumerate(items, 1):
        title, body = _red(ctx, it.title), _red(ctx, it.body)
        cat = taxonomy.ensure_category(db, it.category, created_by="ingest") if it.category else None
        h = content_hash(title, body, cat, it.product, it.version, it.active)
        row = db.execute(select(KBArticle).where(KBArticle.external_id == it.external_id)).scalar_one_or_none()
        if row and row.content_hash == h and row.indexed_collection == info.collection:
            stats["skipped"] += 1
            m.INGESTED.labels("kb", "skipped").inc()
        else:
            if row is None:
                row = KBArticle(external_id=it.external_id, title=title, body=body, content_hash=h)
                db.add(row)
                stats["inserted"] += 1
                m.INGESTED.labels("kb", "inserted").inc()
            else:
                stats["updated"] += 1
                m.INGESTED.labels("kb", "updated").inc()
            row.title, row.body, row.category, row.product = title, body, cat, it.product
            row.version, row.active, row.content_hash, row.updated_at = it.version, it.active, h, utcnow()
            db.flush()
            docs.index_kb_article(ctx, info.embedder, info.collection, row)
            row.indexed_collection = info.collection if row.active else None
            db.commit()
        if job is not None and n % 20 == 0:
            _progress(db, job, n, stats)
    db.commit()
    if job is not None:
        _progress(db, job, len(items), stats)
    return stats


def _progress(db: Session, job: IngestJob, processed: int, stats: dict) -> None:
    job.processed = processed
    job.inserted, job.updated, job.skipped = stats["inserted"], stats["updated"], stats["skipped"]
    db.commit()


def reconcile_unindexed(ctx: Context, db: Session, limit: int = 2000) -> int:
    """Self-healing: re-index any resolved ticket / active KB article missing from the active collection
    (e.g. the vector DB was down when it was ingested). Run periodically by Celery beat."""
    info = indexer.get_active(ctx, db)
    tickets = list(db.execute(
        select(Ticket).where(Ticket.status == "resolved", (Ticket.indexed_collection.is_(None)) |
                             (Ticket.indexed_collection != info.collection)).limit(limit)).scalars())
    tickets = [t for t in tickets if t.resolution_steps]
    for i in range(0, len(tickets), BATCH):
        docs.index_tickets(ctx, info.embedder, info.collection, tickets[i : i + BATCH])
    for t in tickets:
        t.indexed_collection = info.collection
    arts = list(db.execute(select(KBArticle).where(
        KBArticle.active.is_(True), (KBArticle.indexed_collection.is_(None)) |
        (KBArticle.indexed_collection != info.collection)).limit(limit)).scalars())
    for a in arts:
        docs.index_kb_article(ctx, info.embedder, info.collection, a)
        a.indexed_collection = info.collection
    db.commit()
    return len(tickets) + len(arts)
