"""Index lifecycle: the active collection and embedder, and blue/green re-indexing (build, catch up, flip one pointer)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.context import Context
from app.core.embeddings import Embedder
from app.db.models import IndexState, KBArticle, Ticket, utcnow
from app.services import docs

log = logging.getLogger(__name__)
ACTIVE_KEY = "active_index"


@dataclass
class IndexInfo:
    collection: str
    backend: str
    model: str
    dim: int
    version: int
    embedder: Embedder

    @property
    def label(self) -> str:
        return f"{self.collection}"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def _collection_name(prefix: str, backend: str, model: str, version: int) -> str:
    return f"{prefix}_{_slug(model if backend != 'hashing' else 'hashing')}_v{version}"


def _model_for(ctx: Context, backend: str, model: str | None) -> str:
    return model or (ctx.settings.embedding_model if backend == "sentence-transformers" else f"hashing-{ctx.settings.hashing_dim}")


def get_active(ctx: Context, db: Session) -> IndexInfo:
    row = db.get(IndexState, ACTIVE_KEY)
    if row is None:  # first boot: create v1 from env config
        backend = ctx.settings.embedding_backend
        model = _model_for(ctx, backend, None)
        emb = ctx.embedders.get(backend, ctx.settings.embedding_model)
        val = {"collection": _collection_name(ctx.settings.collection_prefix, backend, model, 1),
               "backend": backend, "model": ctx.settings.embedding_model, "dim": emb.dim, "version": 1}
        row = IndexState(key=ACTIVE_KEY, value=val)
        db.add(row)
        db.commit()
    v = row.value
    emb = ctx.embedders.get(v["backend"], v["model"])
    ctx.vectors.ensure_collection(v["collection"], v["dim"])
    return IndexInfo(v["collection"], v["backend"], v["model"], v["dim"], v["version"], emb)


def reindex(ctx: Context, db: Session, backend: str | None = None, model: str | None = None,
            progress=None) -> dict:
    """Build a fresh collection from Postgres, catch up with concurrent writes, then swap."""
    cur = get_active(ctx, db)
    backend = backend or cur.backend
    model_name = model or (cur.model if backend == cur.backend else ctx.settings.embedding_model)
    emb = ctx.embedders.get(backend, model_name)
    version = cur.version + 1
    new_coll = _collection_name(ctx.settings.collection_prefix, backend, _model_for(ctx, backend, model_name), version)
    ctx.vectors.drop_collection(new_coll)
    ctx.vectors.ensure_collection(new_coll, emb.dim)
    started = utcnow()
    n = 0
    n += _copy_all(ctx, db, emb, new_coll, since=None)
    if progress:
        progress(n)
    # catch-up: rows written while we were copying
    for _ in range(3):
        extra = _copy_all(ctx, db, emb, new_coll, since=started)
        n += extra
        if extra == 0:
            break
        started = utcnow()
    row = db.get(IndexState, ACTIVE_KEY)
    row.value = {"collection": new_coll, "backend": backend, "model": model_name, "dim": emb.dim, "version": version}
    db.commit()
    db.execute(update(Ticket).where(Ticket.status == "resolved").values(indexed_collection=new_coll))
    db.execute(update(KBArticle).where(KBArticle.active.is_(True)).values(indexed_collection=new_coll))
    db.commit()
    ctx.vectors.drop_collection(cur.collection) if cur.collection != new_coll else None
    return {"old": cur.collection, "new": new_coll, "documents": n, "dim": emb.dim}


def _copy_all(ctx: Context, db: Session, emb: Embedder, coll: str, since, batch: int = 128) -> int:
    total = 0
    q = select(Ticket).where(Ticket.status == "resolved")
    if since:
        q = q.where(Ticket.updated_at >= since)
    rows = list(db.execute(q).scalars())
    for i in range(0, len(rows), batch):
        docs.index_tickets(ctx, emb, coll, rows[i : i + batch])
        total += len(rows[i : i + batch])
    kq = select(KBArticle).where(KBArticle.active.is_(True))
    if since:
        kq = kq.where(KBArticle.updated_at >= since)
    arts = list(db.execute(kq).scalars())
    for a in arts:
        docs.index_kb_article(ctx, emb, coll, a)
        total += 1
    return total
