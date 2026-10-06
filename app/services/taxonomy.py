"""Evolving ticket classes: add / list / merge categories without retraining anything.

The classifier is a kNN vote over indexed tickets, so a new class becomes predictable the moment a few
labelled seed tickets are indexed.
"""
from __future__ import annotations

import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.audit import write_audit
from app.core.context import Context
from app.db.models import Category, Ticket
from app.services import indexer

UNKNOWN = "unknown"
UNCATEGORIZED = "uncategorized"


def canonical(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def active_categories(db: Session) -> list[str]:
    return [c for (c,) in db.execute(select(Category.name).where(Category.active.is_(True)).order_by(Category.name))]


def ensure_category(db: Session, name: str | None, created_by: str = "ingest", description: str = "") -> str:
    if not name:
        return UNCATEGORIZED
    name = canonical(name)
    existing = db.execute(select(Category).where(func.lower(Category.name) == name.lower())).scalar_one_or_none()
    if existing:
        if existing.merged_into:          # follow merges so stale producers keep working
            return existing.merged_into
        return existing.name
    db.add(Category(name=name, description=description, created_by=created_by))
    db.commit()
    return name


def list_categories(db: Session) -> list[dict]:
    counts = dict(db.execute(select(Ticket.category, func.count()).group_by(Ticket.category)).all())
    return [
        {"name": c.name, "description": c.description, "active": c.active, "merged_into": c.merged_into,
         "tickets": int(counts.get(c.name, 0))}
        for c in db.execute(select(Category).order_by(Category.name)).scalars()
    ]


def merge_category(ctx: Context, db: Session, src: str, dst: str, actor: str) -> dict:
    src_c = db.execute(select(Category).where(Category.name == src)).scalar_one_or_none()
    dst_name = ensure_category(db, dst, created_by=actor)
    if src_c is None or src_c.name == dst_name:
        raise ValueError("unknown source category or src == dst")
    moved = db.query(Ticket).filter(Ticket.category == src).update({"category": dst_name})
    src_c.active = False
    src_c.merged_into = dst_name
    db.commit()
    info = indexer.get_active(ctx, db)
    ctx.vectors.set_payload_where(info.collection, {"category": dst_name}, {"category": src})
    write_audit(db, actor, "category.merge", {"src": src, "dst": dst_name, "tickets": moved})
    return {"merged": src, "into": dst_name, "tickets_moved": moved}
