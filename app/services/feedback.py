"""Agent feedback loop: ratings feed health metrics; corrections/resolutions flow back into the index."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import metrics as m
from app.core.audit import write_audit
from app.core.context import Context
from app.db.models import Feedback, QueryLog
from app.schemas import FeedbackIn, TicketIn
from app.services import ingestion, taxonomy


def record_feedback(ctx: Context, db: Session, fb: FeedbackIn, user_id: str) -> dict:
    q = db.get(QueryLog, fb.query_id)
    if q is None:
        raise LookupError("unknown query_id")
    db.add(Feedback(query_id=fb.query_id, user_id=user_id, helpful=fb.helpful, rating=fb.rating,
                    correct_category=fb.correct_category, comment=fb.comment))
    db.commit()
    m.FEEDBACK.labels(str(fb.helpful).lower()).inc()
    promoted = None
    if fb.resolved_steps:
        # agent solved it: promote complaint + steps to a resolved ticket (category = correction, else predicted)
        cat = fb.correct_category or (q.category if q.is_known else None)
        if cat:
            cat = taxonomy.ensure_category(db, cat, created_by=user_id)
        item = TicketIn(external_id=f"fb-{fb.query_id}", subject=q.text[:120], body=q.text, category=cat,
                        product=(q.analysis or {}).get("product"), severity=(q.analysis or {}).get("severity"),
                        resolution_steps=fb.resolved_steps, status="resolved")
        ingestion.ingest_tickets(ctx, db, [item], source="feedback")
        write_audit(db, user_id, "feedback.promote_ticket", {"query_id": fb.query_id, "category": cat})
        promoted = item.external_id
    return {"recorded": True, "promoted_ticket": promoted}


def helpful_ratio(db: Session) -> float | None:
    rows = db.execute(select(Feedback.helpful)).scalars().all()
    return round(sum(rows) / len(rows), 3) if rows else None
