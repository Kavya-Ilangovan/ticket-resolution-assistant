"""Tamper-evident audit trail: each row stores sha256(prev_hash || canonical_json(entry)).

`verify_chain` recomputes the chain; any edited/deleted row breaks it. For external non-repudiation,
the head hash can be anchored periodically on a public chain (e.g. Ethereum Sepolia) - intentionally
left out of the core path (see docs/PRODUCTION.md).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, utcnow

GENESIS = "0" * 64


def _digest(prev: str, ts: datetime, actor: str, action: str, payload: dict) -> str:
    body = json.dumps({"ts": ts.replace(tzinfo=None).isoformat(), "actor": actor, "action": action, "payload": payload},
                      sort_keys=True, default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


def write_audit(db: Session, actor: str, action: str, payload: dict) -> AuditLog:
    last = db.execute(select(AuditLog).order_by(AuditLog.id.desc()).limit(1)).scalar_one_or_none()
    prev = last.hash if last else GENESIS
    ts = utcnow()
    row = AuditLog(ts=ts, actor=actor, action=action, payload=payload, prev_hash=prev,
                   hash=_digest(prev, ts, actor, action, payload))
    db.add(row)
    db.commit()
    return row


def verify_chain(db: Session) -> dict:
    prev = GENESIS
    n = 0
    for row in db.execute(select(AuditLog).order_by(AuditLog.id)).scalars():
        if row.prev_hash != prev or row.hash != _digest(prev, row.ts, row.actor, row.action, row.payload):
            return {"valid": False, "broken_at_id": row.id, "entries": n}
        prev = row.hash
        n += 1
    return {"valid": True, "entries": n, "head": prev}
