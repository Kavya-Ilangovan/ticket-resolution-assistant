"""Seed the database + vector index with the bundled synthetic telecom data.

    python -m scripts.seed              # base taxonomy (8 classes)
    python -m scripts.seed --novel      # additionally the unseen '5G Home Internet' class
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.core.context import get_context
from app.db.session import init_engine, session_scope
from app.schemas import KBArticleIn, TicketIn
from app.services import ingestion

DATA = Path(__file__).resolve().parent.parent / "data"


def read(name: str, folder: str = "synthetic") -> list[dict]:
    """Read data/<folder>/<name>.jsonl (UTF-8 explicitly: Windows defaults to cp1252)."""
    path = DATA / folder / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed(ctx, db, novel: bool = False) -> dict:
    out = {"tickets": ingestion.ingest_tickets(ctx, db, [TicketIn(**r) for r in read("tickets")], source="seed"),
           "kb": ingestion.ingest_kb(ctx, db, [KBArticleIn(**r) for r in read("kb")])}
    if novel:
        out["novel_tickets"] = ingestion.ingest_tickets(ctx, db, [TicketIn(**r) for r in read("novel_tickets")], source="seed")
        out["novel_kb"] = ingestion.ingest_kb(ctx, db, [KBArticleIn(**r) for r in read("novel_kb")])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel", action="store_true")
    a = ap.parse_args()
    init_engine(get_settings())
    ctx = get_context()
    with session_scope() as db:
        print(json.dumps(seed(ctx, db, a.novel), indent=2))
    ctx.vectors.client.close()   # embedded Qdrant: close cleanly (avoids a noisy traceback at interpreter exit)
