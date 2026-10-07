"""Row -> vector-point conversion (shared by ingestion, reconcile and re-index)."""
from __future__ import annotations

from app.core.context import Context
from app.core.embeddings import Embedder, sparse_encode
from app.core.text import chunk_text, strip_boilerplate
from app.core.vectorstore import point_id
from app.db.models import KBArticle, Ticket


def origin_of(external_id: str) -> str:
    """Hugging Face imports are namespaced (`hf-...`, `kb-hf-...`); everything else is the telecom data."""
    return "hf" if external_id.startswith(("hf-", "kb-hf-")) else "telecom"


def ticket_embed_text(t: Ticket) -> str:
    return strip_boilerplate(f"{t.subject}\n{t.body}")


def ticket_payload(t: Ticket) -> dict:
    return {
        "doc_type": "ticket", "external_id": t.external_id, "title": t.subject or t.body[:80],
        "text": t.body[:1500], "category": t.category, "product": t.product, "severity": t.severity,
        "resolution_steps": list(t.resolution_steps or []), "issue_key": t.issue_key,
    }


def index_tickets(ctx: Context, emb: Embedder, collection: str, tickets: list[Ticket]) -> None:
    if not tickets:
        return
    texts = [ticket_embed_text(t) for t in tickets]
    ctx.vectors.upsert(collection, [point_id("ticket", t.external_id) for t in tickets], emb.embed(texts),
                       [sparse_encode(x) for x in texts], [ticket_payload(t) for t in tickets])


def index_kb_article(ctx: Context, emb: Embedder, collection: str, a: KBArticle) -> int:
    # replace all previous chunks of this article (article may have shrunk or been edited)
    ctx.vectors.delete_where(collection, {"doc_type": "kb", "external_id": a.external_id})
    if not a.active:
        return 0
    chunks = chunk_text(a.body)
    texts = [f"{a.title}\n{c}" for c in chunks]
    vecs = emb.embed(texts)
    payloads = [{"doc_type": "kb", "external_id": a.external_id, "title": a.title, "text": c, "chunk": i,
                 "category": a.category, "product": a.product, "version": a.version}
                for i, c in enumerate(chunks)]
    ctx.vectors.upsert(collection, [point_id("kb", a.external_id, i) for i in range(len(chunks))], vecs,
                       [sparse_encode(x) for x in texts], payloads)
    return len(chunks)
