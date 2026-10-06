"""Background jobs. All are idempotent so at-least-once delivery (acks_late) is safe."""
from __future__ import annotations

import logging

from app.core import metrics as m
from app.core.audit import write_audit
from app.core.context import get_context
from app.db.models import IngestJob, utcnow
from app.db.session import session_scope
from app.schemas import KBArticleIn, TicketIn
from app.services import indexer, ingestion, taxonomy
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)


def _run(job_id: str, kind: str, fn):
    ctx = get_context()
    with session_scope() as db:
        job = db.get(IngestJob, job_id)
        job.status = "running"
        db.commit()
        try:
            result = fn(ctx, db, job)
            job.status, job.finished_at = "done", utcnow()
            m.JOBS.labels(kind, "done").inc()
            return result
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            db.rollback()
            job = db.get(IngestJob, job_id)
            job.status, job.error, job.finished_at = "failed", str(e)[:2000], utcnow()
            m.JOBS.labels(kind, "failed").inc()
            return None


@celery_app.task(name="tra.ingest_tickets", autoretry_for=(ConnectionError,), retry_backoff=True, max_retries=3)
def ingest_tickets_task(job_id: str, items: list[dict], source: str = "api"):
    return _run(job_id, "tickets", lambda ctx, db, job: ingestion.ingest_tickets(
        ctx, db, [TicketIn(**i) for i in items], job=job, source=source))


@celery_app.task(name="tra.ingest_kb", autoretry_for=(ConnectionError,), retry_backoff=True, max_retries=3)
def ingest_kb_task(job_id: str, items: list[dict]):
    return _run(job_id, "kb", lambda ctx, db, job: ingestion.ingest_kb(ctx, db, [KBArticleIn(**i) for i in items], job=job))


@celery_app.task(name="tra.reindex")
def reindex_task(job_id: str, backend: str | None = None, model: str | None = None, actor: str = "system"):
    def go(ctx, db, job):
        res = indexer.reindex(ctx, db, backend, model)
        job.processed = job.total = res["documents"]
        write_audit(db, actor, "index.reindex", res)
        return res
    return _run(job_id, "reindex", go)


@celery_app.task(name="tra.seed_category")
def seed_category_task(job_id: str, items: list[dict]):
    return _run(job_id, "seed_category", lambda ctx, db, job: ingestion.ingest_tickets(
        ctx, db, [TicketIn(**i) for i in items], job=job, source="seed"))


@celery_app.task(name="tra.reconcile")
def reconcile_task():
    ctx = get_context()
    with session_scope() as db:
        n = ingestion.reconcile_unindexed(ctx, db)
    return n


@celery_app.task(name="tra.merge_category")
def merge_category_task(src: str, dst: str, actor: str):
    ctx = get_context()
    with session_scope() as db:
        return taxonomy.merge_category(ctx, db, src, dst, actor)


@celery_app.task(name="tra.seed_demo")
def seed_demo_task(job_id: str, novel: bool = False):
    from scripts.seed import seed
    return _run(job_id, "seed_demo", lambda ctx, db, job: seed(ctx, db, novel=novel))


@celery_app.task(name="tra.import_hf")
def import_hf_task(job_id: str, limit: int = 3000, kb_limit: int = 300, lang: str = "en", per_queue: int | None = None):
    """Download the Hugging Face dataset and add tickets + derived KB articles (needs network and `datasets`)."""
    from app.services import hf_import

    def go(ctx, db, job):
        plan = hf_import.build_plan(hf_import.iter_rows(), lang=lang or None, limit=limit or None, kb_limit=kb_limit,
                                    per_queue=per_queue)
        if not plan.tickets:
            raise RuntimeError(f"nothing importable: {plan.report}")
        job.total = len(plan.tickets) + len(plan.kb)
        db.commit()
        return hf_import.run_import(ctx, db, plan, job=job)
    return _run(job_id, "import_hf", go)
