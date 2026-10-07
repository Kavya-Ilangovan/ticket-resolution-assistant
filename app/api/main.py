"""FastAPI microservice: complaint analysis, semantic search, grounded resolution, ingestion, admin."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import schemas as sc
from app.config import Settings, get_settings
from app.core import metrics as m
from app.core.audit import verify_chain, write_audit
from app.core.auth import User, get_current_user, mint_dev_token, require_role
from app.core.context import Context, get_context
from app.db.models import IngestJob, KBArticle, Ticket
from app.db.session import get_db, init_engine
from app.logging_setup import request_id_var, setup_logging
from app.services import analyzer, drift, feedback, indexer, resolver, retriever, taxonomy
from app.worker import tasks

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level)
    if s.app_env == "prod" and s.auth_mode == "off":
        raise RuntimeError("AUTH_MODE=off is forbidden when APP_ENV=prod")
    if s.app_env == "prod" and s.jwt_secret == "change-me-in-prod" and s.auth_mode == "jwt":
        raise RuntimeError("set a real JWT_SECRET")
    init_engine(s)
    from app.db.session import session_scope
    with session_scope() as db:
        indexer.get_active(get_context(), db)   # create collection / pointer on first boot
    fb, problem = s.firebase_web_result
    if problem:
        log.warning("Google sign-in DISABLED: %s", problem)
    elif fb:
        log.info("Google sign-in enabled (Firebase project %s); admins: %d configured email(s)", fb.get("projectId"), len(s.admin_email_list))
    log.info("startup complete (env=%s, embeddings=%s)", s.app_env, s.embedding_backend)
    yield


app = FastAPI(title="Intelligent Support Ticket Resolution Assistant", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- web UI
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


@app.get("/v1/config", tags=["ops"])
def public_config(s: Settings = Depends(get_settings)):
    """Non-secret settings the browser UI needs before login."""
    fb, problem = s.firebase_web_result if s.auth_mode != "off" else (None, None)
    return {"auth_mode": s.auth_mode, "dev_login": s.auth_mode == "jwt" and s.app_env != "prod", "firebase": fb,
            "google_login": bool(fb) and s.firebase_enabled, "google_login_problem": problem, "email_password_login": s.auth_mode == "firebase" and bool(fb),
            "allowed_domains": s.allowed_domain_list, "llm_enabled": s.llm_enabled,
            "embedding_backend": s.embedding_backend, "env": s.app_env, "version": app.version}


# ------------------------------------------------------------- middleware
@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    request_id_var.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error")
        response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    m.HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    m.HTTP_LATENCY.labels(path).observe(time.perf_counter() - t0)
    response.headers["x-request-id"] = rid
    return response


def limited(user: User = Depends(get_current_user), ctx: Context = Depends(get_context)) -> User:
    n = ctx.kv.incr_window(f"rl:{user.uid}", 60)
    if n > ctx.settings.rate_limit_per_minute:
        m.RATE_LIMITED.inc()
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
    return user


agent = Depends(limited)
admin = Depends(require_role("admin"))


def _job(db: Session, kind: str, total: int) -> IngestJob:
    j = IngestJob(kind=kind, total=total, status="pending")
    db.add(j)
    db.commit()
    return j


def _job_out(j: IngestJob) -> sc.JobOut:
    return sc.JobOut(job_id=j.id, kind=j.kind, status=j.status, total=j.total, processed=j.processed,
                     inserted=j.inserted, updated=j.updated, skipped=j.skipped, error=j.error)


# ------------------------------------------------------------------ health
@app.get("/health", response_model=sc.Health, tags=["ops"])
def health():
    return sc.Health(status="ok")


@app.get("/ready", response_model=sc.Health, tags=["ops"])
def ready(response: Response, db: Session = Depends(get_db), ctx: Context = Depends(get_context)):
    checks: dict = {}
    try:
        db.execute(select(1))
        checks["postgres"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["postgres"] = f"error: {e.__class__.__name__}"
    checks["vector_db"] = "ok" if ctx.vectors.ping() else "error"
    checks["redis"] = "ok" if ctx.kv.ping() else "error"
    checks["llm"] = "configured" if ctx.llm.enabled else "not configured (extractive fallback)"
    try:
        info = indexer.get_active(ctx, db)
        checks["index"] = info.collection
        if (info.backend != ctx.settings.embedding_backend):
            checks["index_warning"] = "active index backend differs from EMBEDDING_BACKEND; run /v1/admin/reindex"
    except Exception as e:  # noqa: BLE001
        checks["index"] = f"error: {e.__class__.__name__}"
    bad = [k for k in ("postgres", "vector_db", "index") if str(checks.get(k, "")).startswith("error")]
    if bad:
        response.status_code = 503
    return sc.Health(status="degraded" if bad else "ready", checks=checks)


@app.get("/metrics", include_in_schema=False)
def metrics(db: Session = Depends(get_db), ctx: Context = Depends(get_context)):
    try:
        info = indexer.get_active(ctx, db)
        for dt in ("ticket", "kb"):
            m.INDEX_DOCS.labels(dt).set(ctx.vectors.count(info.collection, {"doc_type": dt}))
        pending = db.execute(select(func.count()).select_from(Ticket).where(
            Ticket.status == "resolved", (Ticket.indexed_collection.is_(None)) | (Ticket.indexed_collection != info.collection))).scalar() or 0
        m.UNINDEXED.set(pending)
        qd = ctx.kv.queue_depth("celery")
        if qd is not None:
            m.QUEUE_DEPTH.set(qd)
    except Exception:  # noqa: BLE001  metrics must never fail the scrape
        log.exception("metrics refresh failed")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -------------------------------------------------------------------- auth
@app.post("/v1/auth/dev-token", tags=["auth"], include_in_schema=True)
def dev_token(uid: str = "agent-1", role: str = "agent", s: Settings = Depends(get_settings)):
    """Local development only (AUTH_MODE=jwt, APP_ENV!=prod). Production uses Firebase ID tokens."""
    if s.auth_mode != "jwt" or s.app_env == "prod":
        raise HTTPException(404)
    return {"access_token": mint_dev_token(uid, role, s), "token_type": "bearer"}


@app.get("/v1/me", response_model=sc.Me, tags=["auth"])
def me(user: User = Depends(get_current_user)):
    return sc.Me(uid=user.uid, role=user.role, email=user.email, name=user.name, provider=user.provider)


# ------------------------------------------------------- core agent features
@app.post("/v1/analyze", response_model=sc.Analysis, tags=["agent"])
def analyze_complaint(body: sc.ComplaintIn, user: User = agent, db: Session = Depends(get_db),
                      ctx: Context = Depends(get_context)):
    """Parse a raw complaint: category/intent, product, severity, sentiment (no generation)."""
    info = indexer.get_active(ctx, db)
    clean = resolver._clean(ctx, body.text)
    tickets, _ = retriever.retrieve(ctx, info, clean, knn_k=ctx.settings.knn_k, kb_k=0)
    return analyzer.analyze(ctx, clean, tickets, taxonomy.active_categories(db))


@app.post("/v1/search", response_model=sc.SearchOut, tags=["agent"])
def semantic_search(body: sc.SearchIn, user: User = agent, db: Session = Depends(get_db),
                    ctx: Context = Depends(get_context)):
    return resolver.search(ctx, db, body.text, body.top_k, body.doc_type, body.category, body.product)


@app.post("/v1/resolve", response_model=sc.ResolveOut, tags=["agent"])
def resolve_complaint(body: sc.ResolveIn, request: Request, user: User = agent, db: Session = Depends(get_db),
                      ctx: Context = Depends(get_context)):
    """Full pipeline: parse -> semantic retrieval -> grounded, cited resolution (RAG)."""
    return resolver.resolve(ctx, db, body.text, user.uid, top_k_tickets=body.top_k_tickets, top_k_kb=body.top_k_kb,
                            use_cache=body.use_cache, request_id=request_id_var.get())


@app.post("/v1/feedback", tags=["agent"])
def submit_feedback(body: sc.FeedbackIn, user: User = agent, db: Session = Depends(get_db),
                    ctx: Context = Depends(get_context)):
    try:
        return feedback.record_feedback(ctx, db, body, user.uid)
    except LookupError:
        raise HTTPException(404, "unknown query_id") from None


# --------------------------------------------------------------- ingestion
@app.post("/v1/ingest/tickets", response_model=sc.JobOut, status_code=202, tags=["ingestion"])
def ingest_tickets(body: sc.TicketBatch, user: User = admin, db: Session = Depends(get_db)):
    """Async, idempotent upsert of (resolved) tickets. Poll /v1/jobs/{id}."""
    j = _job(db, "tickets", len(body.tickets))
    tasks.ingest_tickets_task.delay(j.id, [t.model_dump() for t in body.tickets])
    db.refresh(j)
    return _job_out(j)


@app.post("/v1/ingest/kb", response_model=sc.JobOut, status_code=202, tags=["ingestion"])
def ingest_kb(body: sc.KBBatch, user: User = admin, db: Session = Depends(get_db)):
    """Async, idempotent upsert of KB articles (version bump / deactivate supported)."""
    j = _job(db, "kb", len(body.articles))
    tasks.ingest_kb_task.delay(j.id, [a.model_dump() for a in body.articles])
    db.refresh(j)
    return _job_out(j)


@app.get("/v1/jobs/{job_id}", response_model=sc.JobOut, tags=["ingestion"])
def get_job(job_id: str, user: User = admin, db: Session = Depends(get_db)):
    j = db.get(IngestJob, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    db.refresh(j)
    return _job_out(j)


# ------------------------------------------------------------------- admin
@app.get("/v1/categories", response_model=list[sc.CategoryOut], tags=["taxonomy"])
def list_categories(user: User = agent, db: Session = Depends(get_db)):
    return taxonomy.list_categories(db)


@app.post("/v1/admin/categories", response_model=sc.JobOut | sc.CategoryOut, status_code=201, tags=["taxonomy"])
def add_category(body: sc.CategoryIn, user: User = admin, db: Session = Depends(get_db)):
    """Introduce a new ticket class at runtime. Seed examples are indexed async; the kNN classifier
    can predict the class as soon as they land - no retraining or redeploy."""
    name = taxonomy.ensure_category(db, body.name, created_by=user.uid, description=body.description)
    write_audit(db, user.uid, "category.add", {"name": name, "seeds": len(body.seed_examples)})
    if not body.seed_examples:
        return sc.CategoryOut(name=name, description=body.description, active=True)
    seeds = [e.model_copy(update={"category": name}).model_dump() for e in body.seed_examples]
    j = _job(db, "seed_category", len(seeds))
    tasks.seed_category_task.delay(j.id, seeds)
    db.refresh(j)
    return _job_out(j)


@app.post("/v1/admin/categories/{name}/merge", tags=["taxonomy"])
def merge_category(name: str, body: sc.MergeIn, user: User = admin, db: Session = Depends(get_db),
                   ctx: Context = Depends(get_context)):
    try:
        return taxonomy.merge_category(ctx, db, name, body.into, user.uid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.post("/v1/admin/reindex", response_model=sc.JobOut, status_code=202, tags=["admin"])
def reindex(body: sc.ReindexIn, user: User = admin, db: Session = Depends(get_db)):
    """Blue/green re-index (e.g. new embedding model). Serving continues on the old index until the swap."""
    j = _job(db, "reindex", 0)
    tasks.reindex_task.delay(j.id, body.embedding_backend, body.embedding_model, user.uid)
    db.refresh(j)
    return _job_out(j)


@app.get("/v1/admin/drift", tags=["admin"])
def drift_report(window_days: int = 7, user: User = admin, db: Session = Depends(get_db)):
    return drift.drift_report(db, window_days)


@app.get("/v1/admin/emerging", tags=["admin"])
def emerging(days: int = 14, min_size: int = 3, user: User = admin, db: Session = Depends(get_db),
             ctx: Context = Depends(get_context)):
    """Clusters of complaints that matched no known class: candidates for new categories."""
    return drift.emerging_topics(ctx, db, days=days, min_size=min_size)


@app.get("/v1/admin/stats", tags=["admin"])
def stats(user: User = admin, db: Session = Depends(get_db), ctx: Context = Depends(get_context)):
    info = indexer.get_active(ctx, db)
    return {"index": info.collection, "embedder": info.embedder.name, "dim": info.dim,
            "tickets_db": db.scalar(select(func.count()).select_from(Ticket)),
            "kb_db": db.scalar(select(func.count()).select_from(KBArticle)),
            "vectors_ticket": ctx.vectors.count(info.collection, {"doc_type": "ticket"}),
            "vectors_kb": ctx.vectors.count(info.collection, {"doc_type": "kb"})}


@app.get("/v1/admin/audit/verify", tags=["admin"])
def audit_verify(user: User = admin, db: Session = Depends(get_db)):
    return verify_chain(db)


@app.get("/v1/admin/jobs", response_model=list[sc.JobOut], tags=["admin"])
def list_jobs(limit: int = 20, user: User = admin, db: Session = Depends(get_db)):
    rows = db.execute(select(IngestJob).order_by(IngestJob.created_at.desc()).limit(min(limit, 100))).scalars()
    return [_job_out(j) for j in rows]


@app.post("/v1/admin/seed-demo", response_model=sc.JobOut, status_code=202, tags=["admin"])
def seed_demo(novel: bool = False, user: User = admin, db: Session = Depends(get_db)):
    """Load the bundled synthetic data (`novel=true` also loads the held-out 5G class)."""
    j = _job(db, "seed_demo", 0)
    tasks.seed_demo_task.delay(j.id, novel)
    db.refresh(j)
    return _job_out(j)


@app.post("/v1/admin/import-hf", response_model=sc.JobOut, status_code=202, tags=["admin"])
def import_hf(limit: int = 3000, kb_limit: int = 300, lang: str = "en", user: User = admin, db: Session = Depends(get_db)):
    """Add the public Hugging Face support-ticket dataset: resolved tickets plus KB articles derived from the best cases.
    The server needs network access and the `datasets` package; progress and errors appear in the job."""
    j = _job(db, "import_hf", 0)
    tasks.import_hf_task.delay(j.id, min(max(limit, 0), 50000), min(max(kb_limit, 0), 5000), lang)
    db.refresh(j)
    return _job_out(j)
