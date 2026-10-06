from celery import Celery

from app.config import get_settings

s = get_settings()
broker = s.redis_url or "memory://"
celery_app = Celery("tra", broker=broker, backend=s.redis_url or "cache+memory://", include=["app.worker.tasks"])
celery_app.conf.update(
    task_always_eager=s.celery_eager,
    task_eager_propagates=True,
    task_acks_late=True,                 # re-deliver if a worker dies mid-task (tasks are idempotent)
    worker_prefetch_multiplier=1,        # fair scheduling for long ingest/re-index tasks
    task_time_limit=3600,
    task_serializer="json",
    result_expires=3600,
    beat_schedule={"reconcile-unindexed": {"task": "tra.reconcile", "schedule": 300.0}},
)
