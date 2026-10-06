import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.context import build_context, set_context
from app.db.session import init_engine, session_scope
from scripts.seed import read, seed

ENV = dict(DATABASE_URL="sqlite:///:memory:", QDRANT_URL=":memory:", AUTH_MODE="jwt", JWT_SECRET="test-secret",
           EMBEDDING_BACKEND="hashing", RATE_LIMIT_PER_MINUTE="1000", CELERY_EAGER="true", APP_ENV="test", OPENROUTER_API_KEY="")


@pytest.fixture
def settings(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    set_context(None)
    return get_settings()


@pytest.fixture
def ctx(settings):
    """Isolated in-memory stack (SQLite + embedded Qdrant), no HTTP."""
    init_engine(settings)
    c = build_context(settings)
    set_context(c)
    yield c
    set_context(None)


@pytest.fixture
def db(ctx):
    with session_scope() as s:
        yield s


@pytest.fixture
def seeded(ctx, db):
    seed(ctx, db)
    return ctx


@pytest.fixture
def client(settings):
    from app.api.main import app
    with TestClient(app) as c:
        yield c
    set_context(None)


def _token(client, role, uid):
    r = client.post("/v1/auth/dev-token", params={"uid": uid, "role": role})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def agent_h(client):
    return _token(client, "agent", "u1")


@pytest.fixture
def admin_h(client):
    return _token(client, "admin", "root")


@pytest.fixture
def seeded_client(client, admin_h):
    for path, key, name in (("tickets", "tickets", "tickets"), ("kb", "articles", "kb")):
        rows = read(name)[:200] if name == "tickets" else read(name)
        job = client.post(f"/v1/ingest/{path}", json={key: rows}, headers=admin_h).json()
        assert client.get(f"/v1/jobs/{job['job_id']}", headers=admin_h).json()["status"] == "done"
    return client
