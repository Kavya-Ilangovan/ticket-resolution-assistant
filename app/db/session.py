from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.db.models import Base

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(settings: Settings | None = None, *, create_tables: bool = True):
    """(Re)initialise the global engine. Safe to call repeatedly (tests/evals do)."""
    global _engine, _SessionLocal
    settings = settings or get_settings()
    kwargs: dict = {"pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in settings.database_url or settings.database_url in ("sqlite://", "sqlite:///"):
            kwargs["poolclass"] = StaticPool
    else:
        kwargs.update(pool_size=10, max_overflow=20)
    _engine = create_engine(settings.database_url, **kwargs)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    if create_tables:  # prod: manage schema with Alembic migrations instead
        Base.metadata.create_all(_engine)
    return _engine


def get_engine():
    if _engine is None:
        init_engine()
    return _engine


def session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    s = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    s = session_factory()()
    try:
        yield s
    finally:
        s.close()
