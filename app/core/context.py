"""Process-wide wiring (vector store, KV, LLM, embedder registry). Replaceable for tests/evals."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.embeddings import EmbedderRegistry
from app.core.kv import KV, build_kv
from app.core.llm import LLMClient
from app.core.vectorstore import VectorStore


@dataclass
class Context:
    settings: Settings
    vectors: VectorStore
    kv: KV
    llm: LLMClient
    embedders: EmbedderRegistry


_ctx: Context | None = None


def build_context(settings: Settings | None = None) -> Context:
    s = settings or get_settings()
    return Context(
        settings=s,
        vectors=VectorStore(s.qdrant_url, s.qdrant_api_key),
        kv=build_kv(s.redis_url),
        llm=LLMClient(s),
        embedders=EmbedderRegistry(s.hashing_dim),
    )


def get_context() -> Context:
    global _ctx
    if _ctx is None:
        _ctx = build_context()
    return _ctx


def set_context(ctx: Context | None) -> None:
    global _ctx
    _ctx = ctx
