from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from app.core.vectorstore import Hit

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def scores(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [float(x) for x in self.model.predict([(query, p) for p in passages])]


@lru_cache
def load_reranker(name: str, model_name: str) -> Reranker | None:
    if name != "cross-encoder":
        return None
    try:
        return CrossEncoderReranker(model_name)
    except Exception as e:
        log.warning("cross-encoder unavailable (%s); keeping the first-stage order", e)
        return None


def rerank_hits(reranker: Reranker | None, query: str, hits: list[Hit]) -> list[Hit]:
    if reranker is None or len(hits) < 2:
        return hits
    passages = [f"{h.payload.get('title', '')}\n{h.payload.get('text', '')}" for h in hits]
    order = sorted(zip(reranker.scores(query, passages), range(len(hits))), reverse=True)
    return [hits[i] for _, i in order]
