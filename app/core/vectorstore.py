"""Thin wrapper over Qdrant. QDRANT_URL: http://host:6333 (server) | path:./.qdrant (embedded, persistent) | :memory:"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

_NS = uuid.UUID("6f1c2b7e-8d0e-4c59-9d8f-3a6c2f0e9b11")


def point_id(doc_type: str, external_id: str, chunk: int = 0) -> str:
    return str(uuid.uuid5(_NS, f"{doc_type}:{external_id}:{chunk}"))


@dataclass
class Hit:
    id: str
    score: float                 # dense cosine similarity (comparable across queries; used for thresholds)
    payload: dict[str, Any]
    rrf: float = 0.0             # fused rank score (hybrid ordering)


def _filter(conds: dict[str, Any] | None) -> qm.Filter | None:
    if not conds:
        return None
    must = []
    for k, v in conds.items():
        if isinstance(v, (list, tuple, set)):
            must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=list(v))))
        else:
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
    return qm.Filter(must=must)


class VectorStore:
    def __init__(self, url: str, api_key: str | None = None):
        self.url = url
        if url == ":memory:":
            self.client = QdrantClient(":memory:")
        elif url.startswith("path:"):      # embedded, persistent, single-process (local dev without Docker)
            self.client = QdrantClient(path=url[5:])
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=20)

    def ensure_collection(self, name: str, dim: int) -> None:
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        )
        if self.url == ":memory:" or self.url.startswith("path:"):
            return   # the embedded engine ignores payload indexes (and warns about it)
        for field in ("doc_type", "category", "product", "external_id"):
            self.client.create_payload_index(name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)

    def drop_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray, sparse: list[tuple[list[int], list[float]]],
               payloads: list[dict]) -> None:
        points = [qm.PointStruct(id=i, vector={"dense": v.tolist(), "sparse": qm.SparseVector(indices=sp[0], values=sp[1])},
                                 payload=p) for i, v, sp, p in zip(ids, vectors, sparse, payloads)]
        for i in range(0, len(points), 256):
            self.client.upsert(collection, points=points[i : i + 256], wait=True)

    def search(self, collection: str, dense: np.ndarray, sparse: tuple[list[int], list[float]], limit: int,
               where: dict[str, Any] | None = None, mode: str = "hybrid") -> list[Hit]:
        """mode: dense | sparse | hybrid (Reciprocal Rank Fusion of both rankings, k=60)."""
        flt = _filter(where)
        pool = limit if mode != "hybrid" else max(limit * 4, 30)
        d_pts, s_pts = [], []
        if mode in ("dense", "hybrid"):
            d_pts = self.client.query_points(collection, query=dense.tolist(), using="dense", limit=pool,
                                             query_filter=flt, with_payload=True).points
        if mode in ("sparse", "hybrid") and sparse[0]:
            s_pts = self.client.query_points(collection, query=qm.SparseVector(indices=sparse[0], values=sparse[1]),
                                             using="sparse", limit=pool, query_filter=flt, with_payload=True,
                                             with_vectors=["dense"]).points
        hits: dict[str, Hit] = {}
        for rank, p in enumerate(d_pts):
            hits[str(p.id)] = Hit(str(p.id), float(p.score), dict(p.payload or {}), 1.0 / (60 + rank + 1))
        for rank, p in enumerate(s_pts):
            pid = str(p.id)
            if pid in hits:
                hits[pid].rrf += 1.0 / (60 + rank + 1)
            else:  # sparse-only candidate: compute its dense cosine so thresholds stay comparable
                v = np.asarray((p.vector or {}).get("dense", []), dtype=np.float32)
                cos = float(v @ dense / ((np.linalg.norm(v) * np.linalg.norm(dense)) or 1.0)) if v.size else 0.0
                hits[pid] = Hit(pid, cos, dict(p.payload or {}), 1.0 / (60 + rank + 1))
        order = (sorted(hits.values(), key=lambda h: (-h.rrf, -h.score, h.id)) if mode != "dense"
                 else sorted(hits.values(), key=lambda h: (-h.score, h.id)))
        return order[:limit]

    def delete_where(self, collection: str, where: dict[str, Any]) -> None:
        self.client.delete(collection, points_selector=qm.FilterSelector(filter=_filter(where)), wait=True)

    def set_payload_where(self, collection: str, payload: dict[str, Any], where: dict[str, Any]) -> None:
        self.client.set_payload(collection, payload=payload, points=qm.FilterSelector(filter=_filter(where)), wait=True)

    def count(self, collection: str, where: dict[str, Any] | None = None) -> int:
        if not self.client.collection_exists(collection):
            return 0
        return int(self.client.count(collection, count_filter=_filter(where), exact=True).count)

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
