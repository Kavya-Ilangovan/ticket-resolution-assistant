"""Embedding backends behind one interface.

* sentence-transformers  -> real semantic embeddings (default in Docker)
* hashing                -> dependency-free signed feature hashing over words + char n-grams.
                            Lexical-ish; exists so CI, offline dev and the test-suite run anywhere.
"""
from __future__ import annotations

import threading
import zlib
from typing import Protocol

import numpy as np

from app.core.text import content_tokens, stem


def sparse_encode(text: str) -> tuple[list[int], list[float]]:
    """Lexical (BM25-style) sparse vector: hashed stems with log-tf. IDF is applied server-side by Qdrant
    (Modifier.IDF), so it stays correct as the corpus evolves without refitting anything."""
    counts: dict[int, float] = {}
    for t in content_tokens(text):
        idx = zlib.crc32(stem(t).encode()) & 0x7FFFFFFF
        counts[idx] = counts.get(idx, 0.0) + 1.0
    idx = sorted(counts)
    return idx, [1.0 + float(np.log(counts[i])) for i in idx]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _features(self, text: str) -> list[str]:
        toks = [stem(t) for t in content_tokens(text)]
        feats = list(toks)
        feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            w = f"<{t}>"
            feats += [w[i : i + 4] for i in range(max(1, len(w) - 3))]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for f in self._features(text):
                h = zlib.crc32(f.encode())
                idx = h % self.dim
                sign = 1.0 if (h >> 16) & 1 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, v in counts.items():
                out[row, idx] = np.sign(v) * np.log1p(abs(v))
            n = np.linalg.norm(out[row])
            if n > 0:
                out[row] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False),
            dtype=np.float32,
        )


class EmbedderRegistry:
    """Caches embedders by (backend, model) so several index versions can coexist during a re-index."""

    def __init__(self, hashing_dim: int = 1024):
        self._cache: dict[tuple[str, str], Embedder] = {}
        self._lock = threading.Lock()
        self._hashing_dim = hashing_dim

    def get(self, backend: str, model: str) -> Embedder:
        key = (backend, model)
        with self._lock:
            if key not in self._cache:
                if backend == "hashing":
                    self._cache[key] = HashingEmbedder(self._hashing_dim)
                elif backend == "sentence-transformers":
                    self._cache[key] = SentenceTransformerEmbedder(model)
                else:
                    raise ValueError(f"unknown embedding backend {backend!r}")
            return self._cache[key]
