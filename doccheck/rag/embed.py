"""Local embeddings via fastembed: dense bge-small (384, cosine) + sparse BM25."""
from __future__ import annotations

import threading
from functools import lru_cache

from qdrant_client import models

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 384

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _dense():
    from fastembed import TextEmbedding
    return TextEmbedding(DENSE_MODEL)


@lru_cache(maxsize=1)
def _sparse():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(SPARSE_MODEL)


def embed_dense(texts: list[str]) -> list[list[float]]:
    with _lock:
        return [v.tolist() for v in _dense().embed(texts, batch_size=32)]


def embed_query_dense(text: str) -> list[float]:
    with _lock:
        return next(iter(_dense().query_embed(text))).tolist()


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    with _lock:
        return [models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
                for v in _sparse().embed(texts, batch_size=32)]


def embed_query_sparse(text: str) -> models.SparseVector:
    with _lock:
        v = next(iter(_sparse().query_embed(text)))
        return models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
