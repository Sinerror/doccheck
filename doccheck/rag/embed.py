"""Local embeddings via fastembed (ONNX, CPU): dense model is configurable + sparse BM25 with a Russian stemmer.

The dense model is chosen by DOCCHECK_EMBED_MODEL (see `scripts/eval_retrieval.py` for the comparison
that picked the default). e5 models are trained with "query: " / "passage: " prefixes and fastembed
does not add them, so we do.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

from qdrant_client import models

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
BM25_LANGUAGE = "russian"

KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "intfloat/multilingual-e5-large": 1024,
}

_lock = threading.Lock()


def dense_model_name() -> str:
    return os.environ.get("DOCCHECK_EMBED_MODEL") or DEFAULT_DENSE_MODEL


def dense_dim(model: str | None = None) -> int:
    model = model or dense_model_name()
    if model in KNOWN_DIMS:
        return KNOWN_DIMS[model]
    from fastembed import TextEmbedding
    for m in TextEmbedding.list_supported_models():
        if m["model"] == model:
            return int(m["dim"])
    raise ValueError(f"unknown embedding model {model!r}")


def _is_e5(model: str) -> bool:
    return "e5" in model.lower()


@lru_cache(maxsize=4)
def _dense(model: str):
    from fastembed import TextEmbedding
    return TextEmbedding(model)


@lru_cache(maxsize=1)
def _sparse():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(SPARSE_MODEL, language=BM25_LANGUAGE)


def embed_dense(texts: list[str], model: str | None = None) -> list[list[float]]:
    model = model or dense_model_name()
    if _is_e5(model):
        texts = [f"passage: {t}" for t in texts]
    with _lock:
        return [v.tolist() for v in _dense(model).embed(texts, batch_size=16)]


def embed_query_dense(text: str, model: str | None = None) -> list[float]:
    model = model or dense_model_name()
    with _lock:
        if _is_e5(model):
            return next(iter(_dense(model).embed([f"query: {text}"]))).tolist()
        return next(iter(_dense(model).query_embed(text))).tolist()


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    with _lock:
        return [models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
                for v in _sparse().embed(texts, batch_size=32)]


def embed_query_sparse(text: str) -> models.SparseVector:
    with _lock:
        v = next(iter(_sparse().query_embed(text)))
        return models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
