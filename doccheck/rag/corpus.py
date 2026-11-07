"""Index the company corpus once per content hash and reuse it across runs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..sources.local import CorpusDoc, corpus_fingerprint, load_corpus
from .chunking import chunk_document
from .qdrant_store import QdrantStore

log = logging.getLogger(__name__)


@dataclass
class CorpusIndex:
    scope: str
    docs: list[CorpusDoc]
    chunks: int
    reused: bool


def index_corpus(store: QdrantStore, root: Path, force: bool = False) -> CorpusIndex:
    """Scope = `corpus:<hash of files + embedding model>`. If points for that scope already exist the corpus
    is not re-embedded: on a laptop CPU that saves ~1 minute per run with a large multilingual model."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus folder not found: {root}")
    scope = "corpus:" + corpus_fingerprint(root, extra=store.dense_model)
    docs = load_corpus(root)
    if not docs:
        raise ValueError(f"no supported documents in {root}")
    existing = store.count(scope)
    if existing and not force:
        return CorpusIndex(scope=scope, docs=docs, chunks=existing, reused=True)
    if existing:
        store.delete_scope(scope)
    total = 0
    for d in docs:
        chunks = chunk_document(d.parsed.sections)
        total += store.upsert_chunks(scope, {"doc_id": d.doc_id, "title": d.title}, chunks, "corpus")
    log.info("indexed corpus %s: %d docs, %d chunks", scope, len(docs), total)
    return CorpusIndex(scope=scope, docs=docs, chunks=total, reused=False)
