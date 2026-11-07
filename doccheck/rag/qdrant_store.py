"""Qdrant store: named dense + sparse (BM25, IDF) vectors, hybrid retrieval with RRF.

Points live in *scopes*:
  * the company corpus is indexed once under `corpus:<content hash>` and reused by every run
    until a file in the corpus folder changes;
  * the document under check is indexed under the run id (so the critic can read it, and the UI can draw it).
One collection per dense model (`doccheck_<model slug>`), because the vector size differs.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from . import embed
from .chunking import Chunk
from .projection import PCABasis, fit_pca

DENSE = "dense"
SPARSE = "bm25"
INDEXED_FIELDS = ("scope", "doc_id", "source")
COLLECTION_PREFIX = "doccheck_"


class IncompatibleSchema(RuntimeError):
    pass


def collection_for(base: str, model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.split("/")[-1].lower()).strip("_")
    return f"{base}_{slug}"


def point_id(scope: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{scope}:{chunk_id}"))


def make_chunk_id(doc_id: str, chunk_idx: int) -> str:
    return f"{doc_id}#{chunk_idx}"


@dataclass
class Hit:
    point_id: str
    chunk_id: str
    score: float
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


def embed_text(title: str | None, section: str | None, text: str) -> str:
    """What gets embedded: the heading path matters for retrieval ("Тариф «Бизнес» (Казахстан)"),
    but the stored `text` (what quotes are checked against) is the chunk alone."""
    head = [x for x in (title, section) if x]
    if len(head) == 2 and head[0] == head[1]:
        head = head[:1]
    return " / ".join(head) + "\n" + text if head else text


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dense_model: str | None = None):
        assert collection.startswith(COLLECTION_PREFIX), "collections must start with doccheck_"
        self.client = client
        self.dense_model = dense_model or embed.dense_model_name()
        self.dim = embed.dense_dim(self.dense_model)
        self.collection = collection
        self._bases: dict[str, PCABasis] = {}

    # -- schema ------------------------------------------------------------------

    def schema_compatible(self) -> bool:
        params = self.client.get_collection(self.collection).config.params
        vecs = params.vectors
        if not isinstance(vecs, dict) or DENSE not in vecs:
            return False
        dv = vecs[DENSE]
        if dv.size != self.dim or dv.distance != models.Distance.COSINE:
            return False
        sv = params.sparse_vectors or {}
        return SPARSE in sv and sv[SPARSE].modifier == models.Modifier.IDF

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            if not self.schema_compatible():
                raise IncompatibleSchema(f"collection {self.collection!r} exists with an incompatible schema")
        else:
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        for f in INDEXED_FIELDS:
            try:
                self.client.create_payload_index(self.collection, f, models.PayloadSchemaType.KEYWORD)
            except Exception:  # already exists / unsupported in :memory:
                pass

    # -- write ---------------------------------------------------------------------

    def upsert_chunks(self, scope: str, doc: dict[str, Any], chunks: list[Chunk], source: str) -> int:
        """Idempotent: the point id is a hash of (scope, chunk_id), so re-indexing overwrites."""
        if not chunks:
            return 0
        texts = [embed_text(doc.get("title"), c.section, c.text) for c in chunks]
        dense = embed.embed_dense(texts, self.dense_model)
        sparse = embed.embed_sparse(texts)
        points = []
        for c, dv, sv in zip(chunks, dense, sparse):
            cid = make_chunk_id(doc["doc_id"], c.chunk_idx)
            payload = {"scope": scope, "doc_id": doc["doc_id"], "source": source, "title": doc.get("title"),
                       "section": c.section, "chunk_idx": c.chunk_idx, "chunk_id": cid, "text": c.text}
            points.append(models.PointStruct(id=point_id(scope, cid), vector={DENSE: dv, SPARSE: sv},
                                             payload=payload))
        for i in range(0, len(points), 64):
            self.client.upsert(self.collection, points=points[i:i + 64], wait=True)
        self._bases.clear()
        return len(points)

    def delete_scope(self, scope: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=self._filter([scope])),
                           wait=True)
        self._bases.clear()

    # -- read ----------------------------------------------------------------------

    @staticmethod
    def _filter(scopes: Iterable[str], source: str | None = None) -> models.Filter:
        must: list[Any] = [models.FieldCondition(key="scope", match=models.MatchAny(any=list(scopes)))]
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        return models.Filter(must=must)

    def hybrid_search(self, scopes: list[str], query: str, k: int = 8, source: str | None = None,
                      dense_vec: list[float] | None = None, mode: str = "hybrid") -> list[Hit]:
        """mode: hybrid (dense + BM25 → RRF), dense or sparse. The last two exist for the retrieval eval."""
        flt = self._filter(scopes, source)
        if mode == "sparse":
            res = self.client.query_points(self.collection, query=embed.embed_query_sparse(query), using=SPARSE,
                                           query_filter=flt, limit=k, with_payload=True)
        else:
            dense_vec = dense_vec or embed.embed_query_dense(query, self.dense_model)
            if mode == "dense":
                res = self.client.query_points(self.collection, query=dense_vec, using=DENSE, query_filter=flt,
                                               limit=k, with_payload=True)
            else:
                res = self.client.query_points(
                    self.collection,
                    prefetch=[models.Prefetch(query=dense_vec, using=DENSE, limit=30, filter=flt),
                              models.Prefetch(query=embed.embed_query_sparse(query), using=SPARSE, limit=30,
                                              filter=flt)],
                    query=models.FusionQuery(fusion=models.Fusion.RRF), query_filter=flt, limit=k,
                    with_payload=True)
        return [Hit(point_id=str(p.id), chunk_id=p.payload.get("chunk_id", ""), score=float(p.score),
                    payload=p.payload) for p in res.points]

    def get_chunk(self, scopes: list[str], chunk_id: str) -> Hit | None:
        try:
            recs = self.client.retrieve(self.collection, ids=[point_id(s, chunk_id) for s in scopes],
                                        with_payload=True)
        except Exception:
            return None
        if not recs:
            return None
        return Hit(point_id=str(recs[0].id), chunk_id=chunk_id, score=1.0, payload=recs[0].payload)

    def count(self, scope: str | None = None) -> int:
        flt = self._filter([scope]) if scope else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def scroll(self, scopes: list[str], with_vectors: bool = True) -> list[models.Record]:
        out: list[models.Record] = []
        offset = None
        while True:
            recs, offset = self.client.scroll(self.collection, scroll_filter=self._filter(scopes), limit=256,
                                              offset=offset, with_payload=True,
                                              with_vectors=[DENSE] if with_vectors else False)
            out.extend(recs)
            if offset is None:
                break
        return out

    # -- projection (for the UI's 2D map) ------------------------------------------

    @staticmethod
    def _dense_of(rec: models.Record) -> list[float]:
        v = rec.vector
        return v[DENSE] if isinstance(v, dict) else v

    def basis(self, scopes: list[str]) -> PCABasis | None:
        key = "|".join(sorted(scopes))
        if key not in self._bases:
            recs = self.scroll(scopes)
            if not recs:
                return None
            self._bases[key] = fit_pca(np.array([self._dense_of(r) for r in recs]))
        return self._bases[key]

    def vector_map(self, scopes: list[str]) -> list[dict[str, Any]]:
        recs = self.scroll(scopes)
        if not recs:
            return []
        vecs = np.array([self._dense_of(r) for r in recs])
        basis = fit_pca(vecs)
        self._bases["|".join(sorted(scopes))] = basis
        xy = basis.project(vecs)
        return [{"id": str(r.id), "x": round(float(p[0]), 5), "y": round(float(p[1]), 5),
                 "doc_id": r.payload.get("doc_id"), "source": r.payload.get("source"),
                 "title": r.payload.get("title"), "section": r.payload.get("section"),
                 "chunk_id": r.payload.get("chunk_id")}
                for r, p in zip(recs, xy)]
