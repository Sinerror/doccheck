"""Qdrant corpus store (SPEC §4): named dense + sparse(bm25, idf) vectors, hybrid RRF retrieval."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from . import embed
from .chunking import Chunk
from .projection import PCABasis, fit_pca

DENSE = "dense"
SPARSE = "bm25"
INDEXED_FIELDS = ("run_id", "paper_id", "source")


class IncompatibleSchema(RuntimeError):
    pass


def point_id(run_id: str, chunk_id: str) -> str:
    # SPEC: uuid5(NAMESPACE_URL, f"{paper_id}:{chunk_idx}"). We prefix run_id so two runs that share
    # a related paper don't steal each other's points (payload run_id would be overwritten). See DECISIONS.md.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{chunk_id}"))


def make_chunk_id(paper_id: str, chunk_idx: int) -> str:
    return f"{paper_id}#{chunk_idx}"


@dataclass
class Hit:
    point_id: str
    chunk_id: str
    score: float
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection
        self._bases: dict[str, PCABasis] = {}

    # -- schema ------------------------------------------------------------------

    def schema_compatible(self) -> bool:
        info = self.client.get_collection(self.collection)
        params = info.config.params
        vecs = params.vectors
        if not isinstance(vecs, dict) or DENSE not in vecs:
            return False
        dv = vecs[DENSE]
        if dv.size != embed.DENSE_DIM or dv.distance != models.Distance.COSINE:
            return False
        sv = params.sparse_vectors or {}
        if SPARSE not in sv:
            return False
        mod = sv[SPARSE].modifier
        return mod == models.Modifier.IDF

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            if not self.schema_compatible():
                raise IncompatibleSchema(f"collection {self.collection!r} exists with an incompatible schema")
        else:
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=embed.DENSE_DIM, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        for f in INDEXED_FIELDS:
            try:
                self.client.create_payload_index(self.collection, f, models.PayloadSchemaType.KEYWORD)
            except Exception:  # already exists / unsupported in :memory:
                pass

    # -- write ---------------------------------------------------------------------

    def upsert_chunks(self, run_id: str, paper: dict[str, Any], chunks: list[Chunk], source: str) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        dense = embed.embed_dense(texts)
        sparse = embed.embed_sparse(texts)
        points = []
        for c, dv, sv in zip(chunks, dense, sparse):
            cid = make_chunk_id(paper["paper_id"], c.chunk_idx)
            payload = {
                "run_id": run_id, "paper_id": paper["paper_id"], "source": source,
                "title": paper.get("title"), "authors": paper.get("authors") or [],
                "year": paper.get("year"), "venue": paper.get("venue"), "url": paper.get("url"),
                "section": c.section, "chunk_idx": c.chunk_idx, "chunk_id": cid, "text": c.text,
            }
            points.append(models.PointStruct(id=point_id(run_id, cid), vector={DENSE: dv, SPARSE: sv},
                                             payload=payload))
        for i in range(0, len(points), 64):
            self.client.upsert(self.collection, points=points[i:i + 64], wait=True)
        self._bases.pop(run_id, None)
        return len(points)

    def delete_run(self, run_id: str) -> None:
        """Drop this run's points (only inside our own doccheck_* collection) before re-running a run_id."""
        assert self.collection.startswith("doccheck_"), "refusing to delete outside doccheck_* collections"
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=self._filter(run_id)),
                           wait=True)
        self._bases.pop(run_id, None)

    # -- read ----------------------------------------------------------------------

    @staticmethod
    def _filter(run_id: str, source: str | None = None, paper_id: str | None = None) -> models.Filter:
        must = [models.FieldCondition(key="run_id", match=models.MatchValue(value=run_id))]
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        if paper_id:
            must.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)))
        return models.Filter(must=must)

    def hybrid_search(self, run_id: str, query: str, k: int = 8, source: str | None = None,
                      dense_vec: list[float] | None = None) -> list[Hit]:
        dense_vec = dense_vec or embed.embed_query_dense(query)
        sparse_vec = embed.embed_query_sparse(query)
        flt = self._filter(run_id, source)
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using=DENSE, limit=30, filter=flt),
                models.Prefetch(query=sparse_vec, using=SPARSE, limit=30, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt,
            limit=k,
            with_payload=True,
        )
        return [Hit(point_id=str(p.id), chunk_id=p.payload.get("chunk_id", ""), score=float(p.score),
                    payload=p.payload) for p in res.points]

    def get_chunk(self, run_id: str, chunk_id: str) -> Hit | None:
        try:
            recs = self.client.retrieve(self.collection, ids=[point_id(run_id, chunk_id)], with_payload=True)
        except Exception:
            return None
        if not recs:
            return None
        r = recs[0]
        return Hit(point_id=str(r.id), chunk_id=chunk_id, score=1.0, payload=r.payload)

    def count(self, run_id: str | None = None) -> int:
        flt = self._filter(run_id) if run_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def scroll_run(self, run_id: str, with_vectors: bool = True) -> list[models.Record]:
        out: list[models.Record] = []
        offset = None
        while True:
            recs, offset = self.client.scroll(self.collection, scroll_filter=self._filter(run_id), limit=256,
                                              offset=offset, with_payload=True,
                                              with_vectors=[DENSE] if with_vectors else False)
            out.extend(recs)
            if offset is None:
                break
        return out

    # -- projection ----------------------------------------------------------------

    @staticmethod
    def _dense_of(rec: models.Record) -> list[float]:
        v = rec.vector
        return v[DENSE] if isinstance(v, dict) else v

    def basis(self, run_id: str) -> PCABasis | None:
        if run_id not in self._bases:
            recs = self.scroll_run(run_id)
            if not recs:
                return None
            self._bases[run_id] = fit_pca(np.array([self._dense_of(r) for r in recs]))
        return self._bases[run_id]

    def vector_map(self, run_id: str) -> list[dict[str, Any]]:
        recs = self.scroll_run(run_id)
        if not recs:
            return []
        vecs = np.array([self._dense_of(r) for r in recs])
        basis = fit_pca(vecs)
        self._bases[run_id] = basis
        xy = basis.project(vecs)
        return [{"id": str(r.id), "x": round(float(p[0]), 5), "y": round(float(p[1]), 5),
                 "paper_id": r.payload.get("paper_id"), "source": r.payload.get("source"),
                 "title": r.payload.get("title"), "chunk_id": r.payload.get("chunk_id")}
                for r, p in zip(recs, xy)]
