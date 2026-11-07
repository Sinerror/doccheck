"""Per-run context handed to every role. Plain dataclass: no framework, no hidden state."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .events import EventBus, Role
from .llm import LLM
from .rag.qdrant_store import Hit, QdrantStore


@dataclass
class RunParams:
    max_claims: int = 10
    debate_rounds: int = 2
    critic_max_iters: int = 4
    claim_concurrency: int = 4
    use_critic: bool = True   # False = analyst only (used to compare analyst models cheaply)


@dataclass
class RunCtx:
    run_id: str
    settings: Settings
    bus: EventBus
    llm: LLM
    store: QdrantStore
    params: RunParams
    corpus_scope: str = ""
    title: str = ""
    doc_context: str = ""  # Scout's one-line context of the document (addressee, region, tariff)

    @property
    def scopes(self) -> list[str]:
        return [s for s in (self.corpus_scope, self.run_id) if s]

    async def search(self, query: str, k: int, source: str | None, role: Role,
                     claim_id: str | None = None) -> list[Hit]:
        """Hybrid retrieval + `rag.retrieval` event with the query's 2D projection."""
        from .rag import embed

        def _run() -> tuple[list[Hit], list[float] | None]:
            qv = embed.embed_query_dense(query, self.store.dense_model)
            hits = self.store.hybrid_search(self.scopes, query, k=k, source=source, dense_vec=qv)
            basis = self.store.basis(self.scopes)
            return hits, (basis.project_one(qv) if basis else None)

        hits, xy = await asyncio.to_thread(_run)
        self.bus.emit("rag.retrieval", {
            "query": query, "query_xy": xy, "point_ids": [h.point_id for h in hits],
            "scores": [round(h.score, 5) for h in hits], "by_role": role, "source": source,
        }, role=role, claim_id=claim_id)
        return hits

    async def chunk(self, chunk_id: str) -> Hit | None:
        return await asyncio.to_thread(self.store.get_chunk, self.scopes, chunk_id)

    async def chunk_text(self, chunk_id: str) -> str | None:
        hit = await self.chunk(chunk_id)
        return hit.text if hit else None


def fmt_hit(h: Hit, max_chars: int | None = None) -> str:
    p = h.payload
    head = f"[chunk_id={h.chunk_id}] {p.get('title') or p.get('doc_id')} | раздел: {p.get('section') or '-'}"
    text = h.text if max_chars is None else h.text[:max_chars]
    return f"{head}\n{text}"


def fmt_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(fmt_hit(h) for h in hits) or "(в источниках ничего не найдено)"


def jsonish(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=1)


__all__ = ["RunCtx", "RunParams", "fmt_hit", "fmt_context", "jsonish", "field"]
