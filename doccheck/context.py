"""Per-run context handed to every role. Plain dataclass: no framework, no hidden state."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .events import EventBus, Role
from .llm import LLM
from .rag.qdrant_store import Hit, QdrantStore


@dataclass
class RunParams:
    max_claims: int = 8
    max_papers: int = 15
    debate_rounds: int = 2
    fulltext_papers: int = 5
    critic_max_iters: int = 4
    claim_concurrency: int = 4


@dataclass
class RunCtx:
    run_id: str
    settings: Settings
    bus: EventBus
    llm: LLM
    store: QdrantStore
    params: RunParams
    http: httpx.AsyncClient | None = None
    title: str = ""
    target_paper_id: str = ""

    async def search(self, query: str, k: int, source: str | None, role: Role,
                     claim_id: str | None = None) -> list[Hit]:
        """Hybrid retrieval + `rag.retrieval` event with the query's 2D projection."""
        from .rag import embed

        def _run() -> tuple[list[Hit], list[float] | None]:
            qv = embed.embed_query_dense(query)
            hits = self.store.hybrid_search(self.run_id, query, k=k, source=source, dense_vec=qv)
            basis = self.store.basis(self.run_id)
            return hits, (basis.project_one(qv) if basis else None)

        hits, xy = await asyncio.to_thread(_run)
        self.bus.emit("rag.retrieval", {
            "query": query, "query_xy": xy, "point_ids": [h.point_id for h in hits],
            "scores": [round(h.score, 5) for h in hits], "by_role": role, "source": source,
        }, role=role, claim_id=claim_id)
        return hits

    async def chunk_text(self, chunk_id: str) -> str | None:
        hit = await asyncio.to_thread(self.store.get_chunk, self.run_id, chunk_id)
        return hit.text if hit else None


def fmt_hit(h: Hit, max_chars: int | None = None) -> str:
    p = h.payload
    head = (f"[chunk_id={h.chunk_id}] source={p.get('source')} | {p.get('title') or ''}"
            f" ({p.get('year') or 'n.d.'}) | section: {p.get('section') or '-'}")
    text = h.text if max_chars is None else h.text[:max_chars]
    return f"{head}\n{text}"


def fmt_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(fmt_hit(h) for h in hits) or "(no evidence retrieved)"


def jsonish(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=1)


__all__ = ["RunCtx", "RunParams", "fmt_hit", "fmt_context", "jsonish", "field"]
