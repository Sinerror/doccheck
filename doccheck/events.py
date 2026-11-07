"""Event contract backend ↔ frontend (SPEC §6).

Every event is `{seq, ts, run_id, type, role?, claim_id?, data}`, appended to
`runs/{run_id}/events.jsonl` and fanned out to live SSE subscribers.
`scripts/gen_ts_types.py` generates `web/src/types/events.ts` from this module.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["scout", "analyst", "critic", "system"]
StageName = Literal["ingest", "scout", "index", "analyze", "report"]


class Event(BaseModel):
    seq: int
    ts: float
    run_id: str
    type: str
    role: Role | None = None
    claim_id: str | None = None
    data: dict[str, Any]


# ---- per-type payloads -------------------------------------------------------

class RunStarted(BaseModel):
    target: dict[str, Any]
    params: dict[str, Any]
    models: dict[str, str]


class StageEvent(BaseModel):
    stage: StageName
    detail: dict[str, Any] = Field(default_factory=dict)


class ScoutClaims(BaseModel):
    claims: list[dict[str, Any]]


class ScoutQuery(BaseModel):
    query: str
    source: Literal["openalex", "arxiv"]
    claim_id: str | None = None
    hits: int | None = None


class ScoutPaperFound(BaseModel):
    paper_id: str
    title: str
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    relevance: float | None = None


class RagIndexed(BaseModel):
    paper_id: str
    chunks: int
    total_points: int
    source: Literal["target", "related"]
    title: str | None = None


class RagRetrieval(BaseModel):
    query: str
    query_xy: list[float] | None
    point_ids: list[str]
    scores: list[float]
    by_role: Role
    source: str | None = None


class AgentStarted(BaseModel):
    role: Role
    claim_id: str | None = None
    round: int = 0
    model: str


class AgentPartial(BaseModel):
    role: Role
    claim_id: str | None = None
    round: int = 0
    text: str
    kind: Literal["reasoning", "thinking"] = "reasoning"


class AgentOutput(BaseModel):
    role: Role
    claim_id: str | None = None
    round: int = 0
    output: dict[str, Any]


class ToolCall(BaseModel):
    role: Role
    name: str
    args: dict[str, Any]
    summary: str


class ToolResult(BaseModel):
    role: Role
    name: str
    args: dict[str, Any]
    summary: str


class GroundingCheck(BaseModel):
    claim_id: str
    chunk_id: str
    quote: str
    score: float
    ok: bool
    by_role: Role
    round: int = 0


class DebateRound(BaseModel):
    claim_id: str
    round: int
    stance: Literal["AGREE", "CHALLENGE"]
    max_severity: str | None = None


class ClaimFinal(BaseModel):
    claim_id: str
    status: Literal["accepted", "revised", "disputed"]
    verdict: str
    history: list[dict[str, Any]]


class Metrics(BaseModel):
    tokens_in: int
    tokens_out: int
    cost_usd: float
    llm_calls: int
    elapsed_s: float
    by_model: dict[str, dict[str, float]] = Field(default_factory=dict)


class RunFinished(BaseModel):
    summary: dict[str, Any]


class ErrorEvent(BaseModel):
    message: str
    where: str | None = None
    fatal: bool = False  # True only when the whole run failed


EVENT_DATA_MODELS: dict[str, type[BaseModel]] = {
    "run.started": RunStarted,
    "stage.started": StageEvent,
    "stage.finished": StageEvent,
    "scout.claims": ScoutClaims,
    "scout.query": ScoutQuery,
    "scout.paper_found": ScoutPaperFound,
    "rag.indexed": RagIndexed,
    "rag.retrieval": RagRetrieval,
    "agent.started": AgentStarted,
    "agent.partial": AgentPartial,
    "agent.output": AgentOutput,
    "tool.call": ToolCall,
    "tool.result": ToolResult,
    "grounding.check": GroundingCheck,
    "debate.round": DebateRound,
    "claim.final": ClaimFinal,
    "metrics": Metrics,
    "run.finished": RunFinished,
    "error": ErrorEvent,
}

ALL_EVENT_TYPES = tuple(EVENT_DATA_MODELS)


# ---- bus ---------------------------------------------------------------------

class EventBus:
    """Single writer for one run: assigns seq, persists to jsonl, fans out to subscribers."""

    def __init__(self, run_id: str, path: Path | None):
        self.run_id = run_id
        self.path = path
        self.seq = 0
        self.history: list[Event] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.closed = False
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._metrics = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "llm_calls": 0}
        self._by_model: dict[str, dict[str, float]] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def emit(self, type: str, data: dict[str, Any] | BaseModel, *, role: Role | None = None,
             claim_id: str | None = None) -> Event:
        model = EVENT_DATA_MODELS[type]
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        payload = model.model_validate(data).model_dump(mode="json")
        with self._lock:
            self.seq += 1
            ev = Event(seq=self.seq, ts=time.time(), run_id=self.run_id, type=type, role=role,
                       claim_id=claim_id, data=payload)
            self.history.append(ev)
            if self._fh:
                self._fh.write(ev.model_dump_json() + "\n")
                self._fh.flush()
        for q in list(self.subscribers):
            q.put_nowait(ev)
        return ev

    def add_usage(self, model: str, tokens_in: int, tokens_out: int, cost: float | None) -> None:
        m = self._metrics
        m["tokens_in"] += tokens_in
        m["tokens_out"] += tokens_out
        m["cost_usd"] += cost or 0.0
        m["llm_calls"] += 1
        bm = self._by_model.setdefault(model, {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "calls": 0})
        bm["tokens_in"] += tokens_in
        bm["tokens_out"] += tokens_out
        bm["cost_usd"] += cost or 0.0
        bm["calls"] += 1
        self.emit("metrics", self.metrics_snapshot())

    def metrics_snapshot(self) -> dict[str, Any]:
        return {**self._metrics, "cost_usd": round(self._metrics["cost_usd"], 6),
                "elapsed_s": round(time.time() - self.t0, 2), "by_model": self._by_model}

    def subscribe(self) -> tuple[list[Event], asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue()
        snapshot = list(self.history)
        if self.closed:
            q.put_nowait(None)
        else:
            self.subscribers.add(q)
        return snapshot, q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def close(self) -> None:
        self.closed = True
        for q in list(self.subscribers):
            q.put_nowait(None)
        if self._fh:
            self._fh.close()
            self._fh = None


def read_events(path: Path) -> list[Event]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Event.model_validate(json.loads(line)))
    return out
