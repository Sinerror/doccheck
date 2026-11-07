"""FastAPI backend (SPEC §7): runs, live SSE, replay SSE, report, vector map, health."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from qdrant_client import QdrantClient
from sse_starlette.sse import EventSourceResponse

from .config import Settings, get_settings
from .context import RunParams
from .events import Event, EventBus, read_events
from .orchestrator import run_pipeline
from .rag.qdrant_store import QdrantStore
from .sources.arxiv import parse_arxiv_id

log = logging.getLogger("doccheck.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="TRIAD", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

ACTIVE: dict[str, EventBus] = {}
_TASKS: set[asyncio.Task] = set()
_STATE: dict[str, Any] = {}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def settings() -> Settings:
    if "settings" not in _STATE:
        _STATE["settings"] = get_settings()
    return _STATE["settings"]


def store() -> QdrantStore:
    if "store" not in _STATE:
        s = settings()
        st = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30), s.collection)
        st.ensure_collection()
        _STATE["store"] = st
    return _STATE["store"]


def run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "bad run id")
    return settings().runs_dir / run_id


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        x = int(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        raise HTTPException(422, f"expected integer, got {v!r}")
    return max(lo, min(hi, x))


@app.post("/api/runs")
async def create_run(request: Request) -> dict[str, str]:
    ctype = request.headers.get("content-type", "")
    pdf_bytes: bytes | None = None
    filename: str | None = None
    concept: str | None = None

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        body: dict[str, Any] = {k: v for k, v in form.items() if isinstance(v, str)}
        upload = form.get("file") or form.get("pdf")
        if upload is not None and not isinstance(upload, str):
            pdf_bytes = await upload.read()
            filename = upload.filename
            if not pdf_bytes.startswith(b"%PDF"):
                raise HTTPException(422, "uploaded file is not a PDF")
    else:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(422, "expected JSON body or multipart form")

    # Paper review mode (original)
    arxiv_id = parse_arxiv_id(str(body.get("arxiv_id") or "")) or None
    url = body.get("url")
    if not arxiv_id and url:
        arxiv_id = parse_arxiv_id(url) if "arxiv.org" in url else None
        if not arxiv_id and pdf_bytes is None:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(url)
            if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                raise HTTPException(422, "url is neither an arXiv link nor a downloadable PDF")
            pdf_bytes, filename = r.content, url.rsplit("/", 1)[-1]
    if not arxiv_id and pdf_bytes is None:
        raise HTTPException(422, "provide arxiv_id, url, concept, or a PDF file")
    s = settings()
    if not s.has_llm_key:
        raise HTTPException(503, "DOCCHECK_LLM_API_KEY is not configured")
    params = RunParams(max_claims=_int(body.get("max_claims"), 8, 1, 20),
                       max_papers=_int(body.get("max_papers"), 15, 0, 40),
                       debate_rounds=_int(body.get("debate_rounds"), 2, 0, 4))
    run_id = new_run_id()
    bus = EventBus(run_id, run_dir(run_id) / "events.jsonl")
    ACTIVE[run_id] = bus
    st = store()
    task = asyncio.create_task(run_pipeline(run_id=run_id, settings=s, store=st, bus=bus, params=params,
                                            arxiv_id=arxiv_id, pdf_bytes=pdf_bytes, filename=filename,
                                            callback_url=body.get("callback_url")))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {"run_id": run_id, "mode": "review"}


def _sse(ev: Event) -> dict[str, str]:
    return {"id": str(ev.seq), "data": ev.model_dump_json()}


@app.get("/api/runs/{run_id}/events")
async def live_events(run_id: str, request: Request):
    bus = ACTIVE.get(run_id)
    path = run_dir(run_id) / "events.jsonl"
    if bus is None and not path.exists():
        raise HTTPException(404, "run not found")

    async def gen():
        if bus is None:  # finished before this process started: dump the recording
            for ev in read_events(path):
                yield _sse(ev)
        else:
            snapshot, q = bus.subscribe()
            try:
                for ev in snapshot:
                    yield _sse(ev)
                while True:
                    ev = await q.get()
                    if ev is None:
                        break
                    if not snapshot or ev.seq > snapshot[-1].seq:
                        yield _sse(ev)
            finally:
                bus.unsubscribe(q)
        yield {"event": "end", "data": "{}"}

    return EventSourceResponse(gen(), ping=15)


@app.get("/api/runs/{run_id}/replay")
async def replay(run_id: str, speed: float = 4.0, max_gap: float = 5.0):
    path = run_dir(run_id) / "events.jsonl"
    if not path.exists():
        raise HTTPException(404, "run not found")
    events = read_events(path)
    speed = max(speed, 0.01)

    async def gen():
        prev = None
        for ev in events:
            if prev is not None:
                await asyncio.sleep(min(max(ev.ts - prev, 0) / speed, max_gap))
            prev = ev.ts
            yield _sse(ev)
        yield {"event": "end", "data": "{}"}

    return EventSourceResponse(gen(), ping=15)


def _run_info(d: Path) -> dict[str, Any] | None:
    ev_path = d / "events.jsonl"
    if not ev_path.exists():
        return None
    info: dict[str, Any] = {"run_id": d.name, "status": "unknown", "title": None, "created_at": None}
    with ev_path.open(encoding="utf-8") as fh:
        first = fh.readline()
    if first:
        ev = json.loads(first)
        info["created_at"] = ev["ts"]
        if ev["type"] == "run.started":
            info["title"] = ev["data"]["target"].get("title")
            info["params"] = ev["data"]["params"]
            info["target"] = ev["data"]["target"]
    rep = d / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text(encoding="utf-8"))
        info.update(status="finished", review_score=r.get("review_score"), grounding_rate=r.get("grounding_rate"),
                    status_counts=r["aggregate"]["status_counts"], cost_usd=r.get("metrics", {}).get("cost_usd"))
    elif (d / "error.txt").exists():
        info["status"] = "failed"
    elif d.name in ACTIVE and not ACTIVE[d.name].closed:
        info["status"] = "running"
    return info


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    root = settings().runs_dir
    if not root.exists():
        return []
    infos = [i for d in root.iterdir() if d.is_dir() and (i := _run_info(d))]
    return sorted(infos, key=lambda i: i.get("created_at") or 0, reverse=True)


@app.get("/api/runs/{run_id}/report")
async def get_report(run_id: str):
    p = run_dir(run_id) / "report.json"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/runs/{run_id}/report.md")
async def get_report_md(run_id: str):
    p = run_dir(run_id) / "report.md"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="triad-{run_id}.md"'})


@app.get("/api/runs/{run_id}/vector-map")
async def vector_map(run_id: str):
    d = run_dir(run_id)
    try:
        points = await asyncio.to_thread(store().vector_map, run_id)
    except Exception as e:  # noqa: BLE001 - Qdrant down → fall back to the snapshot
        log.warning("vector map from qdrant failed: %s", e)
        points = []
    if not points and (d / "vector_map.json").exists():
        points = json.loads((d / "vector_map.json").read_text(encoding="utf-8"))
    if not points and not d.exists():
        raise HTTPException(404, "run not found")
    return points


@app.get("/api/health")
async def health() -> dict[str, Any]:
    s = settings()
    out: dict[str, Any] = {"llm_key": s.has_llm_key, "llm_base_url": s.llm_base_url, "collection": s.collection,
                           "models": {"scout": s.scout_model, "analyst": s.analyst_model, "critic": s.critic_model}}
    try:
        st = store()
        out["qdrant"] = True
        out["points"] = await asyncio.to_thread(st.count)
    except Exception as e:  # noqa: BLE001
        out["qdrant"] = False
        out["qdrant_error"] = str(e)[:300]
    out["ok"] = bool(out["qdrant"] and out["llm_key"])
    return out
