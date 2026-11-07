"""FastAPI backend: start a check, live SSE, replay SSE, report, examples, corpus, health.
If `web/dist` exists (docker image), the UI is served from the same origin."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from sse_starlette.sse import EventSourceResponse

from .config import ROOT, Settings, get_settings
from .context import RunParams
from .events import Event, EventBus, read_events
from .orchestrator import run_pipeline
from .rag.qdrant_store import QdrantStore
from .sources.local import SUPPORTED, load_corpus, load_file, parse_bytes, parse_text

log = logging.getLogger("doccheck.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="DocCheck", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

ACTIVE: dict[str, EventBus] = {}
_TASKS: set[asyncio.Task] = set()
_STATE: dict[str, Any] = {}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
MAX_UPLOAD = 5 * 1024 * 1024
MAX_TEXT = 40_000


def settings() -> Settings:
    if "settings" not in _STATE:
        _STATE["settings"] = get_settings()
    return _STATE["settings"]


def store() -> QdrantStore:
    if "store" not in _STATE:
        s = settings()
        st = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60), s.collection)
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


def _example_path(name: str) -> Path:
    d = settings().examples_dir
    p = (d / name).resolve()
    if p.parent != d.resolve() or not p.is_file():
        raise HTTPException(404, "example not found")
    return p


@app.post("/api/runs")
async def create_run(request: Request) -> dict[str, str]:
    """JSON `{text, title?}` or `{example: "<file name>"}`, or multipart with `file` (md/txt/html/docx/pdf).
    Options: max_claims, debate_rounds, use_critic, callback_url."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        body: dict[str, Any] = {k: v for k, v in form.items() if isinstance(v, str)}
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(422, "multipart request needs a `file`")
        data = await upload.read()
        filename = upload.filename or "document.txt"
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file is larger than 5 MB")
        if Path(filename).suffix.lower() not in SUPPORTED:
            raise HTTPException(422, f"unsupported file type; use one of {', '.join(SUPPORTED)}")
        try:
            doc = await asyncio.to_thread(parse_bytes, data, filename)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"could not read the file: {e}")
    else:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(422, "expected JSON body or multipart form")
        if body.get("example"):
            p = _example_path(str(body["example"]))
            doc, filename = load_file(p), p.name
        elif body.get("text"):
            text = str(body["text"])[:MAX_TEXT]
            filename = (str(body.get("title") or "Текст").strip()[:80] or "Текст") + ".md"
            doc = parse_text(text, Path(filename).stem)
        else:
            raise HTTPException(422, "provide `text`, `example` or a file")
    if not doc.sections:
        raise HTTPException(422, "the document is empty")
    s = settings()
    if not s.has_llm_key and not s.llm_base_url.startswith(("http://127.0.0.1", "http://localhost")):
        raise HTTPException(503, "DOCCHECK_LLM_API_KEY is not configured (Replay of recorded runs still works)")
    params = RunParams(max_claims=_int(body.get("max_claims"), 10, 1, 20),
                       debate_rounds=_int(body.get("debate_rounds"), 2, 0, 4),
                       use_critic=str(body.get("use_critic", "true")).lower() not in ("false", "0", "no"))
    run_id = new_run_id()
    bus = EventBus(run_id, run_dir(run_id) / "events.jsonl")
    ACTIVE[run_id] = bus
    task = asyncio.create_task(run_pipeline(run_id=run_id, settings=s, store=store(), bus=bus, doc=doc,
                                            filename=filename, params=params,
                                            callback_url=body.get("callback_url")))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {"run_id": run_id}


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
    """Replays a recorded run with its original timing ÷ speed. Needs no LLM key and no Qdrant."""
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
        agg = r["aggregate"]
        info.update(status="finished", grounding_rate=r.get("grounding_rate"), claims=len(r["claims"]),
                    needs_review=agg["needs_review"]["count"], verdict_counts=agg["verdict_counts"],
                    status_counts=agg["status_counts"], cost_usd=r.get("metrics", {}).get("cost_usd"))
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
                             headers={"Content-Disposition": f'attachment; filename="doccheck-{run_id}.md"'})


@app.get("/api/runs/{run_id}/vector-map")
async def vector_map(run_id: str):
    p = run_dir(run_id) / "vector_map.json"
    if p.exists():
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
    if not run_dir(run_id).exists():
        raise HTTPException(404, "run not found")
    return []


@app.get("/api/examples")
async def examples() -> list[dict[str, Any]]:
    d = settings().examples_dir
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in SUPPORTED:
            doc = load_file(p)
            out.append({"name": p.name, "title": doc.title or p.stem, "chars": doc.chars,
                        "preview": doc.full_text[:400]})
    return out


@app.get("/api/examples/{name}")
async def example(name: str) -> dict[str, Any]:
    p = _example_path(name)
    return {"name": p.name, "text": p.read_text(encoding="utf-8")}


@app.get("/api/corpus")
async def corpus() -> list[dict[str, Any]]:
    docs = await asyncio.to_thread(load_corpus, settings().corpus_dir)
    return [{"doc_id": d.doc_id, "title": d.title, "sections": d.parsed.headings, "chars": d.parsed.chars}
            for d in docs]


@app.get("/api/health")
async def health() -> dict[str, Any]:
    s = settings()
    out: dict[str, Any] = {"llm_key": s.has_llm_key, "llm_base_url": s.llm_base_url, "collection": s.collection,
                           "analyst_local": s.analyst_is_local, "models": s.models()}
    try:
        st = store()
        out["qdrant"] = True
        out["points"] = await asyncio.to_thread(st.count)
    except Exception as e:  # noqa: BLE001
        out["qdrant"] = False
        out["qdrant_error"] = str(e)[:300]
    out["ok"] = bool(out["qdrant"] and out["llm_key"])
    return out


_DIST = ROOT / "web" / "dist"
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith("api/"):  # unknown API paths must not fall through to index.html
            raise HTTPException(404, "not found")
        f = (_DIST / path).resolve()
        if path and f.is_file() and _DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(_DIST / "index.html")
