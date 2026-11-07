import json

import pytest
from fastapi.testclient import TestClient

from doccheck import api
from doccheck.config import Settings
from doccheck.events import EventBus


@pytest.fixture
def client(tmp_path, mem_store, monkeypatch):
    s = Settings()
    s.runs_dir = tmp_path
    monkeypatch.setitem(api._STATE, "settings", s)
    monkeypatch.setitem(api._STATE, "store", mem_store)
    bus = EventBus("rec", tmp_path / "rec" / "events.jsonl")
    bus.emit("run.started", {"target": {"title": "T"}, "params": {"max_claims": 1}, "models": {}})
    bus.emit("stage.started", {"stage": "ingest"})
    bus.emit("run.finished", {"summary": {"review_score": 7.0}})
    bus.close()
    (tmp_path / "rec" / "report.json").write_text(json.dumps({
        "review_score": 7.0, "grounding_rate": 1.0, "aggregate": {"status_counts": {"accepted": 1}},
        "metrics": {"cost_usd": 0.01}}), encoding="utf-8")
    (tmp_path / "rec" / "report.md").write_text("# report", encoding="utf-8")
    (tmp_path / "rec" / "vector_map.json").write_text(json.dumps([{"id": "p", "x": 0, "y": 0}]), encoding="utf-8")
    yield TestClient(api.app)


def _sse_data(text: str) -> list[dict]:
    return [json.loads(line[5:].strip()) for line in text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "{}"]


def test_health(client):
    h = client.get("/api/health").json()
    assert h["qdrant"] is True and h["collection"] and "points" in h


def test_list_and_report(client):
    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == "rec" and runs[0]["status"] == "finished" and runs[0]["title"] == "T"
    assert client.get("/api/runs/rec/report").json()["review_score"] == 7.0
    assert client.get("/api/runs/nope/report").status_code == 404
    md = client.get("/api/runs/rec/report.md")
    assert md.text == "# report" and md.headers["content-type"].startswith("text/markdown")
    assert client.get("/api/runs/..%2Fetc/report").status_code in (400, 404)


def test_replay_and_recorded_events(client):
    r = client.get("/api/runs/rec/replay?speed=1000")
    evs = _sse_data(r.text)
    assert [e["type"] for e in evs] == ["run.started", "stage.started", "run.finished"]
    assert "event: end" in r.text
    evs2 = _sse_data(client.get("/api/runs/rec/events").text)
    assert [e["seq"] for e in evs2] == [1, 2, 3]


def test_vector_map_falls_back_to_snapshot(client):
    assert client.get("/api/runs/rec/vector-map").json() == [{"id": "p", "x": 0, "y": 0}]


def test_create_run_validation(client):
    assert client.post("/api/runs", json={"max_claims": 3}).status_code == 422
    assert client.post("/api/runs", files={"file": ("x.pdf", b"not a pdf")}).status_code == 422
