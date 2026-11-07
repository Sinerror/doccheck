import json

import pydantic
import pytest

from doccheck.events import ALL_EVENT_TYPES, EventBus, read_events


async def test_bus_persists_and_fans_out(tmp_path):
    bus = EventBus("r1", tmp_path / "r1" / "events.jsonl")
    bus.emit("stage.started", {"stage": "ingest"})
    snapshot, q = bus.subscribe()
    assert [e.seq for e in snapshot] == [1]
    bus.emit("scout.query", {"query": "q", "source": "arxiv"}, role="scout", claim_id="C1")
    ev = await q.get()
    assert ev.seq == 2 and ev.role == "scout" and ev.claim_id == "C1" and ev.data["hits"] is None
    bus.add_usage("gpt-6.1-sol", 1000, 100, 0.003)
    m = (await q.get()).data
    assert m["tokens_in"] == 1000 and m["llm_calls"] == 1 and m["cost_usd"] == 0.003
    bus.close()
    assert await q.get() is None
    events = read_events(tmp_path / "r1" / "events.jsonl")
    assert [e.type for e in events] == ["stage.started", "scout.query", "metrics"]
    assert all(e.run_id == "r1" for e in events)


def test_payload_is_validated(tmp_path):
    bus = EventBus("r", None)
    with pytest.raises(pydantic.ValidationError):
        bus.emit("stage.started", {"stage": "not-a-stage"})
    with pytest.raises(KeyError):
        bus.emit("made.up", {})


def test_subscribe_after_close_gets_history_and_end():
    bus = EventBus("r", None)
    bus.emit("stage.started", {"stage": "scout"})
    bus.close()
    snap, q = bus.subscribe()
    assert len(snap) == 1 and q.get_nowait() is None


def test_contract_covers_spec_types():
    spec = {"run.started", "stage.started", "stage.finished", "scout.claims", "scout.query", "scout.paper_found",
            "rag.indexed", "rag.retrieval", "agent.started", "agent.partial", "agent.output", "tool.call",
            "tool.result", "grounding.check", "debate.round", "claim.final", "metrics", "run.finished", "error"}
    assert spec == set(ALL_EVENT_TYPES)


def test_gen_ts_types(tmp_path):
    import subprocess
    import sys
    out = tmp_path / "events.ts"
    subprocess.run([sys.executable, "scripts/gen_ts_types.py", "--out", str(out)], check=True)
    ts = out.read_text(encoding="utf-8")
    assert "export interface RagRetrieval" in ts and '"rag.retrieval": RagRetrieval' in ts
    assert "export type TriadEvent" in ts
    json.dumps(ts)  # plain text
