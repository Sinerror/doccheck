"""End-to-end pipeline with no network: Qdrant in memory, the real corpus, LLM calls replayed from
tests/fixtures/llm/pipeline/*.json (recorded from the live API).

Re-record:  DOCCHECK_RECORD=1 uv run pytest tests/test_pipeline_offline.py
"""
import json

import pytest

from tests.conftest import CORPUS, EVAL_DOCS, LLM_FIXTURES, RECORD
from doccheck.config import Settings
from doccheck.context import RunParams
from doccheck.events import ALL_EVENT_TYPES, EventBus, read_events
from doccheck.llm import LLM
from doccheck.orchestrator import run_pipeline
from doccheck.sources.local import load_file

FIX = LLM_FIXTURES / "pipeline"
DOC = EVAL_DOCS / "d05_otvet_api.md"


@pytest.mark.skipif(not RECORD and not (FIX / "scout-claims.json").exists(), reason="no fixture recorded")
async def test_full_pipeline_offline(tmp_path, mem_store):
    settings = Settings()
    settings.runs_dir = tmp_path
    bus = EventBus("offline", tmp_path / "offline" / "events.jsonl")
    llm = LLM(settings, bus=bus, mode="record" if RECORD else "replay", fixtures_dir=FIX)
    params = RunParams(max_claims=3, debate_rounds=1, claim_concurrency=1)
    report = await run_pipeline(run_id="offline", settings=settings, store=mem_store, bus=bus,
                                doc=load_file(DOC), filename=DOC.name, params=params, llm=llm, corpus_dir=CORPUS)
    err = tmp_path / "offline" / "error.txt"
    assert report is not None, err.read_text() if err.exists() else "no report"
    assert 1 <= len(report["claims"]) <= 3
    for c in report["claims"]:
        assert c["status"] in ("accepted", "revised", "disputed")
        assert c["history"][0]["critic"]["stance"] in ("AGREE", "CHALLENGE")
        assert c["quote"] and c["verdict_ru"]
    assert report["aggregate"]["grounding"]["checks"] > 0
    assert report["summary"] and report["business"]["claims_total"] == len(report["claims"])
    assert report["corpus"]["docs"] and len(report["corpus"]["docs"]) == 11
    events = read_events(tmp_path / "offline" / "events.jsonl")
    types = {e.type for e in events}
    missing = set(ALL_EVENT_TYPES) - types - {"error", "tool.call", "tool.result"}
    assert not missing, missing
    assert events[0].type == "run.started" and events[-1].type == "run.finished"
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert sum(e.type == "claim.final" for e in events) == len(report["claims"])
    saved = json.loads((tmp_path / "offline" / "report.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == "offline"
    assert (tmp_path / "offline" / "report.md").exists() and (tmp_path / "offline" / "vector_map.json").exists()
    # the document under check is not kept in the index after the run
    assert mem_store.count("offline") == 0
