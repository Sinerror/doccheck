"""End-to-end pipeline with no network: sources are stubbed, Qdrant is in-memory,
LLM calls are replayed from tests/fixtures/llm/pipeline/*.json (recorded from the live API).

Re-record:  DOCCHECK_RECORD=1 uv run pytest tests/test_pipeline_offline.py
"""
import json

import pytest

from tests.conftest import LLM_FIXTURES, RECORD, make_pdf
from doccheck.config import Settings
from doccheck.context import RunParams
from doccheck.events import ALL_EVENT_TYPES, EventBus, read_events
from doccheck.llm import LLM
from doccheck.orchestrator import run_pipeline
from doccheck.roles import scout as scout_mod
from doccheck.sources import arxiv, openalex

FIX = LLM_FIXTURES / "pipeline"

TARGET_SECTIONS = [
    ("Abstract", "We propose GatedMix, a simple sequence model that replaces recurrence with gated token mixing. "
                 "GatedMix reaches 27.1 BLEU on WMT 2014 English-German, outperforming all previous single models, "
                 "while training 3x faster than an LSTM baseline. To our knowledge, GatedMix is the first model to "
                 "use gated token mixing for translation."),
    ("1 Introduction", "Recurrent networks process tokens sequentially, which limits parallelism during training. "
                       "Convolutional sequence models reduce this cost but need many layers to relate distant "
                       "positions. We introduce gated token mixing, which relates all positions in a constant "
                       "number of operations. Our contributions are the GatedMix architecture and a study of its "
                       "speed and accuracy on machine translation."),
    ("2 Model", "GatedMix stacks six layers. Each layer mixes token representations with a learned gate and a "
                "position-wise feed-forward network. Residual connections and layer normalisation follow each "
                "sub-layer. The model has 61M parameters."),
    ("3 Results", "On WMT 2014 English-German, GatedMix obtains 27.1 BLEU, compared with 26.4 for ConvS2S and 24.6 "
                  "for GNMT. Training takes 2.1 days on 4 GPUs, about three times faster than our LSTM baseline "
                  "trained on the same hardware. We did not evaluate ensembles."),
    ("4 Conclusion", "GatedMix is the first model to use gated token mixing for translation. It outperforms all "
                     "previous single models on WMT 2014 English-German and trains 3x faster than an LSTM."),
]

RELATED = [
    {"paper_id": "openalex:W1", "openalex_id": "W1", "arxiv_id": None, "doi": "10.0/convs2s",
     "title": "Convolutional Sequence to Sequence Learning", "year": 2017, "venue": "ICML", "authors": ["J. Gehring"],
     "url": "https://example.org/convs2s", "pdf_url": None,
     "abstract": "We introduce an architecture based entirely on convolutional neural networks. ConvS2S reaches "
                 "26.4 BLEU on WMT 2014 English-German and outperforms the LSTM setup of Wu et al. (2016) on "
                 "WMT'14 English-French, while being an order of magnitude faster on GPU."},
    {"paper_id": "openalex:W2", "openalex_id": "W2", "arxiv_id": None, "doi": "10.0/gatedmixers",
     "title": "Gated Token Mixers for Language Modeling", "year": 2016, "venue": "ACL", "authors": ["A. Author"],
     "url": "https://example.org/gtm", "pdf_url": None,
     "abstract": "We propose gated token mixers, a layer that relates all positions of a sequence through a learned "
                 "gate. Applied to language modeling and to machine translation, gated token mixers match LSTM "
                 "quality while training twice as fast."},
    {"paper_id": "openalex:W3", "openalex_id": "W3", "arxiv_id": None, "doi": "10.0/transformer",
     "title": "A Strong Attention Baseline for Translation", "year": 2017, "venue": "NeurIPS", "authors": ["B. Author"],
     "url": "https://example.org/att", "pdf_url": None,
     "abstract": "Our attention-only model obtains 28.4 BLEU on WMT 2014 English-German as a single model, "
                 "establishing a new state of the art."},
]


@pytest.fixture
def stub_sources(monkeypatch):
    pdf = make_pdf("GatedMix: Gated Token Mixing for Translation", TARGET_SECTIONS)

    async def fetch_metadata(client, aid):
        return {"title": "GatedMix: Gated Token Mixing for Translation", "authors": ["T. Tester"], "year": 2017,
                "abstract": TARGET_SECTIONS[0][1]}

    async def download_pdf(client, aid):
        return pdf

    async def ax_search(client, q, max_results=8):
        return []

    async def oa_search(client, q, per_page=8, mailto="", api_key=""):
        return [dict(p) for p in RELATED]

    monkeypatch.setattr(arxiv, "fetch_metadata", fetch_metadata)
    monkeypatch.setattr(arxiv, "download_pdf", download_pdf)
    monkeypatch.setattr(arxiv, "search", ax_search)
    monkeypatch.setattr(openalex, "search", oa_search)
    monkeypatch.setattr(scout_mod, "ARXIV_QUERIES_PER_CLAIM", 0)


@pytest.mark.skipif(not RECORD and not (FIX / "scout-claims.json").exists(), reason="no fixture recorded")
async def test_full_pipeline_offline(tmp_path, mem_store, stub_sources):
    settings = Settings()
    settings.runs_dir = tmp_path
    bus = EventBus("offline", tmp_path / "offline" / "events.jsonl")
    llm = LLM(settings, bus=bus, mode="record" if RECORD else "replay", fixtures_dir=FIX)
    params = RunParams(max_claims=2, max_papers=3, debate_rounds=1, claim_concurrency=1)
    report = await run_pipeline(run_id="offline", settings=settings, store=mem_store, bus=bus, params=params,
                                arxiv_id="0000.00000", llm=llm)
    assert report is not None, (tmp_path / "offline" / "error.txt").read_text() if (
        tmp_path / "offline" / "error.txt").exists() else "no report"
    assert 1 <= len(report["claims"]) <= 2
    for c in report["claims"]:
        assert c["status"] in ("accepted", "revised", "disputed")
        assert c["history"][0]["critic"]["stance"] in ("AGREE", "CHALLENGE")
    assert report["aggregate"]["grounding"]["checks"] > 0
    assert report["review_score"] is not None and report["meta_review"]
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
