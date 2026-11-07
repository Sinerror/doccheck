"""LLM layer: pure helpers + recorded-stream replay (tests/fixtures/llm/unit/*.json).

Re-record with live API:  DOCCHECK_RECORD=1 uv run pytest tests/test_llm.py
"""
import json

import pytest

from tests.conftest import LLM_FIXTURES, RECORD
from doccheck.config import Settings
from doccheck.events import EventBus
from doccheck.llm import LLM, extract_partial_string, inline_schema, tool_spec
from doccheck.models import AnalystVerdict, Critique

FIX = LLM_FIXTURES / "unit"


def test_extract_partial_string():
    assert extract_partial_string('{"reasoning": "The cla') == "The cla"
    assert extract_partial_string('{"reasoning":"a \\"quoted\\" word\\nnext", "verdict"') == 'a "quoted" word\nnext'
    assert extract_partial_string('{"verdict": "SUPPORTED", "reasoning": "x\\u00e9y') == "xéy"
    assert extract_partial_string('{"reasoning": "trailing \\') == "trailing "
    assert extract_partial_string('{"verdict": "SUPPORTED"') is None


def test_inline_schema_has_no_refs():
    for model in (AnalystVerdict, Critique):
        s = json.dumps(inline_schema(model))
        assert "$ref" not in s and "$defs" not in s
        assert list(inline_schema(model)["properties"])[0] == "reasoning"
    spec = tool_spec("submit_verdict", "d", AnalystVerdict)
    assert spec["function"]["parameters"]["properties"]["evidence"]["items"]["properties"]["stance"]["enum"] == [
        "supports", "contradicts", "context"]


def _llm(bus):
    return LLM(Settings(), bus=bus, mode="record" if RECORD else "replay", fixtures_dir=FIX)


@pytest.mark.skipif(not RECORD and not (FIX / "unit-structured.json").exists(), reason="no fixture recorded")
async def test_structured_call_streams_reasoning_and_validates():
    bus = EventBus("t", None)
    llm = _llm(bus)
    system = ("You are ANALYST. Verify the claim against the context. Every quote must be a verbatim substring of "
              "the cited chunk.")
    user = ('CLAIM C1: "The Transformer reaches 28.4 BLEU on WMT14 English-German."\n\n'
            "[chunk_id=t#1] source=target\nThe big transformer model outperforms the best previously reported models "
            "(including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.\n\n"
            "Return your verdict via the `submit_verdict` tool.")
    out = await llm.structured(role="analyst", model=llm.settings.analyst_model, system=system, user=user,
                               schema=AnalystVerdict, tool_name="submit_verdict", claim_id="C1",
                               fixture_key="unit-structured")
    assert out.verdict.value in ("SUPPORTED", "PARTIALLY_SUPPORTED")
    assert out.evidence and out.evidence[0].chunk_id == "t#1"
    types = [e.type for e in bus.history]
    assert types[0] == "agent.started" and types[-1] == "agent.output"
    assert "agent.partial" in types and "metrics" in types
    partials = [e.data["text"] for e in bus.history if e.type == "agent.partial" and e.data["kind"] == "reasoning"]
    assert partials and partials[-1] == out.reasoning  # streamed text converges to the final field
    m = [e for e in bus.history if e.type == "metrics"][-1].data
    assert m["llm_calls"] == 1 and m["tokens_in"] > 0 and m["tokens_out"] > 0


async def test_aborted_stream_is_retried_with_low_reasoning_effort(monkeypatch):
    import httpx
    import openai

    from doccheck.llm import CallResult

    bus = EventBus("t", None)
    llm = LLM(Settings(), bus=bus, mode="live")
    seen = []

    async def fake_one_call(**kw):
        seen.append(kw.get("extra"))
        if kw.get("extra") is None:
            raise openai.APIError("response aborted: upstream time limit", httpx.Request("POST", "http://x"), body=None)
        return CallResult(content="ok", usage={"prompt_tokens": 10, "completion_tokens": 2}, model="gpt-6.1-sol")

    monkeypatch.setattr(llm, "_one_call", fake_one_call)
    res = await llm.call(role="critic", model="gpt-6.1-sol", messages=[])
    assert res.content == "ok"
    assert seen == [None, {"reasoning_effort": "low"}]
    assert [e.type for e in bus.history] == ["error", "metrics"]
    assert bus.history[0].data["fatal"] is False


async def test_schema_failures_fall_back_to_another_model(monkeypatch):
    from doccheck.llm import CallResult, ToolCallOut

    bus = EventBus("t", None)
    llm = LLM(Settings(), bus=bus, mode="live")
    models = []
    good = AnalystVerdict(reasoning="r", verdict="SUPPORTED", confidence=0.9, evidence=[], concerns=[])

    async def fake_call(**kw):
        models.append(kw["model"])
        args = '{"reasoning": "r"}' if len(models) < 3 else good.model_dump_json()
        return CallResult(tool_calls=[ToolCallOut("c", "submit_verdict", args)], model=kw["model"])

    monkeypatch.setattr(llm, "call", fake_call)
    out = await llm.structured(role="scout", model="gpt-6-luna", system="s", user="u", schema=AnalystVerdict,
                               tool_name="submit_verdict")
    assert out.verdict.value == "SUPPORTED"
    assert models == ["gpt-6-luna", "gpt-6-luna", llm.settings.fallback_model]
    errs = [e for e in bus.history if e.type == "error"]
    assert len(errs) == 1 and not errs[0].data["fatal"]


def test_runaway_tool_arguments_are_not_echoed_back():
    from doccheck.roles.critic import MAX_ECHO_ARGS, _echo_args
    assert _echo_args('{"query": "x"}') == '{"query": "x"}'
    assert _echo_args(None) == "{}"
    big = _echo_args('{"reasoning": "' + "a" * 200_000)
    assert len(big) < 200 and json.loads(big)["note"].startswith("previous arguments were")
    assert MAX_ECHO_ARGS >= 4000



async def test_runaway_stream_is_cut_and_retried_with_low_effort(monkeypatch):
    """The router ignores max_tokens for some models; the client-side guard must stop a degenerate stream."""
    settings = Settings()
    settings.max_stream_chars = 1000
    bus = EventBus("t", None)
    llm = LLM(settings, bus=bus, mode="live")
    seen, closed = [], []

    async def fake_raw(model, messages, tools, tool_choice, fixture_key, extra=None):
        seen.append(extra)
        try:
            if extra is None:  # degenerate: endless content
                while True:
                    yield {"choices": [{"delta": {"content": "x" * 200}}]}
            yield {"choices": [{"delta": {"content": "ok"}}]}
            yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        finally:
            closed.append(extra)

    monkeypatch.setattr(llm, "_raw_stream", fake_raw)
    res = await llm.call(role="critic", model="gpt-6.1-sol", messages=[{"role": "user", "content": "hi"}])
    assert res.content == "ok"
    assert seen == [None, {"reasoning_effort": "low"}]
    assert closed == [None, {"reasoning_effort": "low"}]  # the runaway stream was closed, not left running
    errs = [e for e in bus.history if e.type == "error"]
    assert len(errs) == 1 and "runaway" in errs[0].data["message"] and not errs[0].data["fatal"]
    m = [e for e in bus.history if e.type == "metrics"]
    assert len(m) == 2 and m[0].data["tokens_out"] > 0  # aborted call is accounted (estimated), not hidden
