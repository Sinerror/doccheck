"""CRITIC: audits the analyst with tools (search_corpus, get_chunk) and submits a structured critique.

Tool loop with tool_choice="auto"; after `critic_max_iters` rounds the call is forced to `submit_critique`.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..config import load_prompt
from ..context import RunCtx, fmt_hit, jsonish
from ..llm import _json_from_text, tool_spec
from ..models import AnalystVerdict, Claim, Critique
from ..rag.qdrant_store import Hit

TOOL_HIT_CHARS = 1600

SEARCH_TOOL = tool_spec("search_corpus", "Hybrid (dense + BM25) search over the company's sources "
                        "(tariffs, offer, SLA, policies). Use it to look for evidence the analyst did not see, "
                        "especially evidence that contradicts the claim or limits its scope.", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Natural-language or keyword query, in Russian"},
        "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of chunks (<= 8)"},
    },
    "required": ["query"],
})
GET_CHUNK_TOOL = tool_spec("get_chunk", "Read the full text of one chunk by its chunk_id.", {
    "type": "object",
    "properties": {"chunk_id": {"type": "string"}},
    "required": ["chunk_id"],
})
SUBMIT_TOOL = tool_spec("submit_critique", "Submit your final critique of the analyst's verdict.", Critique)
FORCE_SUBMIT = {"type": "function", "function": {"name": "submit_critique"}}


MAX_ECHO_ARGS = 6000


def _echo_args(args: str | None) -> str:
    """Arguments echoed back into the history. A runaway/invalid blob must not be re-sent as context."""
    if not args:
        return "{}"
    if len(args) <= MAX_ECHO_ARGS:
        return args
    return json.dumps({"note": f"previous arguments were {len(args)} chars and invalid; omitted"})


def _summary_hits(hits: list[Hit]) -> str:
    return f"{len(hits)} hits: " + ", ".join(h.chunk_id for h in hits[:8])


async def _run_tool(ctx: RunCtx, name: str, args: dict[str, Any], claim: Claim,
                    surfaced: dict[str, Hit]) -> str:
    if name == "search_corpus":
        query = str(args.get("query") or claim.text)
        k = max(1, min(int(args.get("k") or 6), 8))
        clean = {"query": query, "k": k}
        ctx.bus.emit("tool.call", {"role": "critic", "name": name, "args": clean,
                                   "summary": f"search_corpus({query!r}, k={k})"},
                     role="critic", claim_id=claim.id)
        hits = await ctx.search(query, k, "corpus", "critic", claim.id)
        for h in hits:
            surfaced[h.chunk_id] = h
        ctx.bus.emit("tool.result", {"role": "critic", "name": name, "args": clean, "summary": _summary_hits(hits)},
                     role="critic", claim_id=claim.id)
        return "\n\n---\n\n".join(fmt_hit(h, TOOL_HIT_CHARS) for h in hits) or "(no results)"
    if name == "get_chunk":
        cid = str(args.get("chunk_id") or "")
        ctx.bus.emit("tool.call", {"role": "critic", "name": name, "args": {"chunk_id": cid},
                                   "summary": f"get_chunk({cid})"}, role="critic", claim_id=claim.id)
        hit = await ctx.chunk(cid)
        if hit:
            surfaced[hit.chunk_id] = hit
        ctx.bus.emit("tool.result", {"role": "critic", "name": name, "args": {"chunk_id": cid},
                                     "summary": f"{len(hit.text)} chars" if hit else "not found"},
                     role="critic", claim_id=claim.id)
        return fmt_hit(hit) if hit else f"chunk_id {cid!r} does not exist"
    return f"unknown tool {name!r}"


async def critique(ctx: RunCtx, claim: Claim, verdict: AnalystVerdict, grounding: list[dict],
                   hits: list[Hit], round_: int) -> tuple[Critique, dict[str, Hit]]:
    max_iters = ctx.params.critic_max_iters
    system, user = load_prompt(
        "critic", max_iters=max_iters, claim_id=claim.id, claim_type=claim.type.value,
        claim_text=claim.text, claim_quote=claim.quote, document_title=ctx.title, round=round_,
        document_context=ctx.doc_context or "(не указан)",
        verdict=jsonish(verdict.model_dump(mode="json")), grounding=jsonish(grounding),
        context_ids="\n".join(f"- {h.chunk_id} ({h.payload.get('title')} / {h.payload.get('section')})" for h in hits))
    model = ctx.settings.critic_model
    ctx.bus.emit("agent.started", {"role": "critic", "claim_id": claim.id, "round": round_, "model": model},
                 role="critic", claim_id=claim.id)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    tools = [SEARCH_TOOL, GET_CHUNK_TOOL, SUBMIT_TOOL]
    surfaced: dict[str, Hit] = {}
    last_err = ""
    for it in range(max_iters + 2):
        forced = it >= max_iters
        res = await ctx.llm.call(role="critic", model=model, messages=messages, tools=tools,
                                 tool_choice=FORCE_SUBMIT if forced else "auto", claim_id=claim.id, round=round_,
                                 stream_tool="submit_critique", fixture_key=f"critic-{claim.id}-r{round_}-i{it}")
        if not res.tool_calls:
            raw = _json_from_text(res.content)
            try:
                crit = Critique.model_validate_json(raw or "")
                return _done(ctx, claim, round_, crit), surfaced
            except (ValidationError, ValueError):
                messages.append({"role": "assistant", "content": res.content})
                messages.append({"role": "user", "content": "Use the tools. Finish by calling `submit_critique`."})
                continue
        messages.append({"role": "assistant", "content": res.content or None, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": _echo_args(tc.arguments)}}
            for tc in res.tool_calls]})
        submitted: Critique | None = None
        for tc in res.tool_calls:
            if tc.name == "submit_critique":
                try:
                    submitted = Critique.model_validate_json(tc.arguments or "")
                    content = "critique accepted"
                except (ValidationError, ValueError) as e:
                    last_err = str(e)[:1500]
                    content = f"INVALID submit_critique arguments, fix and resubmit:\n{last_err}"
            else:
                try:
                    args = json.loads(tc.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                content = await _run_tool(ctx, tc.name, args, claim, surfaced)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        if submitted is not None:
            return _done(ctx, claim, round_, submitted), surfaced
    raise RuntimeError(f"critic failed to submit a valid critique for {claim.id}: {last_err}")


def _done(ctx: RunCtx, claim: Claim, round_: int, crit: Critique) -> Critique:
    ctx.bus.emit("agent.output", {"role": "critic", "claim_id": claim.id, "round": round_,
                                  "output": crit.model_dump(mode="json")}, role="critic", claim_id=claim.id)
    return crit
