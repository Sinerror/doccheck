"""ANALYST: verifies one claim against a deterministically assembled context from the corpus. No tools.

The document under check is never evidence for itself: the context comes from the company corpus only.
"""
from __future__ import annotations

from ..config import load_prompt
from ..context import RunCtx, fmt_context, jsonish
from ..models import AnalystRevision, AnalystVerdict, Claim, Critique
from ..rag.qdrant_store import Hit

CONTEXT_K = 8


async def gather_context(ctx: RunCtx, claim: Claim) -> list[Hit]:
    return await ctx.search(claim.text, CONTEXT_K, "corpus", "analyst", claim.id)


def _claim_vars(claim: Claim) -> dict:
    return {"claim_id": claim.id, "claim_type": claim.type.value, "claim_text": claim.text,
            "claim_quote": claim.quote}


async def analyze(ctx: RunCtx, claim: Claim, hits: list[Hit]) -> AnalystVerdict:
    system, user = load_prompt("analyst", context=fmt_context(hits), document_title=ctx.title, **_claim_vars(claim))
    return await ctx.llm.structured(role="analyst", model=ctx.settings.analyst_model, system=system, user=user,
                                    schema=AnalystVerdict, tool_name="submit_verdict",
                                    tool_description="Submit the verdict for this claim.",
                                    claim_id=claim.id, round=0, fixture_key=f"analyst-{claim.id}-r0")


async def revise(ctx: RunCtx, claim: Claim, hits: list[Hit], previous: AnalystVerdict, prev_round: int,
                 grounding: list[dict], critique: Critique) -> AnalystRevision:
    round_ = prev_round + 1
    system, user = load_prompt(
        "analyst_revise", context=fmt_context(hits), previous=jsonish(previous.model_dump(mode="json")),
        prev_round=prev_round, grounding=jsonish(grounding), critique=jsonish(critique.model_dump(mode="json")),
        document_title=ctx.title, **_claim_vars(claim))
    return await ctx.llm.structured(role="analyst", model=ctx.settings.analyst_model, system=system, user=user,
                                    schema=AnalystRevision, tool_name="submit_revision",
                                    tool_description="Submit the revised verdict.",
                                    claim_id=claim.id, round=round_, fixture_key=f"analyst-{claim.id}-r{round_}")
