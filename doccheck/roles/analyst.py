"""ANALYST: verifies one claim against a deterministically assembled context. No tools."""
from __future__ import annotations

from ..config import load_prompt
from ..context import RunCtx, fmt_context, jsonish
from ..models import AnalystRevision, AnalystVerdict, Claim, Critique
from ..rag.qdrant_store import Hit

RELATED_K = 8
TARGET_K = 4


async def gather_context(ctx: RunCtx, claim: Claim) -> list[Hit]:
    related = await ctx.search(claim.text, RELATED_K, "related", "analyst", claim.id)
    target = await ctx.search(claim.text, TARGET_K, "target", "analyst", claim.id)
    return target + related


def _claim_vars(claim: Claim) -> dict:
    return {"claim_id": claim.id, "claim_type": claim.type.value, "claim_section": claim.section,
            "claim_text": claim.text}


async def analyze(ctx: RunCtx, claim: Claim, hits: list[Hit]) -> AnalystVerdict:
    system, user = load_prompt("analyst", context=fmt_context(hits), **_claim_vars(claim))
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
        **_claim_vars(claim))
    return await ctx.llm.structured(role="analyst", model=ctx.settings.analyst_model, system=system, user=user,
                                    schema=AnalystRevision, tool_name="submit_revision",
                                    tool_description="Submit the revised verdict.",
                                    claim_id=claim.id, round=round_, fixture_key=f"analyst-{claim.id}-r{round_}")
