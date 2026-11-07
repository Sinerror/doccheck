"""SCOUT: reads the document under check and extracts the statements that can be checked against the sources."""
from __future__ import annotations

import re

from ..config import load_prompt
from ..context import RunCtx
from ..models import Claim, ScoutClaimsOutput
from ..sources.local import ParsedDoc

MAX_DOC_CHARS = 16000


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def dedup_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[str] = set()
    out = []
    for c in claims:
        key = _norm(c.text)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


async def extract_claims(ctx: RunCtx, doc: ParsedDoc, title: str) -> tuple[list[Claim], str]:
    text = doc.full_text
    truncated = len(text) > MAX_DOC_CHARS
    system, user = load_prompt("scout_claims", max_claims=ctx.params.max_claims, title=title,
                               document=text[:MAX_DOC_CHARS] + ("\n[...документ обрезан...]" if truncated else ""))
    out = await ctx.llm.structured(role="scout", model=ctx.settings.scout_model, system=system, user=user,
                                   schema=ScoutClaimsOutput, tool_name="submit_claims",
                                   tool_description="Submit the extracted checkable claims.",
                                   fixture_key="scout-claims")
    claims = dedup_claims(out.claims)[: ctx.params.max_claims]
    for i, c in enumerate(claims, 1):  # normalise ids
        c.id = f"C{i}"
    ctx.doc_context = out.document_context.strip()
    ctx.bus.emit("scout.claims", {"claims": [c.model_dump(mode="json") for c in claims],
                                  "document_context": ctx.doc_context}, role="scout")
    return claims, ctx.doc_context
