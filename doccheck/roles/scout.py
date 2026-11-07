"""SCOUT: extract claims, search OpenAlex + arXiv, dedup, rank, fetch full texts."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from ..config import load_prompt
from ..context import RunCtx
from ..models import ClaimWithQueries, ScoutClaimsOutput, ScoutRankOutput
from ..sources import arxiv, openalex
from ..sources.pdf import ParsedPDF, abstract_like, parse_pdf

log = logging.getLogger(__name__)

MAX_CANDIDATES_FOR_RANKING = 60
ARXIV_QUERIES_PER_CLAIM = 1  # arXiv is rate-limited to 1 req / 3 s


def _section_text(parsed: ParsedPDF, pattern: str, limit: int) -> str:
    return parsed.section_text(pattern)[:limit]


async def extract_claims(ctx: RunCtx, parsed: ParsedPDF, title: str, abstract: str) -> list[ClaimWithQueries]:
    intro = _section_text(parsed, r"introduction", 5000)
    concl = _section_text(parsed, r"conclusion|discussion", 3000)
    if not intro and parsed.sections:
        intro = parsed.sections[0][1][:5000]
    system, user = load_prompt(
        "scout_claims", max_claims=ctx.params.max_claims, title=title,
        headings="\n".join(f"- {h}" for h in parsed.headings), abstract=abstract or "(not found)",
        intro=intro or "(not found)", conclusion=concl or "(not found)")
    out = await ctx.llm.structured(role="scout", model=ctx.settings.scout_model, system=system, user=user,
                                   schema=ScoutClaimsOutput, tool_name="submit_claims",
                                   tool_description="Submit the extracted claims and search queries.",
                                   fixture_key="scout-claims")
    claims = out.claims[: ctx.params.max_claims]
    for i, c in enumerate(claims, 1):  # normalise ids
        c.id = f"C{i}"
        c.search_queries = [q for q in c.search_queries if q.strip()][:3]
    ctx.bus.emit("scout.claims", {"claims": [c.model_dump(mode="json") for c in claims]}, role="scout")
    return claims


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def dedup_papers(papers: list[dict[str, Any]], exclude: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Dedup by DOI / arXiv id / normalised title. Merges fields from duplicates (e.g. abstract, pdf_url)."""
    seen: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    excl_keys = set()
    if exclude:
        excl_keys = {k for k in _keys(exclude)}
    for p in papers:
        keys = _keys(p)
        if not keys or keys & excl_keys:
            continue
        existing = next((seen[k] for k in keys if k in seen), None)
        if existing is None:
            q = dict(p)
            q["hits"] = 1
            out.append(q)
            for k in keys:
                seen[k] = q
        else:
            existing["hits"] = existing.get("hits", 1) + 1
            for f in ("abstract", "pdf_url", "doi", "arxiv_id", "venue", "year"):
                if not existing.get(f) and p.get(f):
                    existing[f] = p[f]
            if p.get("arxiv_id") and not str(existing["paper_id"]).startswith("arxiv:"):
                existing["paper_id"] = f"arxiv:{p['arxiv_id']}"
            for k in _keys(existing):
                seen[k] = existing
    return out


def _keys(p: dict[str, Any]) -> set[str]:
    keys = set()
    if p.get("doi"):
        keys.add("doi:" + str(p["doi"]).lower())
    if p.get("arxiv_id"):
        keys.add("arxiv:" + str(p["arxiv_id"]))
    t = norm_title(p.get("title", ""))
    if len(t) > 10:
        keys.add("title:" + t)
    return keys


async def search_literature(ctx: RunCtx, claims: list[ClaimWithQueries], target: dict[str, Any]) -> list[dict]:
    assert ctx.http is not None
    oa_sem = asyncio.Semaphore(3)
    results: list[dict[str, Any]] = []

    async def oa(query: str, claim_id: str) -> None:
        async with oa_sem:
            try:
                hits = await openalex.search(ctx.http, query, per_page=8, mailto=ctx.settings.openalex_mailto,
                                             api_key=ctx.settings.openalex_api_key)
            except (httpx.HTTPError, ValueError) as e:
                log.warning("openalex %r failed: %s", query, e)
                ctx.bus.emit("error", {"message": f"openalex search failed: {type(e).__name__}: {e}"[:500],
                                        "where": "scout.openalex"},
                             role="scout")
                hits = []
        for h in hits:
            h["found_for"] = claim_id
        results.extend(hits)
        ctx.bus.emit("scout.query", {"query": query, "source": "openalex", "claim_id": claim_id,
                                     "hits": len(hits)}, role="scout", claim_id=claim_id)

    async def ax(query: str, claim_id: str) -> None:
        try:
            hits = await arxiv.search(ctx.http, query, max_results=6)
        except (httpx.HTTPError, ValueError, Exception) as e:  # noqa: BLE001 - XML errors too
            log.warning("arxiv %r failed: %s", query, e)
            ctx.bus.emit("error", {"message": f"arxiv search failed: {type(e).__name__}: {e}"[:500],
                                    "where": "scout.arxiv"}, role="scout")
            hits = []
        for h in hits:
            h["found_for"] = claim_id
        results.extend(hits)
        ctx.bus.emit("scout.query", {"query": query, "source": "arxiv", "claim_id": claim_id,
                                     "hits": len(hits)}, role="scout", claim_id=claim_id)

    tasks = []
    for c in claims:
        for q in c.search_queries:
            tasks.append(oa(q, c.id))
        for q in c.search_queries[:ARXIV_QUERIES_PER_CLAIM]:
            tasks.append(ax(q, c.id))
    await asyncio.gather(*tasks)
    return dedup_papers(results, exclude=target)


async def rank_papers(ctx: RunCtx, claims: list[ClaimWithQueries], candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    # prefer candidates found by several queries / with abstracts before asking the LLM
    candidates = sorted(candidates, key=lambda p: (-(p.get("hits") or 1), -bool(p.get("abstract"))))
    candidates = candidates[:MAX_CANDIDATES_FOR_RANKING]
    keyed = {f"P{i + 1}": p for i, p in enumerate(candidates)}
    lines = [f"{k} | {p.get('year') or 'n.d.'} | {p.get('title')} | {(p.get('abstract') or '')[:350]}"
             for k, p in keyed.items()]
    system, user = load_prompt(
        "scout_rank", max_papers=ctx.params.max_papers, title=ctx.title,
        claims="\n".join(f"{c.id}: {c.text}" for c in claims), candidates="\n".join(lines))
    out = await ctx.llm.structured(role="scout", model=ctx.settings.scout_model, system=system, user=user,
                                   schema=ScoutRankOutput, tool_name="submit_selection",
                                   tool_description="Submit the selected candidate keys.",
                                   fixture_key="scout-rank")
    chosen: list[dict] = []
    for key in out.selected:
        key = key.strip().split()[0].strip(",.;:") if key.strip() else key
        p = keyed.get(key)
        if p is not None and p not in chosen:
            chosen.append(p)
        if len(chosen) >= ctx.params.max_papers:
            break
    n = max(len(chosen), 1)
    for i, p in enumerate(chosen):
        p["relevance"] = round(1 - i / n, 3)
        ctx.bus.emit("scout.paper_found", {"paper_id": p["paper_id"], "title": p.get("title") or "",
                                           "year": p.get("year"), "venue": p.get("venue"), "url": p.get("url"),
                                           "relevance": p["relevance"]}, role="scout")
    return chosen


async def fetch_fulltexts(ctx: RunCtx, papers: list[dict]) -> None:
    """Top-N papers with an open-access PDF get full text; the rest keep the abstract."""
    assert ctx.http is not None
    budget = ctx.params.fulltext_papers

    async def one(p: dict) -> None:
        try:
            if p.get("arxiv_id"):
                data = await arxiv.download_pdf(ctx.http, p["arxiv_id"])
            else:
                r = await ctx.http.get(p["pdf_url"], timeout=45, follow_redirects=True,
                                       headers=arxiv.HEADERS)
                r.raise_for_status()
                data = r.content
                if not data.startswith(b"%PDF"):
                    raise ValueError("not a PDF")
            parsed = await asyncio.to_thread(parse_pdf, data)
            if sum(len(t) for _, t in parsed.sections) < 2000:
                raise ValueError("PDF has too little text")
            p["parsed"] = parsed
            if not p.get("abstract"):
                p["abstract"] = abstract_like(parsed)
        except Exception as e:  # noqa: BLE001 - any failure → abstract only
            log.info("full text for %s unavailable: %s", p["paper_id"], e)
            p["fulltext_error"] = str(e)[:200]
            ctx.bus.emit("error", {"message": f"full text unavailable for {p['paper_id']}, using abstract: "
                                              f"{type(e).__name__}: {e}"[:500], "where": "scout.fulltext"},
                         role="scout")

    targets = [p for p in papers if p.get("pdf_url")][:budget]
    await asyncio.gather(*(one(p) for p in targets))
