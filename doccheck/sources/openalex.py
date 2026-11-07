"""OpenAlex works search. Abstracts are rebuilt from `abstract_inverted_index`."""
from __future__ import annotations

import re
from typing import Any

import httpx

API = "https://api.openalex.org/works"
SELECT = ("id,doi,title,display_name,publication_year,ids,authorships,primary_location,"
          "best_oa_location,open_access,abstract_inverted_index,locations,cited_by_count")


def reconstruct_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = [(i, word) for word, idxs in inv.items() for i in idxs]
    positions.sort()
    return " ".join(w for _, w in positions)


_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


def _arxiv_id(work: dict[str, Any]) -> str | None:
    for loc in [work.get("primary_location"), work.get("best_oa_location"), *(work.get("locations") or [])]:
        if not loc:
            continue
        for key in ("landing_page_url", "pdf_url"):
            m = _ARXIV_IN_URL.search(loc.get(key) or "")
            if m:
                return m.group(1)
    doi = (work.get("doi") or "").lower()
    m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi)
    return m.group(1) if m else None


def normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    oa_id = (work.get("id") or "").rsplit("/", 1)[-1]
    arxiv_id = _arxiv_id(work)
    doi = work.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "").lower()
    loc = work.get("primary_location") or {}
    venue = ((loc.get("source") or {}) or {}).get("display_name")
    best = work.get("best_oa_location") or {}
    pdf_url = best.get("pdf_url") or (work.get("open_access") or {}).get("oa_url")
    if arxiv_id:
        pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    return {
        "paper_id": f"arxiv:{arxiv_id}" if arxiv_id else f"openalex:{oa_id}",
        "openalex_id": oa_id, "arxiv_id": arxiv_id, "doi": doi,
        "title": work.get("title") or work.get("display_name") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"), "venue": venue,
        "authors": [a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])][:12],
        "url": (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (loc.get("landing_page_url") or work.get("id"))),
        "pdf_url": pdf_url, "cited_by_count": work.get("cited_by_count"),
    }


async def search(client: httpx.AsyncClient, query: str, per_page: int = 8, mailto: str = "",
                 api_key: str = "") -> list[dict[str, Any]]:
    params: dict[str, Any] = {"search": query, "filter": "has_abstract:true", "per-page": per_page,
                              "select": SELECT}
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    resp = await client.get(API, params=params, timeout=30)
    resp.raise_for_status()
    return [normalize_work(w) for w in resp.json().get("results", [])]
