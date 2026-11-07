"""arXiv API client. Global rate limit: at most one request every 3 seconds."""
from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

API = "https://export.arxiv.org/api/query"
PDF = "https://export.arxiv.org/pdf/{id}"
ATOM = "{http://www.w3.org/2005/Atom}"
MIN_INTERVAL_S = 3.0
HEADERS = {"User-Agent": "triad-reviewer/0.1 (research prototype)"}

_sem = asyncio.Semaphore(1)
_last_call = 0.0

_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)


def parse_arxiv_id(text: str) -> str | None:
    """Extract a bare arXiv id (without version) from an id or abs/pdf URL."""
    if not text:
        return None
    m = _ID_RE.search(text.strip())
    if not m:
        return None
    return m.group(1) or m.group(3)


async def _throttled_get(client: httpx.AsyncClient, url: str, **kw) -> httpx.Response:
    global _last_call
    async with _sem:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            resp = await client.get(url, headers=HEADERS, follow_redirects=True, **kw)
        finally:
            _last_call = time.monotonic()
    resp.raise_for_status()
    return resp


def _parse_feed(xml: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        abs_url = (e.findtext(f"{ATOM}id") or "").strip()
        aid = parse_arxiv_id(abs_url)
        if not aid:
            continue
        title = re.sub(r"\s+", " ", e.findtext(f"{ATOM}title") or "").strip()
        summary = re.sub(r"\s+", " ", e.findtext(f"{ATOM}summary") or "").strip()
        published = e.findtext(f"{ATOM}published") or ""
        authors = [a.findtext(f"{ATOM}name") or "" for a in e.findall(f"{ATOM}author")]
        out.append({
            "paper_id": f"arxiv:{aid}", "arxiv_id": aid, "title": title, "abstract": summary,
            "year": int(published[:4]) if published[:4].isdigit() else None, "venue": "arXiv",
            "authors": authors, "url": f"https://arxiv.org/abs/{aid}", "pdf_url": PDF.format(id=aid),
            "doi": None,
        })
    return out


async def fetch_metadata(client: httpx.AsyncClient, arxiv_id: str) -> dict[str, Any] | None:
    resp = await _throttled_get(client, API, params={"id_list": arxiv_id, "max_results": 1}, timeout=30)
    items = _parse_feed(resp.text)
    return items[0] if items else None


async def search(client: httpx.AsyncClient, query: str, max_results: int = 8) -> list[dict[str, Any]]:
    terms = re.findall(r"[A-Za-z0-9\-]+", query)[:8]
    if not terms:
        return []
    q = " AND ".join(f"all:{t}" for t in terms[:4]) if len(terms) <= 4 else "all:" + " ".join(terms)
    resp = await _throttled_get(client, API, params={"search_query": q, "max_results": max_results,
                                                     "sortBy": "relevance"}, timeout=30)
    return _parse_feed(resp.text)


async def download_pdf(client: httpx.AsyncClient, arxiv_id: str) -> bytes:
    resp = await _throttled_get(client, PDF.format(id=arxiv_id), timeout=90)
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"arXiv {arxiv_id}: response is not a PDF")
    return resp.content
