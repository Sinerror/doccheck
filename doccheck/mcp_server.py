"""MCP server: the Critic's tools over the company corpus, for any MCP client (Claude Desktop, IDE agents, n8n).

    uv run python -m doccheck.mcp_server                 # stdio
    uv run python -m doccheck.mcp_server --http 8765     # streamable HTTP on 127.0.0.1:8765/mcp

Tools:
  search_corpus(query, k)        hybrid dense + BM25 search (same retrieval the pipeline uses)
  get_chunk(chunk_id)            full text of one chunk
  check_quote(chunk_id, quote)   the deterministic grounding check: is the quote verbatim in that chunk?
  list_sources()                 documents and sections of the corpus

The corpus folder (DOCCHECK_CORPUS_DIR) is indexed on first use and cached by content hash.
"""
from __future__ import annotations

import argparse
import threading
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import grounding
from .config import get_settings
from .rag.corpus import CorpusIndex, index_corpus
from .rag.qdrant_store import QdrantStore

server = MCPServer(
    name="doccheck",
    instructions="Search and quote the company's sources (tariffs, offer, SLA, policies). "
                 "Quote only text returned by get_chunk/search_corpus and verify it with check_quote.",
)

_state: dict[str, Any] = {}
_lock = threading.Lock()


def _index(store: QdrantStore | None = None) -> tuple[QdrantStore, CorpusIndex]:
    """Lazily connect and index; tests inject an in-memory store."""
    with _lock:
        if store is not None:
            _state.clear()
            _state["store"] = store
        if "store" not in _state:
            from qdrant_client import QdrantClient
            s = get_settings()
            st = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60), s.collection)
            st.ensure_collection()
            _state["store"] = st
        if "index" not in _state:
            _state["index"] = index_corpus(_state["store"], get_settings().corpus_dir)
        return _state["store"], _state["index"]


@server.tool(description="Hybrid (dense + BM25, RRF) search over the company's sources. Returns chunks with "
                         "chunk_id, document, section and text. Query in Russian works best.")
def search_corpus(query: str, k: int = 5) -> list[dict[str, Any]]:
    st, idx = _index()
    hits = st.hybrid_search([idx.scope], query, k=max(1, min(int(k), 10)), source="corpus")
    return [{"chunk_id": h.chunk_id, "document": h.payload.get("title"), "section": h.payload.get("section"),
             "score": round(h.score, 4), "text": h.text} for h in hits]


@server.tool(description="Full text of one chunk by its chunk_id (e.g. '04_oferta.md#5').")
def get_chunk(chunk_id: str) -> dict[str, Any]:
    st, idx = _index()
    hit = st.get_chunk([idx.scope], chunk_id)
    if hit is None:
        return {"chunk_id": chunk_id, "found": False}
    return {"chunk_id": chunk_id, "found": True, "document": hit.payload.get("title"),
            "section": hit.payload.get("section"), "text": hit.text}


@server.tool(description="Deterministic check that a quote is verbatim (fuzzy partial_ratio >= 90, typography "
                         "normalised) in the given chunk. Use it before citing anything.")
def check_quote(chunk_id: str, quote: str) -> dict[str, Any]:
    st, idx = _index()
    r = grounding.check_one(chunk_id, quote, lambda cid: (h.text if (h := st.get_chunk([idx.scope], cid)) else None))
    return r.as_dict()


@server.tool(description="List the documents of the corpus with their section headings.")
def list_sources() -> list[dict[str, Any]]:
    _, idx = _index()
    return [{"doc_id": d.doc_id, "title": d.title, "sections": d.parsed.headings} for d in idx.docs]


def main() -> None:
    ap = argparse.ArgumentParser(prog="doccheck-mcp")
    ap.add_argument("--http", type=int, default=None, help="serve streamable HTTP on this port instead of stdio")
    args = ap.parse_args()
    if args.http:
        server.run("streamable-http", host="127.0.0.1", port=args.http)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
