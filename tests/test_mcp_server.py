"""The MCP server exposes the Critic's tools; exercised through the MCP tool-call path, Qdrant in memory."""
import json

import pytest

from doccheck import mcp_server


@pytest.fixture
def server(mem_store):
    mcp_server._index(mem_store)
    yield mcp_server.server
    mcp_server._state.clear()


def _items(result) -> list:
    """MCP returns a list result as one TextContent per item, a dict as a single TextContent."""
    assert not result.is_error, result
    return [json.loads(c.text) for c in result.content]


def _one(result) -> dict:
    items = _items(result)
    assert len(items) == 1
    return items[0]


async def test_tools_are_listed(server):
    names = {t.name for t in await server.list_tools()}
    assert names == {"search_corpus", "get_chunk", "check_quote", "list_sources"}


async def test_search_get_and_check_quote(server):
    hits = _items(await server.call_tool("search_corpus", {"query": "пробный период для Казахстана", "k": 3}))
    assert any(h["section"] == "Пробный период (Казахстан)" for h in hits)
    cid = next(h["chunk_id"] for h in hits if h["section"] == "Пробный период (Казахстан)")
    chunk = _one(await server.call_tool("get_chunk", {"chunk_id": cid}))
    assert chunk["found"] and "7 дней" in chunk["text"]
    ok = _one(await server.call_tool("check_quote", {"chunk_id": cid, "quote": "пробный период тарифа «Бизнес» составляет 7 дней"}))
    bad = _one(await server.call_tool("check_quote", {"chunk_id": cid, "quote": "пробный период составляет 14 дней без ограничений"}))
    assert ok["ok"] is True and bad["ok"] is False
    assert _one(await server.call_tool("get_chunk", {"chunk_id": "nope.md#0"}))["found"] is False


async def test_list_sources(server):
    docs = _items(await server.call_tool("list_sources", {}))
    assert len(docs) == 11 and any("4. Возврат денежных средств" in d["sections"] for d in docs)
