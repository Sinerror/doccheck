from tests.conftest import CORPUS
from doccheck.rag.chunking import Chunk
from doccheck.rag.corpus import index_corpus
from doccheck.rag.qdrant_store import QdrantStore, collection_for, embed_text, make_chunk_id, point_id

TARIFFS = {"doc_id": "tariffs.md", "title": "Тарифы"}
OFFER = {"doc_id": "offer.md", "title": "Оферта"}


def _chunks(section, texts):
    return [Chunk(section=section, text=t, chunk_idx=i) for i, t in enumerate(texts)]


def _fill(store, scope="corpus:1"):
    store.upsert_chunks(scope, TARIFFS, _chunks("Тариф «Бизнес»", [
        "Тариф «Бизнес» рассчитан на команды до 100 пользователей. Стоимость — 12 990 ₽ в месяц.",
        "Доступен REST API с лимитом 300 запросов в минуту.",
    ]), "corpus")
    store.upsert_chunks(scope, OFFER, _chunks("4. Возврат", [
        "4.1. Клиент вправе получить возврат полной суммы в течение 14 календарных дней.",
        "4.4. Возврат осуществляется в течение 10 рабочих дней.",
        "8.2. Споры рассматриваются в Арбитражном суде города Москвы.",
    ]), "corpus")


def test_upsert_is_idempotent_and_ids_deterministic(mem_store):
    _fill(mem_store)
    assert mem_store.count("corpus:1") == 5
    _fill(mem_store)
    assert mem_store.count("corpus:1") == 5
    assert point_id("s", make_chunk_id("a.md", 0)) == point_id("s", "a.md#0")
    assert point_id("s1", "a#0") != point_id("s2", "a#0")


def test_hybrid_search_filters_by_scope_and_source(mem_store):
    _fill(mem_store)
    _fill(mem_store, "corpus:other")
    mem_store.upsert_chunks("run1", {"doc_id": "check:x.md", "title": "Ответ"},
                            _chunks("Ответ", ["Возврат возможен в течение 30 дней."]), "document")
    hits = mem_store.hybrid_search(["corpus:1"], "сколько дней на возврат денег", k=3)
    assert hits[0].payload["doc_id"] == "offer.md"
    assert all(h.payload["scope"] == "corpus:1" for h in hits)
    corpus_only = mem_store.hybrid_search(["corpus:1", "run1"], "возврат 30 дней", k=8, source="corpus")
    assert corpus_only and all(h.payload["source"] == "corpus" for h in corpus_only)
    both = mem_store.hybrid_search(["corpus:1", "run1"], "возврат 30 дней", k=8)
    assert any(h.payload["source"] == "document" for h in both)
    for mode in ("dense", "sparse"):
        assert mem_store.hybrid_search(["corpus:1"], "лимит API запросов", k=2, mode=mode)


def test_get_chunk_and_vector_map(mem_store):
    _fill(mem_store)
    h = mem_store.get_chunk(["run-x", "corpus:1"], "offer.md#2")
    assert h is not None and "Арбитражном" in h.text
    assert mem_store.get_chunk(["corpus:1"], "nope.md#0") is None
    vm = mem_store.vector_map(["corpus:1"])
    assert len(vm) == 5 and {p["doc_id"] for p in vm} == {"tariffs.md", "offer.md"}
    assert all(-1 <= p["x"] <= 1 and -1 <= p["y"] <= 1 for p in vm)
    assert mem_store.basis(["corpus:1"]) is not None


def test_schema_compatibility_and_collection_names(mem_store):
    assert mem_store.schema_compatible()
    from qdrant_client import models
    mem_store.client.create_collection("doccheck_bad",
                                       vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT))
    assert not QdrantStore(mem_store.client, "doccheck_bad").schema_compatible()
    assert collection_for("doccheck", "intfloat/multilingual-e5-large") == "doccheck_multilingual_e5_large"


def test_delete_scope_only_touches_that_scope(mem_store):
    _fill(mem_store)
    _fill(mem_store, "keep")
    mem_store.delete_scope("corpus:1")
    assert mem_store.count("corpus:1") == 0 and mem_store.count("keep") == 5


def test_embed_text_carries_heading_path():
    assert embed_text("Тарифы", "Тариф «Бизнес»", "текст") == "Тарифы / Тариф «Бизнес»\nтекст"
    assert embed_text("Ответ", "Ответ", "текст") == "Ответ\nтекст"
    assert embed_text(None, None, "текст") == "текст"


def test_corpus_is_indexed_once_per_content_hash(mem_store):
    first = index_corpus(mem_store, CORPUS)
    assert not first.reused and first.chunks > 40 and len(first.docs) == 11
    second = index_corpus(mem_store, CORPUS)
    assert second.reused and second.scope == first.scope and second.chunks == first.chunks
    hits = mem_store.hybrid_search([first.scope], "пробный период для Казахстана", k=3)
    assert any(h.payload["section"] == "Пробный период (Казахстан)" for h in hits)
