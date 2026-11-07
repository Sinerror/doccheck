from doccheck.rag.chunking import Chunk
from doccheck.rag.qdrant_store import make_chunk_id, point_id

TARGET = {"paper_id": "arxiv:1706.03762", "title": "Attention Is All You Need", "year": 2017}
RELATED = {"paper_id": "arxiv:1705.03122", "title": "Convolutional Sequence to Sequence Learning", "year": 2017}


def _chunks(texts):
    return [Chunk(section="S", text=t, chunk_idx=i) for i, t in enumerate(texts)]


def _fill(store, run_id="run1"):
    store.upsert_chunks(run_id, TARGET, _chunks([
        "The Transformer is based solely on attention mechanisms, dispensing with recurrence entirely.",
        "Training took 3.5 days on eight P100 GPUs and reached 28.4 BLEU on English-German.",
    ]), "target")
    store.upsert_chunks(run_id, RELATED, _chunks([
        "ConvS2S uses convolutional neural networks for sequence to sequence learning with gated linear units.",
        "Our convolutional model outperforms recurrent baselines on WMT English-French translation.",
        "Position embeddings give the convolutional model a sense of order in the sequence.",
    ]), "related")


def test_upsert_is_idempotent_and_ids_deterministic(mem_store):
    _fill(mem_store)
    assert mem_store.count("run1") == 5
    _fill(mem_store)
    assert mem_store.count("run1") == 5
    assert point_id("run1", make_chunk_id("arxiv:1", 0)) == point_id("run1", "arxiv:1#0")
    assert point_id("run1", "a#0") != point_id("run2", "a#0")


def test_hybrid_search_filters_by_run_and_source(mem_store):
    _fill(mem_store)
    _fill(mem_store, "other-run")
    hits = mem_store.hybrid_search("run1", "how many GPUs and days of training", k=3)
    assert hits[0].chunk_id == "arxiv:1706.03762#1"
    assert all(h.payload["run_id"] == "run1" for h in hits)
    rel = mem_store.hybrid_search("run1", "convolutional sequence model", k=8, source="related")
    assert rel and all(h.payload["source"] == "related" for h in rel)
    assert len(rel) == 3


def test_get_chunk_and_vector_map(mem_store):
    _fill(mem_store)
    h = mem_store.get_chunk("run1", "arxiv:1705.03122#2")
    assert h is not None and "Position embeddings" in h.text
    assert mem_store.get_chunk("run1", "arxiv:0000.00000#0") is None
    vm = mem_store.vector_map("run1")
    assert len(vm) == 5
    assert {p["source"] for p in vm} == {"target", "related"}
    assert all(-1 <= p["x"] <= 1 and -1 <= p["y"] <= 1 for p in vm)
    basis = mem_store.basis("run1")
    assert basis is not None


def test_schema_compatibility(mem_store):
    assert mem_store.schema_compatible()
    from qdrant_client import models
    mem_store.client.create_collection("bad", vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT))
    from doccheck.rag.qdrant_store import QdrantStore
    assert not QdrantStore(mem_store.client, "bad").schema_compatible()


def test_delete_run_only_touches_that_run(mem_store):
    _fill(mem_store)
    _fill(mem_store, "keep")
    mem_store.delete_run("run1")
    assert mem_store.count("run1") == 0 and mem_store.count("keep") == 5
