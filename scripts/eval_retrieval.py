"""Retrieval eval: which embedding model finds the right corpus section for a business claim?

Two query sets, a query is a hit@k if any of the top-k chunks comes from a gold (file, section):
  * claims      — labelled sentences of the planted-error set (data/eval/labels.json) with a gold section:
                  correct claims and planted errors alike. They share wording with the corpus (easy for BM25).
  * paraphrased — colloquial questions written to avoid the corpus wording (data/eval/retrieval_paraphrased.json):
                  the realistic "how would a person ask" case, where the dense model has to do the work.

Runs fully offline (Qdrant :memory:, fastembed on CPU), costs nothing.

    uv run python scripts/eval_retrieval.py                      # all models, writes evals/retrieval.md
    uv run python scripts/eval_retrieval.py --models intfloat/multilingual-e5-large
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doccheck.rag.corpus import index_corpus  # noqa: E402
from doccheck.rag.qdrant_store import QdrantStore, collection_for  # noqa: E402

MODELS = [
    "BAAI/bge-small-en-v1.5",  # TRIAD's original English model: the baseline
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
]
MODES = ["dense", "sparse", "hybrid"]
KS = (1, 3, 5)


def load_paraphrased(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"id": q["id"], "text": q["text"], "kind": "paraphrased", "gold": {tuple(g) for g in q["gold"]}}
            for q in data["queries"]]


def load_queries(labels_path: Path) -> list[dict]:
    lab = json.loads(labels_path.read_text(encoding="utf-8"))
    out = []
    for d in lab["docs"]:
        for it in d["planted"] + d["clean"]:
            if it["gold"]:
                out.append({"id": it["id"], "text": it["text"], "kind": it.get("kind", "ok"),
                            "gold": {(g["file"], g["section"]) for g in it["gold"]}})
    return out


def evaluate(model: str, sets: dict[str, list[dict]], corpus: Path) -> dict:
    client = QdrantClient(":memory:")
    st = QdrantStore(client, collection_for("doccheck_eval", model), dense_model=model)
    st.ensure_collection()
    t0 = time.time()
    idx = index_corpus(st, corpus)
    index_s = time.time() - t0
    res: dict = {"model": model, "chunks": idx.chunks, "index_s": round(index_s, 1), "sets": {}}
    for set_name, queries in sets.items():
        res["sets"][set_name] = _eval_set(st, idx.scope, queries)
    return res


def _eval_set(st: QdrantStore, scope: str, queries: list[dict]) -> dict:
    out: dict = {}
    for mode in MODES:
        hits = {k: 0 for k in KS}
        rr = 0.0
        misses = []
        t1 = time.time()
        for q in queries:
            top = st.hybrid_search([scope], q["text"], k=max(KS), mode=mode)
            ranks = [i for i, h in enumerate(top) if (h.payload["doc_id"], h.payload["section"]) in q["gold"]]
            first = ranks[0] + 1 if ranks else None
            for k in KS:
                hits[k] += int(first is not None and first <= k)
            rr += 1 / first if first else 0.0
            if first is None or first > 3:
                misses.append({"id": q["id"], "rank": first,
                               "top1": f"{top[0].payload['doc_id']} / {top[0].payload['section']}" if top else None})
        n = len(queries)
        out[mode] = {**{f"recall@{k}": round(hits[k] / n, 3) for k in KS}, "mrr": round(rr / n, 3),
                     "ms_per_query": round(1000 * (time.time() - t1) / n, 1), "misses": misses}
    return out


SET_TITLES = {
    "claims": "Claims from the planted-error set (wording close to the corpus)",
    "paraphrased": "Paraphrased colloquial questions (wording avoids the corpus)",
}


def render(results: list[dict], sizes: dict[str, int]) -> str:
    L = ["# Retrieval eval: which embedding model for Russian business documents", "",
         "Corpus: `data/corpus/` (11 synthetic files, " + str(results[0]["chunks"]) + " chunks). "
         "A hit@k means a chunk from a gold (file, section) is in the top k. Qdrant in memory, fastembed on CPU. "
         "Reproduce: `uv run python scripts/eval_retrieval.py`.", ""]
    for set_name, n in sizes.items():
        L += [f"## {SET_TITLES.get(set_name, set_name)} — {n} queries", "",
              "| dense model | mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |", "|---|---|---|---|---|---|---|"]
        bm25_done = False
        for r in results:
            for mode, m in r["sets"][set_name].items():
                name = r["model"].split("/")[-1]
                if mode == "sparse":
                    if bm25_done:
                        continue
                    name, bm25_done = "— (BM25 only, Russian stemmer)", True
                L.append(f"| {name} | {mode} | {m['recall@1']:.0%} | {m['recall@3']:.0%} | {m['recall@5']:.0%} | "
                         f"{m['mrr']:.2f} | {m['ms_per_query']} |")
        L.append("")
    L += ["Index time per model (embedding the corpus on CPU): " +
          ", ".join(f"{r['model'].split('/')[-1]} {r['index_s']} s" for r in results), ""]
    key = "paraphrased" if "paraphrased" in sizes else next(iter(sizes))
    best = max(results, key=lambda r: (r["sets"][key]["hybrid"]["recall@5"], r["sets"][key]["hybrid"]["mrr"]))
    L += [f"Best hybrid result on the {key} set: **{best['model']}**.", "",
          f"## Misses of {best['model'].split('/')[-1]} (hybrid, {key}, rank > 3)", ""]
    misses = best["sets"][key]["hybrid"]["misses"]
    for m in misses:
        L.append(f"- `{m['id']}` rank {m['rank'] or '>5'}; top-1 was {m['top1']}")
    if not misses:
        L.append("- none")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--labels", default=str(ROOT / "data" / "eval" / "labels.json"))
    ap.add_argument("--paraphrased", default=str(ROOT / "data" / "eval" / "retrieval_paraphrased.json"))
    ap.add_argument("--out", default=str(ROOT / "evals" / "retrieval.md"))
    args = ap.parse_args()
    sets = {"claims": load_queries(Path(args.labels)),
            "paraphrased": load_paraphrased(Path(args.paraphrased))}
    results = []
    for m in args.models:
        print(f"== {m}", flush=True)
        r = evaluate(m, sets, Path(args.corpus))
        for set_name, modes in r["sets"].items():
            for mode, v in modes.items():
                print(f"   {set_name:11s} {mode:7s} r@1={v['recall@1']:.2f} r@3={v['recall@3']:.2f} "
                      f"r@5={v['recall@5']:.2f} mrr={v['mrr']:.2f}", flush=True)
        results.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(results, {k: len(v) for k, v in sets.items()}), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
