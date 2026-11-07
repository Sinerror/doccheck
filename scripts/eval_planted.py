"""Planted-error eval: run DocCheck over data/eval/docs and score it against data/eval/labels.json.

    uv run python scripts/eval_planted.py --label full                       # full pipeline, ~$4-6
    uv run python scripts/eval_planted.py --label analyst-only --no-critic --reuse-claims <eval id>
    DOCCHECK_ANALYST_BASE_URL=http://127.0.0.1:1234/v1 DOCCHECK_ANALYST_MODEL=<local model> \
        uv run python scripts/eval_planted.py --label local-analyst --no-critic --reuse-claims <eval id>
    uv run python scripts/eval_planted.py --rescore                           # only re-render evals/results.md

`--reuse-claims` feeds the Scout's claims from an earlier eval run back in, so two analyst models are compared
on exactly the same claims (and the comparison costs nothing for the Scout).

Matching: every claim carries `quote`, the sentence of the document it came from. A claim is matched to the
labelled sentence with the best fuzzy score (≥ MATCH_THRESHOLD); unmatched claims are "extra".

Scoring (per planted error, over all claims matched to it):
  caught (strict)    final verdict ∈ expected, or the claim is disputed and the critic suggests a verdict ∈ expected
  flagged for human  the claim is in the report's needs_review list (the business outcome: a person looks at it)
  analyst r0         the analyst's first verdict ∈ expected (before any debate)
  critic-only        caught (strict) but not by analyst r0
False positives: claims matched to a correct sentence (or extra) whose final verdict is CONTRADICTED / OVERSTATED
(strict) or that are sent to a human (review).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapidfuzz import fuzz  # noqa: E402

EVALS = ROOT / "evals"
LABELS = ROOT / "data" / "eval" / "labels.json"
DOCS = ROOT / "data" / "eval" / "docs"
MATCH_THRESHOLD = 80
STRICT_FP = {"CONTRADICTED", "OVERSTATED"}
KIND_RU = {"number": "неверное число", "deadline": "неверный срок/дата", "condition": "изменённое условие",
           "object_swap": "свойство другого тарифа/региона", "outdated": "из архивной редакции",
           "overstatement": "преувеличение", "not_in_sources": "нет в источниках",
           "wrong_clause": "неверная ссылка на пункт"}


def norm(s: str) -> str:
    s = s.replace("ё", "е").replace("Ё", "Е").replace("«", '"').replace("»", '"').replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def match_score(claim: dict[str, Any], label_text: str) -> float:
    lt = norm(label_text)
    q = norm(claim.get("quote") or "")
    t = norm(claim.get("text") or "")
    scores = [fuzz.token_set_ratio(t, lt)]
    if q:
        scores += [fuzz.ratio(q, lt), fuzz.partial_ratio(lt, q) if len(q) >= len(lt) * 0.6 else 0]
    return max(scores)


def assign(claims: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, str | None]:
    """claim id → label id (or None)."""
    out: dict[str, str | None] = {}
    for c in claims:
        best, best_s = None, 0.0
        for lab in labels:
            s = match_score(c, lab["text"])
            if s > best_s:
                best, best_s = lab["id"], s
        out[c["id"]] = best if best_s >= MATCH_THRESHOLD else None
    return out


def _r0(c: dict[str, Any]) -> str:
    return c["history"][0]["analyst"]["verdict"]


def caught_strict(c: dict[str, Any], expected: list[str]) -> bool:
    if c["verdict"] in expected:
        return True
    return c.get("status") == "disputed" and c.get("critic_suggested") in expected


def score_doc(report: dict[str, Any], doc_labels: dict[str, Any]) -> dict[str, Any]:
    claims = report["claims"]
    review = set((report["aggregate"].get("needs_review") or {}).get("claims", {}))
    planted = doc_labels["planted"]
    clean = doc_labels["clean"]
    by_id = {lab["id"]: lab for lab in planted + clean}
    amap = assign(claims, planted + clean)
    rows = []
    for p in planted:
        cs = [c for c in claims if amap.get(c["id"]) == p["id"]]
        exp = p["expected"]
        rows.append({
            "id": p["id"], "kind": p["kind"], "text": p["text"], "expected": exp,
            "claims": [c["id"] for c in cs], "extracted": bool(cs),
            "r0": [_r0(c) for c in cs], "final": [c["verdict"] for c in cs],
            "status": [c["status"] for c in cs], "critic_suggested": [c.get("critic_suggested") for c in cs],
            "caught": any(caught_strict(c, exp) for c in cs),
            "caught_r0": any(_r0(c) in exp for c in cs),
            "flagged": any(c["id"] in review for c in cs),
        })
    for r in rows:
        r["critic_only"] = r["caught"] and not r["caught_r0"]
    others = []
    for c in claims:
        lab = amap.get(c["id"])
        if lab is not None and lab in {p["id"] for p in planted}:
            continue
        others.append({"claim": c["id"], "label": lab, "text": c["text"], "kind": "clean" if lab else "extra",
                       "r0": _r0(c), "final": c["verdict"], "status": c["status"],
                       "fp_strict": c["verdict"] in STRICT_FP, "fp_review": c["id"] in review})
    clean_extracted = {o["label"] for o in others if o["label"]}
    return {"doc": doc_labels["file"], "run_id": report["run_id"], "rows": rows, "others": others,
            "clean_total": len(clean), "clean_extracted": len(clean_extracted),
            "grounding": report["aggregate"]["grounding"], "metrics": report["metrics"],
            "claims": len(claims), "claim_errors": len(report.get("claim_errors") or []),
            "labels_seen": sorted(by_id)}


def summarize(docs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for d in docs for r in d["rows"]]
    others = [o for d in docs for o in d["others"]]
    clean = [o for o in others if o["kind"] == "clean"]
    n = len(rows) or 1

    def share(xs, key):
        return round(sum(1 for x in xs if x[key]) / len(xs), 3) if xs else None

    kinds: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], {"n": 0, "caught": 0, "flagged": 0, "caught_r0": 0, "extracted": 0})
        k["n"] += 1
        for f in ("caught", "flagged", "caught_r0", "extracted"):
            k[f] += int(r[f])
    g_ok = sum(d["grounding"]["ok"] for d in docs)
    g_n = sum(d["grounding"]["checks"] for d in docs)
    cost = sum(d["metrics"]["cost_usd"] for d in docs)
    return {
        "docs": len(docs), "planted": len(rows), "claims": sum(d["claims"] for d in docs),
        "extracted": round(sum(r["extracted"] for r in rows) / n, 3),
        "recall_strict": round(sum(r["caught"] for r in rows) / n, 3),
        "recall_flagged": round(sum(r["flagged"] for r in rows) / n, 3),
        "recall_r0": round(sum(r["caught_r0"] for r in rows) / n, 3),
        "critic_only": sum(r["critic_only"] for r in rows),
        "clean_claims": len(clean), "fp_strict_clean": share(clean, "fp_strict"), "fp_review_clean": share(clean, "fp_review"),
        "extra_claims": len(others) - len(clean),
        "fp_strict_all": share(others, "fp_strict"), "fp_review_all": share(others, "fp_review"),
        "grounding_rate": round(g_ok / g_n, 4) if g_n else None, "grounding": f"{g_ok}/{g_n}",
        "cost_usd": round(cost, 4), "cost_per_doc": round(cost / max(len(docs), 1), 4),
        "elapsed_s": round(sum(d["metrics"]["elapsed_s"] for d in docs), 1),
        "llm_calls": sum(d["metrics"]["llm_calls"] for d in docs),
        "claim_errors": sum(d["claim_errors"] for d in docs),
        "by_kind": kinds,
    }


def prompts_hash() -> str:
    h = hashlib.sha1()
    for p in sorted((ROOT / "doccheck" / "prompts").glob("*.md")):
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


# ---- running ----------------------------------------------------------------------------------------------

async def run_eval(args: argparse.Namespace) -> Path:
    from qdrant_client import QdrantClient

    from doccheck.config import get_settings
    from doccheck.context import RunParams
    from doccheck.events import EventBus
    from doccheck.models import Claim
    from doccheck.orchestrator import run_pipeline
    from doccheck.rag.qdrant_store import QdrantStore
    from doccheck.sources.local import load_file

    s = get_settings()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    docs = [d for d in labels["docs"] if not args.docs or any(d["file"].startswith(x) for x in args.docs)]
    eval_id = f"eval-{time.strftime('%Y%m%d-%H%M%S')}-{args.label}"
    out_dir = EVALS / "runs" / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)
    store = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60), s.collection)
    store.ensure_collection()
    params = RunParams(max_claims=args.max_claims, debate_rounds=args.debate_rounds, use_critic=not args.no_critic)
    sem = asyncio.Semaphore(args.concurrency)
    scored: list[dict[str, Any]] = []
    spent = 0.0

    async def one(d: dict[str, Any]) -> None:
        nonlocal spent
        async with sem:
            if args.budget and spent >= args.budget:
                print(f"  skip {d['file']}: budget ${args.budget} reached", flush=True)
                return
            run_id = f"{eval_id}-{d['file'][:3]}"
            claims_override = None
            if args.reuse_claims:
                src = EVALS / "runs" / args.reuse_claims / "claims" / f"{d['file']}.json"
                claims_override = [Claim.model_validate(c) for c in json.loads(src.read_text(encoding="utf-8"))]
            bus = EventBus(run_id, s.runs_dir / run_id / "events.jsonl")
            report = await run_pipeline(run_id=run_id, settings=s, store=store, bus=bus,
                                        doc=load_file(DOCS / d["file"]), filename=d["file"], params=params,
                                        write_summary=False, claims_override=claims_override)
            if report is None:
                err = (s.runs_dir / run_id / "error.txt")
                print(f"  FAILED {d['file']}: {err.read_text(encoding='utf-8')[-400:] if err.exists() else ''}",
                      flush=True)
                if "401" in str(err.read_text(encoding="utf-8") if err.exists() else "") or \
                        "402" in str(err.read_text(encoding="utf-8") if err.exists() else ""):
                    raise SystemExit("LLM endpoint refused the key (401/402): stopping the eval")
                return
            (out_dir / "claims").mkdir(exist_ok=True)
            (out_dir / "claims" / f"{d['file']}.json").write_text(json.dumps(
                [{k: c[k] for k in ("id", "text", "type", "quote", "section")} for c in report["claims"]],
                ensure_ascii=False, indent=1), encoding="utf-8")
            sd = score_doc(report, d)
            scored.append(sd)
            spent += report["metrics"]["cost_usd"]
            caught = sum(r["caught"] for r in sd["rows"])
            print(f"  {d['file']}: {caught}/{len(sd['rows'])} caught · {len(report['claims'])} claims · "
                  f"${report['metrics']['cost_usd']:.3f} · total ${spent:.2f}", flush=True)

    await asyncio.gather(*(one(d) for d in docs))
    scored.sort(key=lambda x: x["doc"])
    result = {"eval_id": eval_id, "label": args.label, "created_at": time.strftime("%Y-%m-%d %H:%M"),
              "prompts": prompts_hash(), "models": s.models(), "embed_model": store.dense_model,
              "params": vars(params), "reuse_claims": args.reuse_claims,
              "summary": summarize(scored), "docs": scored}
    path = EVALS / f"{eval_id}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ---- rendering -------------------------------------------------------------------------------------------

def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def mean_sd(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return pct(xs[0])
    return f"{statistics.mean(xs):.0%} ± {statistics.stdev(xs):.0%}"


def render(results: list[dict[str, Any]]) -> str:
    lab = json.loads(LABELS.read_text(encoding="utf-8"))
    L = ["# Planted-error eval", "",
         f"Dataset: `data/eval/` — {lab['totals']['docs']} synthetic business documents (support replies, "
         f"commercial proposals, an internal memo) with **{lab['totals']['planted']} planted errors** of 8 kinds and "
         f"{lab['totals']['clean']} correct statements, checked against the synthetic corpus in `data/corpus/` "
         "(tariffs RU/KZ, offer + archived offer, SLA, support, integrations, security, migration, features). "
         "Built by `scripts/build_eval_set.py`.", ""]
    if not results:
        return "\n".join(L + ["_No eval runs yet._", ""])
    by_label: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r)
    L += ["## Summary by configuration", "",
          "Mean ± sd over runs of the same configuration (stochastic models: each run is one sample).", "",
          "| configuration | runs | analyst | recall strict | flagged for human | analyst r0 | caught only after debate | "
          "FP strict (correct claims) | FP → human (correct claims) | grounding | $ / document |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, rs in by_label.items():
        S = [r["summary"] for r in rs]
        L.append(f"| {label} | {len(rs)} | `{rs[-1]['models']['analyst']}` | {mean_sd([x['recall_strict'] for x in S])} | "
                 f"{mean_sd([x['recall_flagged'] for x in S])} | {mean_sd([x['recall_r0'] for x in S])} | "
                 f"{statistics.mean(x['critic_only'] for x in S):.1f} of {S[-1]['planted']} | "
                 f"{mean_sd([x['fp_strict_clean'] for x in S])} | {mean_sd([x['fp_review_clean'] for x in S])} | "
                 f"{mean_sd([x['grounding_rate'] for x in S])} | ${statistics.mean(x['cost_per_doc'] for x in S):.3f} |")
    latest = max(results, key=lambda r: r["created_at"]) if results else None
    full = [r for r in results if r["params"].get("use_critic")]
    focus = max(full, key=lambda r: r["created_at"]) if full else latest
    s = focus["summary"]
    L += ["", f"## Latest full-pipeline run: `{focus['eval_id']}`", "",
          f"Models: " + ", ".join(f"{k} = `{v}`" for k, v in focus["models"].items()) +
          f" · embeddings `{focus['embed_model']}` · prompts `{focus['prompts']}` · "
          f"max_claims={focus['params']['max_claims']}, debate_rounds={focus['params']['debate_rounds']}", "",
          "| metric | value |", "|---|---|",
          f"| planted errors extracted as claims (Scout) | {pct(s['extracted'])} |",
          f"| **recall, strict** (right verdict, or critic's position on a disputed claim) | **{pct(s['recall_strict'])}** |",
          f"| **recall, flagged for human** (error lands in the review list) | **{pct(s['recall_flagged'])}** |",
          f"| recall, analyst alone (round 0) | {pct(s['recall_r0'])} |",
          f"| caught only after the analyst ⇄ critic debate | {s['critic_only']} of {s['planted']} |",
          f"| false positives on correct claims, strict (CONTRADICTED / OVERSTATED) | {pct(s['fp_strict_clean'])} of {s['clean_claims']} |",
          f"| correct claims sent to a human anyway | {pct(s['fp_review_clean'])} |",
          f"| extra claims (not labelled) | {s['extra_claims']} |",
          f"| grounding rate (quotes verified by code) | {pct(s['grounding_rate'])} ({s['grounding']}) |",
          f"| cost (router price estimate) | ${s['cost_usd']:.2f} total · ${s['cost_per_doc']:.3f} per document |",
          f"| LLM calls / wall time (sum over documents) | {s['llm_calls']} / {s['elapsed_s']} s |", "",
          "### Recall by error kind", "",
          "| kind | n | extracted | caught strict | flagged for human | analyst r0 |", "|---|---|---|---|---|---|"]
    for k, v in sorted(s["by_kind"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {k} ({KIND_RU.get(k, k)}) | {v['n']} | {v['extracted']} | {v['caught']} | {v['flagged']} | "
                 f"{v['caught_r0']} |")
    L += ["", "### Every planted error", "",
          "| id | kind | planted statement | analyst r0 | final | status | caught |", "|---|---|---|---|---|---|---|"]
    for d in focus["docs"]:
        for r in d["rows"]:
            mark = "✅" if r["caught"] else ("👁" if r["flagged"] else "—")
            L.append(f"| {r['id']} | {r['kind']} | {r['text'][:110].replace('|', '/')} | {', '.join(r['r0']) or 'not extracted'} | "
                     f"{', '.join(r['final'])} | {', '.join(r['status'])} | {mark} |")
    fps = [(d["doc"], o) for d in focus["docs"] for o in d["others"] if o["fp_strict"]]
    L += ["", "### False positives (strict) in that run", ""]
    if not fps:
        L.append("- none")
    for doc, o in fps:
        L.append(f"- `{doc}` {o['claim']} ({o['kind']}): «{o['text'][:140]}» → {o['final']} ({o['status']})")
    L += ["", "## History", "", "| eval | label | analyst | prompts | recall strict | flagged | r0 | FP strict | grounding | cost |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["created_at"]):
        x = r["summary"]
        L.append(f"| `{r['eval_id']}` | {r['label']} | `{r['models']['analyst']}` | `{r['prompts']}` | "
                 f"{pct(x['recall_strict'])} | {pct(x['recall_flagged'])} | {pct(x['recall_r0'])} | "
                 f"{pct(x['fp_strict_clean'])} | {pct(x['grounding_rate'])} | ${x['cost_usd']:.2f} |")
    L += ["", "## Definitions and caveats", "",
          "- *Strict* catch: the final verdict is one the label expects (CONTRADICTED/OVERSTATED for most kinds; "
          "UNVERIFIABLE also counts for statements absent from the sources and for references to clauses that do not exist), "
          "or the claim stayed disputed and the critic's suggested verdict is an expected one.",
          "- *Flagged for human*: the claim is in the report's review list (verdict ≠ SUPPORTED, disputed, confidence < 0.6, "
          "or a failed quote). This is what matters in production: the error does not leave the company unseen.",
          "- Claims are matched to labelled sentences by the `quote` the Scout copies from the document (fuzzy ≥ 80).",
          "- The dataset and the corpus are synthetic and were written by the author of the system: the numbers show how the "
          "pipeline behaves on these error types, not a benchmark of real-world accuracy.",
          "- Analyst-only rows reuse the Scout's claims of an earlier run (`--reuse-claims`), so analyst models are compared "
          "on identical claims.", ""]
    return "\n".join(L)


def load_results() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EVALS.glob("eval-*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="full")
    ap.add_argument("--docs", nargs="*", help="file prefixes, e.g. d01 d02")
    ap.add_argument("--max-claims", type=int, default=10)
    ap.add_argument("--debate-rounds", type=int, default=2)
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--reuse-claims", default=None, help="eval id whose Scout claims to reuse")
    ap.add_argument("--concurrency", type=int, default=2, help="documents in parallel")
    ap.add_argument("--budget", type=float, default=0.0, help="stop starting new documents after $X")
    ap.add_argument("--rescore", action="store_true", help="only re-render evals/results.md")
    args = ap.parse_args()
    if not args.rescore:
        path = asyncio.run(run_eval(args))
        print(json.dumps(json.loads(path.read_text(encoding="utf-8"))["summary"], ensure_ascii=False, indent=1))
    (EVALS / "results.md").write_text(render(load_results()), encoding="utf-8")
    print(f"wrote {EVALS / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
