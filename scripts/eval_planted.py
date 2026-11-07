"""Planted-error eval (SPEC §10).

Takes a real paper (default 1706.03762), plants three errors into its text, runs the full pipeline on the
altered text and measures, per planted error:
  * extracted   — did Scout turn the planted statement into a claim at all;
  * analyst     — did the analyst's first verdict (round 0) flag it;
  * critic-only — caught only because the critic challenged a verdict the analyst had let through;
  * final       — is it flagged in the final result (final verdict, or the critic's position on a disputed claim).
Plus false positives on the non-planted claims and grounding_rate. Writes evals/results.md (+ a JSON dump).

    uv run python scripts/eval_planted.py [--arxiv 1706.03762] [--max-claims 6] [--max-papers 10]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doccheck.config import get_settings  # noqa: E402
from doccheck.context import RunParams  # noqa: E402
from doccheck.events import EventBus  # noqa: E402
from doccheck.orchestrator import run_pipeline  # noqa: E402
from doccheck.rag.qdrant_store import QdrantStore  # noqa: E402
from doccheck.sources import arxiv  # noqa: E402
from doccheck.sources.pdf import ParsedPDF, parse_pdf  # noqa: E402

STRICT = {"CONTRADICTED", "OVERSTATED"}


def prompts_hash() -> str:
    h = hashlib.sha1()
    for f in sorted((ROOT / "doccheck" / "prompts").glob("*.md")):
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


@dataclass
class Plant:
    id: str
    kind: str
    statement: str
    edits: list[tuple[str, str, str]]  # (section prefix, original substring, replacement)
    markers: list[list[str]]           # a claim matches if it contains every marker of any one group
    fuzzy: bool = True                 # also accept partial_ratio >= 85 against the planted sentence


# Planted into the Abstract (which Scout reads) and into the body, so the error is consistent across the text.
PLANTS_1706 = [
    Plant(
        id="A",
        kind="distorted number in results",
        statement="Our model achieves 31.7 BLEU on the WMT 2014 English-to-German translation task, improving over "
                  "the existing best results, including ensembles, by over 5 BLEU.",
        edits=[
            ("Abstract", "Our model achieves 28.4 BLEU", "Our model achieves 31.7 BLEU"),
            ("Abstract", "including ensembles, by over 2 BLEU", "including ensembles, by over 5 BLEU"),
            ("6.1", "by more than 2 . 0 BLEU, establishing a new state-of-the-art BLEU score of 28 . 4",
             "by more than 5 . 0 BLEU, establishing a new state-of-the-art BLEU score of 31 . 7"),
        ],
        markers=[["31.7"], ["5 bleu"]],
        fuzzy=False,  # an untouched "28.4 BLEU" claim is textually near-identical; only the planted numbers count
    ),
    Plant(
        id="B",
        kind="false 'we are the first to…'",
        statement="We are the first to apply attention mechanisms to neural machine translation.",
        edits=[
            ("Abstract", "dispensing with recurrence and convolutions entirely.",
             "dispensing with recurrence and convolutions entirely. We are the first to apply attention "
             "mechanisms to neural machine translation."),
            ("7 Conclusion", "with multi-headed self-attention.",
             "with multi-headed self-attention. Ours is also the first work to apply attention mechanisms to "
             "neural machine translation."),
        ],
        markers=[["first", "apply attention"], ["first", "attention mechanisms to neural machine translation"]],
    ),
    Plant(
        id="C",
        kind="fabricated statement about another work",
        statement="The convolutional ConvS2S model of Gehring et al. fails to converge on WMT 2014 English-to-French "
                  "without pre-training.",
        edits=[
            ("Abstract", "a small fraction of the training costs of the best models from the literature.",
             "a small fraction of the training costs of the best models from the literature. By contrast, the "
             "convolutional ConvS2S model of Gehring et al. fails to converge on WMT 2014 English-to-French "
             "without pre-training."),
            ("6.1", "outperforming all of the previously published single models,",
             "outperforming all of the previously published single models (the ConvS2S model of Gehring et al. "
             "fails to converge on this task without pre-training),"),
        ],
        markers=[["convs2s"], ["gehring"], ["fail", "converge"]],
    ),
]


def norm(t: str) -> str:
    t = t.lower().replace("englishto", "english-to")
    t = re.sub(r"(\d) \. (\d)", r"\1.\2", t)
    return re.sub(r"\s+", " ", t)


def apply_plants(parsed: ParsedPDF, plants: list[Plant]) -> ParsedPDF:
    sections = list(parsed.sections)
    for plant in plants:
        for prefix, old, new in plant.edits:
            idx = next((i for i, (s, t) in enumerate(sections) if s.startswith(prefix) and old in t), None)
            if idx is None:
                raise SystemExit(f"plant {plant.id}: {old!r} not found in a section starting with {prefix!r}")
            name, text = sections[idx]
            sections[idx] = (name, text.replace(old, new, 1))
    return ParsedPDF(title=parsed.title, sections=sections, page_count=parsed.page_count)


def matches(plant: Plant, claim_text: str) -> bool:
    c = norm(claim_text)
    if any(all(m in c for m in group) for group in plant.markers):
        return True
    return plant.fuzzy and fuzz.partial_ratio(norm(plant.statement), c) >= 85


def critic_flagged(claim: dict) -> bool:
    """Critic challenged with a medium/high issue and pointed at a strict verdict, in any round."""
    for h in claim["history"]:
        cr = h["critic"]
        sev = {i["severity"] for i in cr["issues"]}
        if cr["stance"] == "CHALLENGE" and sev & {"medium", "high"} and cr.get("suggested_verdict") in STRICT:
            return True
    return False


def evaluate(report: dict, plants: list[Plant]) -> dict:
    claims = report["claims"]
    rows, planted_ids = [], set()
    for plant in plants:
        hit = [c for c in claims if matches(plant, c["text"])]
        planted_ids |= {c["id"] for c in hit}
        row = {"plant": plant.id, "kind": plant.kind, "statement": plant.statement,
               "claims": [c["id"] for c in hit], "extracted": bool(hit)}
        if hit:
            c = hit[0]
            r0 = c["history"][0]["analyst"]["verdict"]
            final = c["verdict"]
            final_flag = final in STRICT or (c["status"] == "disputed" and c.get("critic_suggested") in STRICT)
            analyst = r0 in STRICT
            row.update({
                "claim_text": c["text"], "analyst_r0": r0, "final_verdict": final, "status": c["status"],
                "critic_suggested": c.get("critic_suggested"),
                "rounds": [{"analyst": h["analyst"]["verdict"], "critic": h["critic"]["stance"],
                            "issues": [f"{i['severity']}:{i['type']}" for i in h["critic"]["issues"]]}
                           for h in c["history"]],
                "caught_by_analyst": analyst,
                "caught_by_critic_only": (not analyst) and (critic_flagged(c) or final_flag),
                "caught_final": final_flag,
                "caught_any": analyst or critic_flagged(c) or final_flag,
                "loose_flag": final != "SUPPORTED" or r0 != "SUPPORTED",
                # critic moved an analyst SUPPORTED to something weaker (or holds a weaker position on a dispute)
                "critic_loose": r0 == "SUPPORTED" and (final != "SUPPORTED" or (
                    c["status"] == "disputed" and c.get("critic_suggested") not in (None, "SUPPORTED"))),
            })
        else:
            row.update({"caught_by_analyst": False, "caught_by_critic_only": False, "caught_final": False,
                        "caught_any": False, "loose_flag": False, "critic_loose": False})
        rows.append(row)
    others = [c for c in claims if c["id"] not in planted_ids]
    n = len(plants)
    return {
        "rows": rows,
        "recall_extracted": sum(r["extracted"] for r in rows) / n,
        "recall_any": sum(r["caught_any"] for r in rows) / n,
        "recall_analyst": sum(r["caught_by_analyst"] for r in rows) / n,
        "recall_final": sum(r["caught_final"] for r in rows) / n,
        "critic_only": sum(r["caught_by_critic_only"] for r in rows),
        "critic_loose": sum(r["critic_loose"] for r in rows),
        "recall_loose": sum(r["loose_flag"] for r in rows) / n,
        "others": [{"id": c["id"], "text": c["text"], "verdict": c["verdict"], "status": c["status"],
                    "analyst_r0": c["history"][0]["analyst"]["verdict"]} for c in others],
        "fp_strict": sum(c["verdict"] in STRICT for c in others),
        "fp_loose": sum(c["verdict"] != "SUPPORTED" for c in others),
        "n_others": len(others),
        "grounding": report["aggregate"]["grounding"],
        "run_id": report.get("run_id"),
        "cost_usd": (report.get("metrics") or {}).get("cost_usd"),
        "models": report.get("models"),
    }


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_markdown(path: Path, ev: dict, report: dict, args, run_id: str) -> None:
    m = report["metrics"]
    L = [
        "# Planted-error eval",
        "",
        f"Paper: **{report['target']['title']}** (arXiv {args.arxiv}) with 3 planted errors · run `{run_id}` · "
        f"{time.strftime('%Y-%m-%d')}",
        "",
        f"Models: " + ", ".join(f"{k}=`{v}`" for k, v in report["models"].items())
        + " · params: " + ", ".join(f"{k}={report['params'][k]}" for k in ("max_claims", "max_papers", "debate_rounds")),
        f"Cost ≈ ${m['cost_usd']:.3f} (router price estimate) · {m['llm_calls']} LLM calls · {m['elapsed_s']:.0f} s",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| planted errors extracted as claims (Scout) | {pct(ev['recall_extracted'])} |",
        f"| **recall, any role** (analyst r0 or critic flagged CONTRADICTED/OVERSTATED) | **{pct(ev['recall_any'])}** |",
        f"| recall, analyst alone (round 0) | {pct(ev['recall_analyst'])} |",
        f"| caught only thanks to the critic (strict) | {ev['critic_only']} of {len(ev['rows'])} |",
        f"| critic weakened an analyst SUPPORTED on a planted error (loose) | {ev.get('critic_loose', 0)} of {len(ev['rows'])} |",
        f"| recall in the final report (final verdict, or critic position on a disputed claim) | {pct(ev['recall_final'])} |",
        f"| loose recall (any verdict ≠ SUPPORTED) | {pct(ev['recall_loose'])} |",
        f"| false positives on {ev['n_others']} non-planted claims, strict (CONTRADICTED/OVERSTATED) | {ev['fp_strict']} |",
        f"| false positives, loose (≠ SUPPORTED) | {ev['fp_loose']} |",
        f"| grounding_rate | {pct(ev['grounding']['rate'] or 0)} ({ev['grounding']['ok']}/{ev['grounding']['checks']}) |",
        "",
        "## Planted errors",
        "",
        "| # | kind | claim | analyst r0 | debate | final | status | caught by |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ev["rows"]:
        if not r["extracted"]:
            L.append(f"| {r['plant']} | {r['kind']} | — not extracted — | | | | | missed (Scout) |")
            continue
        debate = " → ".join(f"{x['analyst']}/{x['critic']}" for x in r["rounds"])
        who = ("analyst" if r["caught_by_analyst"] else "critic" if r["caught_by_critic_only"] else "—")
        L.append(f"| {r['plant']} | {r['kind']} | {', '.join(r['claims'])}: {r['claim_text'][:110]}… | "
                 f"{r['analyst_r0']} | {debate} | {r['final_verdict']} | {r['status']} | {who} |")
    L += ["", "Planted statements:", ""]
    for r in ev["rows"]:
        L.append(f"- **{r['plant']}** ({r['kind']}): “{r['statement']}”")
    L += ["", "## Non-planted claims (false-positive check)", "", "| claim | text | analyst r0 | final | status |",
          "|---|---|---|---|---|"]
    for c in ev["others"]:
        L.append(f"| {c['id']} | {c['text'][:120]}… | {c['analyst_r0']} | {c['verdict']} | {c['status']} |")
    L += ["", "## History of runs", "",
          "Each row is one eval run (`evals/<run>.json`). `prompts` is a hash of `triad/prompts/*.md` at scoring time.", "",
          "| run | prompts | extracted | recall strict | analyst r0 | critic-only | recall loose | FP strict | FP loose | grounding | cost |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted((ROOT / "evals").glob("eval-planted-*.json")):
        h = json.loads(f.read_text(encoding="utf-8"))
        L.append(f"| `{f.stem}` | `{h.get('prompts', '?')}` | {pct(h['recall_extracted'])} | {pct(h['recall_any'])} | "
                 f"{pct(h['recall_analyst'])} | {h['critic_only']} | {pct(h['recall_loose'])} | {h['fp_strict']}/{h['n_others']} | "
                 f"{h['fp_loose']}/{h['n_others']} | {pct(h['grounding']['rate'] or 0)} | "
                 f"{'$%.3f' % h['cost_usd'] if h.get('cost_usd') is not None else '?'} |")
    L += [
        "",
        "## Definitions and caveats",
        "",
        "- A claim is matched to a planted error by marker words (e.g. `31.7`, `ConvS2S`) or, for B and C, fuzzy "
        "match ≥ 85 against the planted sentence (A uses only the planted numbers).",
        "- *Strict* catch = CONTRADICTED or OVERSTATED. *Critic-only* = the analyst's round-0 verdict was not strict, "
        "but the critic challenged with a medium/high issue suggesting a strict verdict, or the final verdict became strict.",
        "- UNVERIFIABLE is the expected verdict for a fabricated statement about a paper that is not in the corpus "
        "(C, when ConvS2S itself was not retrieved); it counts toward loose recall only.",
        "- Run `eval-planted-20261019-135108` used the prompts before the evidence-standard fix (see DECISIONS.md: "
        "a restatement of the claim in the same paper is not evidence); later runs use the fixed prompts.",
        "- Non-planted claims are claims of the original paper; some are legitimately PARTIALLY_SUPPORTED (e.g. the "
        "known 41.0 vs 41.8 BLEU inconsistency in v7), so the loose FP count is an upper bound.",
        "- n = 1 paper × 1 run with stochastic models: these are real numbers from one run, not a benchmark. "
        "Re-run the script to get another sample.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


def rescore(args) -> int:
    """Re-score a finished eval run from runs/<id>/report.json (no LLM calls)."""
    s = get_settings()
    report = json.loads((s.runs_dir / args.rescore / "report.json").read_text(encoding="utf-8"))
    ev = evaluate(report, PLANTS_1706)
    ev["prompts"] = args.prompts or prompts_hash()
    out_dir = ROOT / "evals"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.rescore}.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "results.md", ev, report, args, args.rescore)
    print(json.dumps({k: v for k, v in ev.items() if k not in ("rows", "others")}, indent=2))
    return 0


async def main_async(args) -> int:
    s = get_settings()
    run_prompts = prompts_hash()  # hash the prompts the run actually used
    async with httpx.AsyncClient(timeout=90) as http:
        data = await arxiv.download_pdf(http, args.arxiv)
    parsed = apply_plants(parse_pdf(data), PLANTS_1706)
    run_id = args.run_id or f"eval-planted-{time.strftime('%Y%m%d-%H%M%S')}"
    st = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30), s.collection)
    st.ensure_collection()
    st.delete_run(run_id)
    bus = EventBus(run_id, s.runs_dir / run_id / "events.jsonl")
    params = RunParams(max_claims=args.max_claims, max_papers=args.max_papers, debate_rounds=args.debate_rounds)
    report = await run_pipeline(run_id=run_id, settings=s, store=st, bus=bus, params=params,
                                arxiv_id=args.arxiv, parsed=parsed)
    if report is None:
        print(f"run failed: see {s.runs_dir / run_id / 'error.txt'}", file=sys.stderr)
        return 1
    ev = evaluate(report, PLANTS_1706)
    ev["prompts"] = run_prompts
    out_dir = ROOT / "evals"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "results.md", ev, report, args, run_id)
    print(json.dumps({k: v for k, v in ev.items() if k not in ("rows", "others")}, indent=2))
    print(f"wrote {out_dir / 'results.md'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", default="1706.03762")
    ap.add_argument("--max-claims", type=int, default=6)
    ap.add_argument("--max-papers", type=int, default=10)
    ap.add_argument("--debate-rounds", type=int, default=2)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--rescore", metavar="RUN_ID", help="re-score an existing eval run without calling the LLM")
    ap.add_argument("--prompts", help="with --rescore: prompts hash the run was made with (default: current)")
    args = ap.parse_args()
    if args.rescore:
        raise SystemExit(rescore(args))
    if args.arxiv != "1706.03762":
        raise SystemExit("the planted edits are written against 1706.03762; add a PLANTS table for other papers")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
