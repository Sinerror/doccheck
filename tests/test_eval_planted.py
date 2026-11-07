"""Scoring logic of scripts/eval_planted.py and integrity of the labelled set (no network, no LLM)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_planted import LABELS, assign, render, score_doc, summarize  # noqa: E402
from doccheck.sources.local import load_corpus, load_file  # noqa: E402

LAB = json.loads(LABELS.read_text(encoding="utf-8"))


def test_labels_are_consistent_with_docs_and_corpus():
    sections = {(d.doc_id, s) for d in load_corpus(ROOT / "data" / "corpus") for s, _ in d.parsed.sections}
    kinds = set()
    for d in LAB["docs"]:
        text = re.sub(r"\s+", " ", load_file(ROOT / "data" / "eval" / "docs" / d["file"]).full_text)
        for it in d["planted"] + d["clean"]:
            assert re.sub(r"\s+", " ", it["text"]) in text, it["id"]
            for g in it["gold"]:
                assert (g["file"], g["section"]) in sections, (it["id"], g)
        kinds |= {p["kind"] for p in d["planted"]}
    assert len(kinds) == 8 and LAB["totals"]["planted"] >= 30


def _c(cid, quote, r0, final, status="accepted", suggested=None, text=None):
    return {"id": cid, "text": text or quote, "quote": quote, "verdict": final, "status": status,
            "critic_suggested": suggested, "history": [{"analyst": {"verdict": r0}}]}


def test_matching_uses_the_quote():
    d = LAB["docs"][0]
    a = d["planted"][0]
    claims = [_c("C1", a["text"], "SUPPORTED", "SUPPORTED",
                 text="Полный возврат возможен в течение 30 календарных дней"),
              _c("C2", "Мы всегда рады помочь вам с любыми вопросами.", "SUPPORTED", "SUPPORTED")]
    m = assign(claims, d["planted"] + d["clean"])
    assert m == {"C1": a["id"], "C2": None}


def test_scoring_strict_flagged_critic_only_and_fp():
    d = LAB["docs"][0]  # d01: A outdated, B condition, C number; ok1, ok2
    A, B, C = d["planted"]
    ok1, ok2 = d["clean"]
    report = {"run_id": "r", "metrics": {"cost_usd": 0.2, "elapsed_s": 10, "llm_calls": 5},
              "aggregate": {"grounding": {"ok": 3, "checks": 3}, "needs_review": {"claims": {"C1": [], "C2": [],
                                                                                               "C4": []}}},
              "claims": [
                  _c("C1", A["text"], "CONTRADICTED", "CONTRADICTED"),                          # analyst catches
                  _c("C2", B["text"], "SUPPORTED", "SUPPORTED", "disputed", "CONTRADICTED"),     # critic-only
                  # C not extracted
                  _c("C3", ok1["text"], "SUPPORTED", "SUPPORTED"),
                  _c("C4", ok2["text"], "SUPPORTED", "OVERSTATED", "revised"),                 # false positive
              ]}
    sd = score_doc(report, d)
    rows = {r["id"]: r for r in sd["rows"]}
    assert rows[A["id"]]["caught"] and rows[A["id"]]["caught_r0"] and not rows[A["id"]]["critic_only"]
    assert rows[B["id"]]["caught"] and rows[B["id"]]["critic_only"] and rows[B["id"]]["flagged"]
    assert not rows[C["id"]]["extracted"] and not rows[C["id"]]["caught"]
    s = summarize([sd])
    assert s["recall_strict"] == round(2 / 3, 3) and s["critic_only"] == 1 and s["recall_r0"] == round(1 / 3, 3)
    assert s["fp_strict_clean"] == 0.5 and s["clean_claims"] == 2
    assert s["by_kind"]["outdated"]["caught"] == 1
    md = render([{"eval_id": "e1", "label": "full", "created_at": "2026-11-07 10:00", "prompts": "p",
                  "models": {"scout": "s", "analyst": "a", "critic": "c"}, "embed_model": "m",
                  "params": {"use_critic": True, "max_claims": 10, "debate_rounds": 2},
                  "summary": s, "docs": [sd]}])
    assert "recall, strict" in md and "| outdated" in md
