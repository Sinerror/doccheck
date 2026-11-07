"""Scoring logic of scripts/eval_planted.py (no network, no LLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_planted import PLANTS_1706, apply_plants, evaluate, matches  # noqa: E402
from doccheck.sources.pdf import ParsedPDF  # noqa: E402


def _claim(cid, text, rounds, final, status, suggested=None):
    return {"id": cid, "text": text, "verdict": final, "status": status, "critic_suggested": suggested,
            "history": [{"analyst": {"verdict": a}, "critic": {"stance": st, "issues": iss, "suggested_verdict": sg}}
                        for a, st, iss, sg in rounds]}


def test_matching_is_specific():
    a, b, c = PLANTS_1706
    assert matches(a, "Our model achieves 31 . 7 BLEU on WMT 2014 English-to-German")
    assert not matches(a, "Our model achieves 28.4 BLEU on WMT 2014 English-to-German")
    assert matches(b, "We are the first to apply attention mechanisms to neural machine translation.")
    assert not matches(b, "the Transformer, the first sequence transduction model based entirely on attention")
    assert matches(c, "ConvS2S fails to converge on WMT'14 En-Fr without pre-training")


def test_apply_plants_edits_text():
    parsed = ParsedPDF(title="T", sections=[
        ("Abstract", "dispensing with recurrence and convolutions entirely. Our model achieves 28.4 BLEU on X, "
                     "including ensembles, by over 2 BLEU. a small fraction of the training costs of the best models "
                     "from the literature."),
        ("6.1 Machine Translation", "by more than 2 . 0 BLEU, establishing a new state-of-the-art BLEU score of 28 . 4 "
                                    "and outperforming all of the previously published single models, at less"),
        ("7 Conclusion", "with multi-headed self-attention. More."),
    ])
    out = apply_plants(parsed, PLANTS_1706)
    text = " ".join(t for _, t in out.sections)
    assert "31.7 BLEU" in text and "31 . 7" in text and "first to apply attention" in text and "ConvS2S" in text
    assert "28.4 BLEU" not in text


def test_evaluate_counts_analyst_critic_and_false_positives():
    hi = [{"severity": "high", "type": "wrong_verdict"}]
    report = {
        "claims": [
            # A: analyst catches it immediately
            _claim("C1", "Our model achieves 31.7 BLEU on English-German", [("CONTRADICTED", "AGREE", [], None)],
                   "CONTRADICTED", "accepted"),
            # B: analyst says SUPPORTED, critic challenges, analyst revises to OVERSTATED
            _claim("C2", "We are the first to apply attention mechanisms to neural machine translation",
                   [("SUPPORTED", "CHALLENGE", hi, "CONTRADICTED"), ("OVERSTATED", "AGREE", [], None)],
                   "OVERSTATED", "revised"),
            # C not extracted; one clean claim wrongly contradicted (false positive)
            _claim("C3", "The Transformer trains faster than recurrent models", [("CONTRADICTED", "AGREE", [], None)],
                   "CONTRADICTED", "accepted"),
        ],
        "aggregate": {"grounding": {"rate": 1.0, "ok": 3, "checks": 3}},
    }
    ev = evaluate(report, PLANTS_1706)
    rows = {r["plant"]: r for r in ev["rows"]}
    assert rows["A"]["caught_by_analyst"] and not rows["A"]["caught_by_critic_only"]
    assert rows["B"]["caught_by_critic_only"] and rows["B"]["caught_final"]
    assert not rows["C"]["extracted"] and not rows["C"]["caught_any"]
    assert round(ev["recall_any"], 3) == round(2 / 3, 3) and ev["critic_only"] == 1
    assert ev["fp_strict"] == 1 and ev["n_others"] == 1
