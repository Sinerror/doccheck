import pytest

from doccheck.report import aggregate, claim_score, debate_status, is_blocking, max_severity, render_markdown


def R(stance, sev=None):
    return {"stance": stance, "max_severity": sev}


@pytest.mark.parametrize("rounds,expected", [
    ([R("AGREE")], "accepted"),
    ([R("CHALLENGE", "low")], "accepted"),           # low-only challenge does not force a revision
    ([R("CHALLENGE", "high"), R("AGREE")], "revised"),
    ([R("CHALLENGE", "medium"), R("CHALLENGE", "low")], "revised"),
    ([R("CHALLENGE", "medium"), R("CHALLENGE", "high"), R("AGREE")], "revised"),
    ([R("CHALLENGE", "high"), R("CHALLENGE", "medium")], "disputed"),
    ([R("CHALLENGE", "high")], "disputed"),           # debate_rounds=0 and critic still objects
])
def test_debate_status_reducer(rounds, expected):
    assert debate_status(rounds) == expected


def test_debate_status_requires_rounds():
    with pytest.raises(ValueError):
        debate_status([])


def test_blocking_and_severity():
    assert max_severity({"issues": [{"severity": "low"}, {"severity": "high"}, {"severity": "medium"}]}) == "high"
    assert max_severity({"issues": []}) is None
    assert not is_blocking("CHALLENGE", None)
    assert not is_blocking("AGREE", "high")
    assert is_blocking("CHALLENGE", "medium")


def test_claim_score_formula():
    assert claim_score("SUPPORTED", "accepted", None) == 1.0
    assert claim_score("SUPPORTED", "disputed", "OVERSTATED") == 0.7
    assert claim_score("CONTRADICTED", "revised", "SUPPORTED") == 0.0  # only disputed claims blend


def _claim(cid, verdict, status, issues=(), suggested=None):
    return {"id": cid, "text": f"claim {cid}", "type": "empirical", "section": "Results", "verdict": verdict,
            "status": status, "confidence": 0.7, "critic_suggested": suggested,
            "history": [{"round": 0, "analyst": {"verdict": verdict, "reasoning": "r", "confidence": 0.7,
                                                 "evidence": [], "concerns": []},
                         "grounding": [], "critic": {"stance": "AGREE", "reasoning": "ok", "issues": list(issues),
                                                     "suggested_verdict": suggested}}]}


def test_aggregate_and_markdown():
    claims = [
        _claim("C1", "SUPPORTED", "accepted"),
        _claim("C2", "OVERSTATED", "revised", [{"type": "overclaim", "severity": "high", "detail": "d"}]),
        _claim("C3", "SUPPORTED", "disputed", [{"type": "missed_contradiction", "severity": "medium", "detail": "d"},
                                               {"type": "overclaim", "severity": "low", "detail": "d"}], "CONTRADICTED"),
    ]
    checks = [{"ok": True, "by_role": "analyst"}, {"ok": False, "by_role": "analyst"}, {"ok": True, "by_role": "critic"}]
    agg = aggregate(claims, checks)
    assert agg["status_counts"] == {"accepted": 1, "revised": 1, "disputed": 1}
    assert agg["verdict_counts"] == {"SUPPORTED": 2, "OVERSTATED": 1}
    assert agg["critic_issues"]["by_type"] == {"overclaim": 2, "missed_contradiction": 1}
    assert agg["grounding"]["rate"] == round(2 / 3, 4)
    assert agg["grounding"]["by_role"]["analyst"] == {"checks": 2, "ok": 1}
    # (1.0 + 0.4 + (1.0+0.0)/2) / 3 * 10 = 6.33
    assert agg["review_score"] == 6.3
    report = {"run_id": "t", "target": {"title": "T", "url": "u"}, "models": {"analyst": "m"}, "aggregate": agg,
              "meta_review": "meta", "claims": claims, "papers": [], "metrics": {}}
    md = render_markdown(report)
    assert "review_score: 6.3" in md and "### C3 · DISPUTED" in md and "Both positions shown" in md
