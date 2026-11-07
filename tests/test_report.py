import pytest

from doccheck.report import aggregate, debate_status, is_blocking, max_severity, render_markdown, review_reasons


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


def _claim(cid, verdict, status, issues=(), suggested=None, confidence=0.9, grounded=True, correction=None):
    return {"id": cid, "text": f"утверждение {cid}", "quote": f"Предложение {cid}.", "type": "number",
            "verdict": verdict, "status": status, "confidence": confidence, "critic_suggested": suggested,
            "correction": correction,
            "history": [{"round": 0, "analyst": {"verdict": verdict, "reasoning": "r", "confidence": confidence,
                                                 "evidence": [{"chunk_id": "a#0", "quote": "q", "stance": "supports"}],
                                                 "concerns": []},
                         "grounding": [{"ok": grounded, "score": 100 if grounded else 40}],
                         "critic": {"stance": "AGREE", "reasoning": "ok", "issues": list(issues),
                                    "suggested_verdict": suggested}}]}


def test_review_rule():
    assert review_reasons(_claim("C1", "SUPPORTED", "accepted")) == []
    assert review_reasons(_claim("C2", "CONTRADICTED", "accepted")) == ["противоречит источникам"]
    assert "аналитик и критик не согласились" in review_reasons(_claim("C3", "SUPPORTED", "disputed"))
    assert any("низкая" in r for r in review_reasons(_claim("C4", "SUPPORTED", "accepted", confidence=0.4)))
    assert "цитата не прошла проверку" in review_reasons(_claim("C5", "SUPPORTED", "accepted", grounded=False))


def test_aggregate_and_markdown():
    claims = [
        _claim("C1", "SUPPORTED", "accepted"),
        _claim("C2", "CONTRADICTED", "revised", [{"type": "missed_contradiction", "severity": "high", "detail": "d"}],
               correction="Срок возврата — 14 календарных дней."),
        _claim("C3", "SUPPORTED", "disputed", [{"type": "outdated_source", "severity": "medium", "detail": "d"},
                                               {"type": "overclaim", "severity": "low", "detail": "d"}], "CONTRADICTED"),
        _claim("C4", "UNVERIFIABLE", "accepted"),
    ]
    checks = [{"ok": True, "by_role": "analyst"}, {"ok": False, "by_role": "analyst"}, {"ok": True, "by_role": "critic"}]
    agg = aggregate(claims, checks)
    assert agg["status_counts"] == {"accepted": 2, "revised": 1, "disputed": 1}
    assert agg["verdict_counts"] == {"SUPPORTED": 2, "CONTRADICTED": 1, "UNVERIFIABLE": 1}
    assert agg["critic_issues"]["by_type"] == {"missed_contradiction": 1, "outdated_source": 1, "overclaim": 1}
    assert agg["grounding"]["rate"] == round(2 / 3, 4)
    assert agg["needs_review"]["count"] == 3 and agg["needs_review"]["share"] == 0.75
    assert set(agg["needs_review"]["claims"]) == {"C2", "C3", "C4"}
    report = {"run_id": "t", "target": {"title": "Ответ клиенту", "filename": "a.md", "chars": 900, "pages": 0.5},
              "models": {"analyst": "m"}, "aggregate": agg, "summary": "Есть ошибка в сроке возврата.",
              "claims": claims, "metrics": {"cost_usd": 0.1, "llm_calls": 9, "elapsed_s": 30}, "corpus": {}}
    md = render_markdown(report)
    assert "Требуют внимания человека: 3 из 4" in md
    assert "по источникам: Срок возврата — 14 календарных дней." in md
    assert "## Не найдено в источниках" in md and "**Спорное.**" in md
