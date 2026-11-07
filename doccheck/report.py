"""Deterministic aggregation (SPEC §3, stage 4). The only LLM call here is the meta-review prose,
and it only sees already-verified results. `review_score` is a transparent formula, not an LLM opinion."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .models import SEVERITY_RANK

VERDICT_WEIGHT = {
    "SUPPORTED": 1.0,
    "PARTIALLY_SUPPORTED": 0.6,
    "UNVERIFIABLE": 0.5,
    "OVERSTATED": 0.4,
    "CONTRADICTED": 0.0,
}

SCORE_FORMULA = (
    "claim_score = VERDICT_WEIGHT[final analyst verdict] "
    "(SUPPORTED 1.0, PARTIALLY_SUPPORTED 0.6, UNVERIFIABLE 0.5, OVERSTATED 0.4, CONTRADICTED 0.0); "
    "for a disputed claim with a critic-suggested verdict, claim_score = mean(analyst weight, critic weight); "
    "review_score = round(10 * mean(claim_score), 1). grounding_rate is reported separately as a trust signal."
)


def max_severity(critique: dict[str, Any] | None) -> str | None:
    if not critique:
        return None
    sev = [i.get("severity") for i in critique.get("issues") or [] if i.get("severity") in SEVERITY_RANK]
    return max(sev, key=SEVERITY_RANK.__getitem__) if sev else None


def is_blocking(stance: str, severity: str | None) -> bool:
    """A critique forces a revision only if it CHALLENGEs with at least one medium/high issue."""
    return stance == "CHALLENGE" and severity is not None and SEVERITY_RANK[severity] >= SEVERITY_RANK["medium"]


def debate_status(rounds: list[dict[str, Any]]) -> str:
    """Reducer over debate rounds [{stance, max_severity}, ...] → accepted | revised | disputed."""
    if not rounds:
        raise ValueError("no debate rounds")
    if not is_blocking(rounds[0]["stance"], rounds[0].get("max_severity")):
        return "accepted"
    last = rounds[-1]
    if not is_blocking(last["stance"], last.get("max_severity")):
        return "revised"
    return "disputed"


def claim_score(final_verdict: str, status: str, critic_suggested: str | None) -> float:
    w = VERDICT_WEIGHT.get(final_verdict, 0.5)
    if status == "disputed" and critic_suggested in VERDICT_WEIGHT:
        w = (w + VERDICT_WEIGHT[critic_suggested]) / 2
    return round(w, 4)


def aggregate(claims: list[dict[str, Any]], grounding_checks: list[dict[str, Any]]) -> dict[str, Any]:
    """claims: [{id, status, verdict, critic_suggested, history:[{critic:{issues}}...]}]."""
    verdicts = Counter(c["verdict"] for c in claims)
    statuses = Counter(c["status"] for c in claims)
    issues_by_type: Counter[str] = Counter()
    issues_by_sev: Counter[str] = Counter()
    for c in claims:
        for h in c.get("history", []):
            for i in (h.get("critic") or {}).get("issues") or []:
                issues_by_type[i["type"]] += 1
                issues_by_sev[i["severity"]] += 1
    scores = [claim_score(c["verdict"], c["status"], c.get("critic_suggested")) for c in claims]
    ok = sum(1 for g in grounding_checks if g["ok"])
    by_role: dict[str, dict[str, int]] = {}
    for g in grounding_checks:
        r = by_role.setdefault(g.get("by_role", "analyst"), {"checks": 0, "ok": 0})
        r["checks"] += 1
        r["ok"] += int(g["ok"])
    return {
        "review_score": round(10 * sum(scores) / len(scores), 1) if scores else None,
        "score_formula": SCORE_FORMULA,
        "claim_scores": {c["id"]: s for c, s in zip(claims, scores)},
        "verdict_counts": dict(verdicts),
        "status_counts": {k: statuses.get(k, 0) for k in ("accepted", "revised", "disputed")},
        "grounding": {
            "rate": round(ok / len(grounding_checks), 4) if grounding_checks else None,
            "checks": len(grounding_checks), "ok": ok, "by_role": by_role,
        },
        "critic_issues": {"total": sum(issues_by_type.values()), "by_type": dict(issues_by_type),
                          "by_severity": dict(issues_by_sev)},
    }


def render_markdown(report: dict[str, Any]) -> str:
    t = report["target"]
    agg = report["aggregate"]
    L: list[str] = []
    L.append(f"# TRIAD review: {t.get('title')}")
    L.append("")
    L.append(f"- Paper: {t.get('url') or t.get('paper_id')}")
    L.append(f"- Run: `{report['run_id']}` · models: " + ", ".join(f"{k}=`{v}`" for k, v in report["models"].items()))
    L.append(f"- **review_score: {agg['review_score']} / 10** · grounding_rate: {agg['grounding']['rate']}"
             f" ({agg['grounding']['ok']}/{agg['grounding']['checks']} quotes verified)")
    sc = agg["status_counts"]
    L.append(f"- Claims: {sc['accepted']} accepted · {sc['revised']} revised · {sc['disputed']} disputed")
    m = report.get("metrics") or {}
    if m:
        L.append(f"- Cost: ${m.get('cost_usd', 0):.4f} (estimate) · {m.get('llm_calls')} LLM calls · "
                 f"{m.get('tokens_in')} in / {m.get('tokens_out')} out tokens · {m.get('elapsed_s')} s")
    L.append("")
    L.append("## Meta-review")
    L.append("")
    L.append(report.get("meta_review") or "_not available_")
    L.append("")
    L.append("## Claims")
    L.append("")
    L.append("| # | Claim | Final verdict | Status | Score |")
    L.append("|---|---|---|---|---|")
    for c in report["claims"]:
        text = c["text"].replace("|", "\\|")
        L.append(f"| {c['id']} | {text} | {c['verdict']} | {c['status']} | {agg['claim_scores'][c['id']]} |")
    L.append("")
    ci = agg["critic_issues"]
    L.append("## What the critic caught")
    L.append("")
    if ci["total"]:
        for k, v in sorted(ci["by_type"].items(), key=lambda kv: -kv[1]):
            L.append(f"- `{k}`: {v}")
        L.append(f"- by severity: " + ", ".join(f"{k}={v}" for k, v in ci["by_severity"].items()))
    else:
        L.append("- no issues raised")
    L.append("")
    for c in report["claims"]:
        L.append(f"### {c['id']} · {c['status'].upper()} · {c['verdict']}")
        L.append("")
        L.append(f"> {c['text']}")
        L.append("")
        L.append(f"*{c['type']}, section “{c['section']}”, confidence {c.get('confidence')}*")
        L.append("")
        for h in c["history"]:
            a = h["analyst"]
            L.append(f"**Round {h['round']} — Analyst: {a['verdict']}** (confidence {a.get('confidence')})")
            L.append("")
            L.append(a.get("reasoning", "").strip())
            if a.get("response_to_critic"):
                L.append("")
                L.append(f"*Response to critic:* {a['response_to_critic'].strip()}")
            L.append("")
            for e, g in zip(a.get("evidence", []), h.get("grounding", [])):
                mark = "✅" if g["ok"] else "❌ ungrounded"
                L.append(f"- {mark} [{e['stance']}] `{e['chunk_id']}`: “{e['quote']}” (match {g['score']:.0f})")
            for con in a.get("concerns", []):
                L.append(f"- concern: {con}")
            cr = h.get("critic")
            if cr:
                L.append("")
                sugg = f", suggests {cr['suggested_verdict']}" if cr.get("suggested_verdict") else ""
                L.append(f"**Round {h['round']} — Critic: {cr['stance']}**{sugg}")
                L.append("")
                L.append(cr.get("reasoning", "").strip())
                for i in cr.get("issues", []):
                    L.append(f"- [{i['severity']}] `{i['type']}`: {i['detail']}")
                    for ce in i.get("counter_evidence", []):
                        L.append(f"  - `{ce['chunk_id']}`: “{ce['quote']}”")
            L.append("")
        if c["status"] == "disputed":
            L.append(f"**Disputed.** Analyst holds {c['verdict']}; critic suggests "
                     f"{c.get('critic_suggested') or 'a different verdict'}. Both positions shown above.")
            L.append("")
    L.append("## Related papers in corpus")
    L.append("")
    for p in report.get("papers", []):
        L.append(f"- {p.get('title')} ({p.get('year') or 'n.d.'}) — {p.get('url') or p['paper_id']}"
                 f" · {p.get('chunks', 0)} chunks{' · full text' if p.get('fulltext') else ' · abstract'}")
    L.append("")
    return "\n".join(L)
