"""Deterministic aggregation and the human-readable report. The only LLM call around here is the short
summary prose, and it only sees already-verified results. What goes to a human is decided by a rule, not an LLM."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .models import SEVERITY_RANK, VERDICT_RU

LOW_CONFIDENCE = 0.6
VERDICT_ICON = {"SUPPORTED": "✅", "PARTIALLY_SUPPORTED": "🟡", "CONTRADICTED": "❌", "OVERSTATED": "🟠",
                "UNVERIFIABLE": "❔"}
STATUS_RU = {"accepted": "критик согласен", "revised": "пересмотрено после спора", "disputed": "спорное"}
NEEDS_REVIEW_RULE = ("claim goes to a human if: final verdict ≠ SUPPORTED, or analyst and critic still disagree "
                     f"(disputed), or confidence < {LOW_CONFIDENCE}, or a quote in the final verdict failed "
                     "the grounding check")


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


def review_reasons(c: dict[str, Any]) -> list[str]:
    reasons = []
    if c["verdict"] != "SUPPORTED":
        reasons.append(VERDICT_RU[c["verdict"]])
    if c.get("status") == "disputed":
        reasons.append("аналитик и критик не согласились")
    if (c.get("confidence") or 0) < LOW_CONFIDENCE:
        reasons.append(f"низкая уверенность ({c.get('confidence')})")
    last = (c.get("history") or [{}])[-1]
    if any(not g.get("ok") for g in last.get("grounding") or []):
        reasons.append("цитата не прошла проверку")
    return reasons


def aggregate(claims: list[dict[str, Any]], grounding_checks: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(c["verdict"] for c in claims)
    statuses = Counter(c["status"] for c in claims)
    issues_by_type: Counter[str] = Counter()
    issues_by_sev: Counter[str] = Counter()
    for c in claims:
        for h in c.get("history", []):
            for i in (h.get("critic") or {}).get("issues") or []:
                issues_by_type[i["type"]] += 1
                issues_by_sev[i["severity"]] += 1
    review = {c["id"]: r for c in claims if (r := review_reasons(c))}
    ok = sum(1 for g in grounding_checks if g["ok"])
    by_role: dict[str, dict[str, int]] = {}
    for g in grounding_checks:
        r = by_role.setdefault(g.get("by_role", "analyst"), {"checks": 0, "ok": 0})
        r["checks"] += 1
        r["ok"] += int(g["ok"])
    return {
        "verdict_counts": dict(verdicts),
        "status_counts": {k: statuses.get(k, 0) for k in ("accepted", "revised", "disputed")},
        "needs_review": {"count": len(review), "share": round(len(review) / len(claims), 3) if claims else None,
                         "claims": review, "rule": NEEDS_REVIEW_RULE},
        "grounding": {
            "rate": round(ok / len(grounding_checks), 4) if grounding_checks else None,
            "checks": len(grounding_checks), "ok": ok, "by_role": by_role,
        },
        "critic_issues": {"total": sum(issues_by_type.values()), "by_type": dict(issues_by_type),
                          "by_severity": dict(issues_by_sev)},
    }


def _md(s: str | None) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(report: dict[str, Any]) -> str:
    t = report["target"]
    agg = report["aggregate"]
    m = report.get("metrics") or {}
    claims = report["claims"]
    nr = agg["needs_review"]
    L: list[str] = [f"# Проверка документа: {t.get('title')}", ""]
    L.append(f"- Файл: `{t.get('filename')}` · {t.get('chars')} символов (~{t.get('pages')} стр.)")
    L.append(f"- Прогон: `{report['run_id']}` · модели: " +
             ", ".join(f"{k} = `{v}`" for k, v in report["models"].items()))
    corp = report.get("corpus") or {}
    if corp:
        L.append(f"- Источники: {len(corp.get('docs', []))} документов, {corp.get('chunks')} фрагментов, "
                 f"эмбеддинги `{corp.get('embed_model')}`")
    L.append(f"- **Требуют внимания человека: {nr['count']} из {len(claims)}** · "
             f"цитаты проверены кодом: {agg['grounding']['ok']}/{agg['grounding']['checks']}"
             + (f" ({agg['grounding']['rate']:.0%})" if agg["grounding"]["rate"] is not None else ""))
    if m:
        L.append(f"- Стоимость: ${m.get('cost_usd', 0):.4f} (оценка по прайсу) · {m.get('llm_calls')} вызовов LLM · "
                 f"{m.get('elapsed_s')} с")
    L += ["", "## Итог", "", report.get("summary") or "_резюме недоступно_", ""]
    L += ["## Утверждения", "", "| # | Утверждение | Вердикт | Статус | На проверку |", "|---|---|---|---|---|"]
    for c in claims:
        flag = "👁 " + "; ".join(nr["claims"][c["id"]]) if c["id"] in nr["claims"] else ""
        L.append(f"| {c['id']} | {_md(c['text'])} | {VERDICT_ICON.get(c['verdict'], '')} {VERDICT_RU[c['verdict']]} | "
                 f"{STATUS_RU.get(c['status'], c['status'])} | {flag} |")
    L.append("")
    L += ["## Что исправить", ""]
    fixes = [c for c in claims if c["verdict"] in ("CONTRADICTED", "OVERSTATED", "PARTIALLY_SUPPORTED")]
    if not fixes:
        L.append("- ошибок не найдено")
    for c in fixes:
        L.append(f"- **{c['id']}** {VERDICT_ICON[c['verdict']]} «{_md(c.get('quote') or c['text'])}»")
        if c.get("correction"):
            L.append(f"  - по источникам: {_md(c['correction'])}")
    unv = [c for c in claims if c["verdict"] == "UNVERIFIABLE"]
    if unv:
        L += ["", "## Не найдено в источниках (уточнить у ответственных)", ""]
        for c in unv:
            L.append(f"- **{c['id']}** «{_md(c.get('quote') or c['text'])}»")
    L += ["", "## Подробно", ""]
    for c in claims:
        L.append(f"### {c['id']} · {VERDICT_ICON.get(c['verdict'], '')} {VERDICT_RU[c['verdict']]} · "
                 f"{STATUS_RU.get(c['status'], c['status'])}")
        L.append("")
        L.append(f"> {c.get('quote') or c['text']}")
        L.append("")
        L.append(f"*Утверждение: {c['text']}* · тип `{c['type']}` · уверенность {c.get('confidence')}")
        L.append("")
        for h in c["history"]:
            a = h["analyst"]
            L.append(f"**Раунд {h['round']} — аналитик: {VERDICT_RU[a['verdict']]}** (уверенность {a.get('confidence')})")
            L.append("")
            L.append((a.get("reasoning") or "").strip())
            if a.get("response_to_critic"):
                L += ["", f"*Ответ критику:* {a['response_to_critic'].strip()}"]
            L.append("")
            for e, g in zip(a.get("evidence", []), h.get("grounding", [])):
                mark = "✅" if g["ok"] else "❌ цитата не найдена"
                L.append(f"- {mark} [{e['stance']}] `{e['chunk_id']}`: «{_md(e['quote'])}» (совпадение {g['score']:.0f})")
            for con in a.get("concerns", []):
                L.append(f"- оговорка: {con}")
            cr = h.get("critic")
            if cr:
                sugg = f", предлагает «{VERDICT_RU[cr['suggested_verdict']]}»" if cr.get("suggested_verdict") else ""
                L += ["", f"**Раунд {h['round']} — критик: {cr['stance']}**{sugg}", "", (cr.get("reasoning") or "").strip()]
                for i in cr.get("issues", []):
                    L.append(f"- [{i['severity']}] `{i['type']}`: {i['detail']}")
                    for ce in i.get("counter_evidence", []):
                        L.append(f"  - `{ce['chunk_id']}`: «{_md(ce['quote'])}»")
            L.append("")
        if c["status"] == "disputed":
            L.append(f"**Спорное.** Аналитик: {VERDICT_RU[c['verdict']]}; критик: "
                     f"{VERDICT_RU.get(c.get('critic_suggested') or '', 'другой вердикт')}. Нужно решение человека.")
            L.append("")
    if report.get("claim_errors"):
        L += ["## Ошибки выполнения", ""]
        for e in report["claim_errors"]:
            L.append(f"- {e['claim_id']}: {e['error']}")
        L.append("")
    L += ["---", "", f"_Правило отбора для человека: {NEEDS_REVIEW_RULE}._", ""]
    return "\n".join(L)
