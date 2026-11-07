"""Generate n8n/triad_daily_arxiv.json and n8n/triad_on_demand.json (SPEC §9).

Conservative node typeVersions, all supported by n8n 2.43.8 (the pinned image):
scheduleTrigger 1.2, httpRequest 4.2, code 2, wait 1.1, webhook 2, respondToWebhook 1.1, telegram 1.2, emailSend 2.1.

    uv run python scripts/gen_n8n_workflows.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://host.docker.internal:8000/api/runs"


def node(id_: str, name: str, type_: str, ver: float, pos: list[int], params: dict, **extra) -> dict:
    d = {"parameters": params, "id": id_, "name": name, "type": type_, "typeVersion": ver, "position": pos}
    d.update(extra)
    return d


def chain(*names: str) -> dict:
    return {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(names, names[1:])}


PICK_JS = r"""// pick the newest entry of the arXiv Atom feed
const xml = String($input.first().json.data ?? '');
const entry = xml.split('<entry>')[1];
if (!entry) throw new Error('arXiv feed has no entries');
const id = (entry.match(/<id>\s*https?:\/\/arxiv\.org\/abs\/([^<\s]+?)(v\d+)?\s*<\/id>/) || [])[1];
const title = ((entry.match(/<title>([\s\S]*?)<\/title>/) || [])[1] || '').replace(/\s+/g, ' ').trim();
if (!id) throw new Error('could not parse arXiv id');
return [{ json: { arxiv_id: id, title } }];"""

READ_ID_JS = r"""const b = $input.first().json.body ?? {};
const id = String(b.arxiv_id ?? '').trim();
if (!id) throw new Error('POST a JSON body like {"arxiv_id": "1706.03762"}');
return [{ json: { arxiv_id: id } }];"""

FORMAT_JS = r"""// TRIAD POSTs runs/{id}/report.json to the resume URL; the Wait node exposes it as $json.body
const r = $input.first().json.body ?? $input.first().json;
const agg = r.aggregate ?? {};
const sc = agg.status_counts ?? {};
const lines = [
  `TRIAD review: ${r.target?.title ?? 'unknown paper'}`,
  `${r.target?.url ?? ''}`,
  `score ${r.review_score ?? '?'} / 10 · grounding ${Math.round((r.grounding_rate ?? 0) * 100)}% · ` +
    `${sc.accepted ?? 0} accepted / ${sc.revised ?? 0} revised / ${sc.disputed ?? 0} disputed`,
  '',
  ...(r.claims ?? []).map(c => `${c.id} [${c.status}] ${c.verdict}: ${String(c.text).slice(0, 160)}`),
  '',
  String(r.meta_review ?? '').slice(0, 1500),
];
return [{ json: { text: lines.join('\n'), run_id: r.run_id, review_score: r.review_score,
                   grounding_rate: r.grounding_rate, status_counts: sc } }];"""


def start_run(id_: str, pos: list[int]) -> dict:
    return node(id_, "Start TRIAD run", "n8n-nodes-base.httpRequest", 4.2, pos, {
        "method": "POST",
        "url": API,
        "sendBody": True,
        "specifyBody": "json",
        "jsonBody": "={{ JSON.stringify({ arxiv_id: $json.arxiv_id, max_claims: 4, max_papers: 10, "
                    "debate_rounds: 2, callback_url: $execution.resumeUrl }) }}",
        "options": {"timeout": 60000},
    })


def wait(id_: str, pos: list[int], webhook_id: str) -> dict:
    return node(id_, "Wait for report", "n8n-nodes-base.wait", 1.1, pos, {
        "resume": "webhook",
        "httpMethod": "POST",
        "limitWaitTime": True,
        "limitType": "afterTimeInterval",
        "resumeAmount": 45,
        "resumeUnit": "minutes",
        "options": {},
    }, webhookId=webhook_id)


def fmt(id_: str, pos: list[int]) -> dict:
    return node(id_, "Format short report", "n8n-nodes-base.code", 2, pos,
                {"mode": "runOnceForAllItems", "jsCode": FORMAT_JS})


def daily() -> dict:
    return {
        "name": "TRIAD · daily arXiv review",
        "nodes": [
            node("a1b2c3d4-0001-4000-8000-000000000001", "Every day 09:00", "n8n-nodes-base.scheduleTrigger", 1.2,
                 [0, 300], {"rule": {"interval": [{"field": "cronExpression", "expression": "0 9 * * *"}]}}),
            node("a1b2c3d4-0002-4000-8000-000000000002", "arXiv: newest cs.CL / cs.LG", "n8n-nodes-base.httpRequest",
                 4.2, [220, 300], {
                     "url": "https://export.arxiv.org/api/query",
                     "sendQuery": True,
                     "queryParameters": {"parameters": [
                         {"name": "search_query", "value": "cat:cs.CL OR cat:cs.LG"},
                         {"name": "sortBy", "value": "submittedDate"},
                         {"name": "sortOrder", "value": "descending"},
                         {"name": "max_results", "value": "10"},
                     ]},
                     "options": {"response": {"response": {"responseFormat": "text"}}},
                 }),
            node("a1b2c3d4-0003-4000-8000-000000000003", "Pick one paper", "n8n-nodes-base.code", 2, [440, 300],
                 {"mode": "runOnceForAllItems", "jsCode": PICK_JS}),
            start_run("a1b2c3d4-0004-4000-8000-000000000004", [660, 300]),
            wait("a1b2c3d4-0005-4000-8000-000000000005", [880, 300], "5f1d2c4e-7a8b-4c9d-8e0f-a1b2c3d4e5f6"),
            fmt("a1b2c3d4-0006-4000-8000-000000000006", [1100, 300]),
            node("a1b2c3d4-0007-4000-8000-000000000007", "Telegram", "n8n-nodes-base.telegram", 1.2, [1320, 200],
                 {"chatId": "REPLACE_WITH_CHAT_ID", "text": "={{ $json.text }}", "additionalFields": {}},
                 disabled=True, notes="Disabled: needs Telegram API credentials and a chat id."),
            node("a1b2c3d4-0008-4000-8000-000000000008", "Email", "n8n-nodes-base.emailSend", 2.1, [1320, 400], {
                "fromEmail": "triad@example.com", "toEmail": "you@example.com",
                "subject": "=TRIAD review: {{ $json.review_score }}/10", "emailFormat": "text",
                "text": "={{ $json.text }}", "options": {}},
                 disabled=True, notes="Disabled: needs SMTP credentials."),
        ],
        "connections": {
            **chain("Every day 09:00", "arXiv: newest cs.CL / cs.LG", "Pick one paper", "Start TRIAD run",
                    "Wait for report", "Format short report"),
            "Format short report": {"main": [[{"node": "Telegram", "type": "main", "index": 0},
                                              {"node": "Email", "type": "main", "index": 0}]]},
        },
        "active": False,
        "settings": {"executionOrder": "v1"},
        "pinData": {},
        "id": "TriadDailyArxiv01",
        "tags": [],
    }


def on_demand() -> dict:
    return {
        "name": "TRIAD · on-demand review",
        "nodes": [
            node("b1b2c3d4-0001-4000-8000-000000000001", "Webhook: POST arxiv_id", "n8n-nodes-base.webhook", 2,
                 [0, 300], {"httpMethod": "POST", "path": "triad-review", "responseMode": "responseNode", "options": {}},
                 webhookId="0d6c9a3e-2f4b-4e8a-9c1d-7b5e3f2a1c0d"),
            node("b1b2c3d4-0002-4000-8000-000000000002", "Read arxiv_id", "n8n-nodes-base.code", 2, [220, 300],
                 {"mode": "runOnceForAllItems", "jsCode": READ_ID_JS}),
            start_run("b1b2c3d4-0004-4000-8000-000000000004", [440, 300]),
            wait("b1b2c3d4-0005-4000-8000-000000000005", [660, 300], "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"),
            fmt("b1b2c3d4-0006-4000-8000-000000000006", [880, 300]),
            node("b1b2c3d4-0007-4000-8000-000000000007", "Respond to Webhook", "n8n-nodes-base.respondToWebhook", 1.1,
                 [1100, 300], {"respondWith": "firstIncomingItem", "options": {}}),
        ],
        "connections": chain("Webhook: POST arxiv_id", "Read arxiv_id", "Start TRIAD run", "Wait for report",
                             "Format short report", "Respond to Webhook"),
        "active": False,
        "settings": {"executionOrder": "v1"},
        "pinData": {},
        "id": "TriadOnDemand001",
        "tags": [],
    }


def main() -> None:
    out = ROOT / "n8n"
    out.mkdir(exist_ok=True)
    for name, wf in (("triad_daily_arxiv.json", daily()), ("triad_on_demand.json", on_demand())):
        (out / name).write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote n8n/{name}")


if __name__ == "__main__":
    main()
