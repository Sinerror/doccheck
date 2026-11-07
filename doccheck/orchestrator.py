"""Deterministic async pipeline: ingest → scout → analyze (analyst ⇄ critic, per claim in parallel) → report.

Plain asyncio, no agent framework. Every step is an explicit function call and every
observable effect is an event on the bus (events.jsonl + SSE), so any run can be replayed offline.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from . import grounding
from .config import Settings, load_prompt
from .context import RunCtx, RunParams, jsonish
from .events import EventBus
from .llm import LLM
from .models import VERDICT_RU, AnalystVerdict, Claim, Summary
from .rag.chunking import chunk_document
from .rag.corpus import index_corpus
from .rag.qdrant_store import Hit, QdrantStore
from .report import aggregate, debate_status, is_blocking, max_severity, render_markdown
from .roles import analyst, critic, scout
from .sources.local import ParsedDoc

log = logging.getLogger(__name__)

MAX_DEBATE_CONTEXT = 20


class Pipeline:
    def __init__(self, ctx: RunCtx):
        self.ctx = ctx
        self.grounding_checks: list[dict[str, Any]] = []
        self.target: dict[str, Any] = {}
        self.corpus: dict[str, Any] = {}
        self.claim_errors: list[dict[str, str]] = []

    @property
    def bus(self) -> EventBus:
        return self.ctx.bus

    def stage(self, name: str) -> None:
        self.bus.emit("stage.started", {"stage": name})

    # -- stage 0: corpus + document --------------------------------------------------

    async def ingest(self, doc: ParsedDoc, filename: str, corpus_dir: Path) -> None:
        ctx = self.ctx
        digest = hashlib.sha1(doc.full_text.encode()).hexdigest()[:10]
        title = doc.title or Path(filename).stem
        self.target = {"doc_id": f"check:{filename}", "filename": filename, "title": title, "sha1": digest,
                       "chars": doc.chars, "pages": round(doc.chars / 1800, 2)}
        ctx.title = title
        self.bus.emit("run.started", {"target": self.target, "params": vars(ctx.params),
                                      "models": ctx.settings.models()}, role="system")
        self.stage("ingest")
        t0 = time.monotonic()
        idx = await asyncio.to_thread(index_corpus, ctx.store, corpus_dir)
        ctx.corpus_scope = idx.scope
        self.corpus = {"scope": idx.scope, "chunks": idx.chunks, "reused": idx.reused,
                       "embed_model": ctx.store.dense_model,
                       "docs": [{"doc_id": d.doc_id, "title": d.title, "sections": len(d.parsed.sections)}
                                for d in idx.docs]}
        for d in idx.docs:
            self.bus.emit("rag.indexed", {"doc_id": d.doc_id, "chunks": len(chunk_document(d.parsed.sections)),
                                          "total_points": idx.chunks, "source": "corpus", "title": d.title,
                                          "reused": idx.reused}, role="system")
        chunks = chunk_document(doc.sections)
        n = await asyncio.to_thread(ctx.store.upsert_chunks, ctx.run_id, self.target, chunks, "document")
        self.bus.emit("rag.indexed", {"doc_id": self.target["doc_id"], "chunks": n, "total_points": idx.chunks + n,
                                      "source": "document", "title": title, "reused": False}, role="system")
        await asyncio.to_thread(ctx.store.basis, ctx.scopes)  # fit + cache the PCA basis for query projection
        self.bus.emit("stage.finished", {"stage": "ingest", "detail": {
            "title": title, "chars": doc.chars, "sections": doc.headings, "chunks": n,
            "corpus_docs": len(idx.docs), "corpus_chunks": idx.chunks, "corpus_reused": idx.reused,
            "index_s": round(time.monotonic() - t0, 2)}})

    # -- stage 1 -------------------------------------------------------------------

    async def run_scout(self, doc: ParsedDoc) -> list[Claim]:
        self.stage("scout")
        claims = await scout.extract_claims(self.ctx, doc, self.target["title"])
        self.bus.emit("stage.finished", {"stage": "scout", "detail": {"claims": len(claims)}})
        return claims

    # -- stage 2/3: analyst ⇄ critic ---------------------------------------------------

    async def _ground(self, claim_id: str, items: list[tuple[str, str]], pool: dict[str, Hit], role: str,
                      round_: int) -> list[dict[str, Any]]:
        texts: dict[str, str | None] = {}
        for cid, _ in items:
            if cid not in texts:
                texts[cid] = pool[cid].text if cid in pool else await self.ctx.chunk_text(cid)
        results = grounding.check_all(items, lambda cid: texts.get(cid))
        out = []
        for r in results:
            d = {"claim_id": claim_id, "chunk_id": r.chunk_id, "quote": r.quote, "score": r.score, "ok": r.ok,
                 "by_role": role, "round": round_}
            self.bus.emit("grounding.check", d, role=role, claim_id=claim_id)
            self.grounding_checks.append(d)
            out.append({**d, "exists": r.exists, "reason": r.reason})
        return out

    async def debate(self, claim: Claim) -> dict[str, Any]:
        ctx = self.ctx
        hits = await analyst.gather_context(ctx, claim)
        pool: dict[str, Hit] = {h.chunk_id: h for h in hits}
        context_hits = list(hits)
        verdict: AnalystVerdict = await analyst.analyze(ctx, claim, context_hits)
        history: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        round_ = 0
        while True:
            g = await self._ground(claim.id, [(e.chunk_id, e.quote) for e in verdict.evidence], pool, "analyst",
                                   round_)
            entry: dict[str, Any] = {"round": round_, "analyst": verdict.model_dump(mode="json"), "grounding": g,
                                     "context_chunk_ids": [h.chunk_id for h in context_hits]}
            if not ctx.params.use_critic:
                history.append(entry)
                break
            crit, surfaced = await critic.critique(ctx, claim, verdict, g, context_hits, round_)
            pool.update(surfaced)
            cg = await self._ground(claim.id, [(ce.chunk_id, ce.quote) for i in crit.issues
                                               for ce in i.counter_evidence], pool, "critic", round_)
            sev = max_severity(crit.model_dump(mode="json"))
            self.bus.emit("debate.round", {"claim_id": claim.id, "round": round_, "stance": crit.stance.value,
                                           "max_severity": sev}, role="system", claim_id=claim.id)
            history.append({**entry, "critic": crit.model_dump(mode="json"), "critic_grounding": cg})
            rounds.append({"stance": crit.stance.value, "max_severity": sev})
            if not is_blocking(crit.stance.value, sev) or round_ >= ctx.params.debate_rounds:
                break
            for h in surfaced.values():
                if h.chunk_id not in {x.chunk_id for x in context_hits} and len(context_hits) < MAX_DEBATE_CONTEXT:
                    context_hits.append(h)
            verdict = await analyst.revise(ctx, claim, context_hits, verdict, round_, g, crit)
            round_ += 1
        status = debate_status(rounds) if rounds else "accepted"
        last_crit = history[-1].get("critic") or {}
        result = {**claim.model_dump(mode="json"), "status": status, "verdict": verdict.verdict.value,
                  "verdict_ru": VERDICT_RU[verdict.verdict.value], "confidence": verdict.confidence,
                  "correction": verdict.correction, "critic_suggested": last_crit.get("suggested_verdict"),
                  "critic_stance": last_crit.get("stance"), "history": history}
        self.bus.emit("claim.final", {"claim_id": claim.id, "status": status, "verdict": verdict.verdict.value,
                                      "correction": verdict.correction,
                                      "history": [{"round": h["round"], "analyst_verdict": h["analyst"]["verdict"],
                                                   "critic_stance": (h.get("critic") or {}).get("stance"),
                                                   "suggested_verdict": (h.get("critic") or {}).get(
                                                       "suggested_verdict"),
                                                   "issues": len((h.get("critic") or {}).get("issues") or [])}
                                                  for h in history]},
                      role="system", claim_id=claim.id)
        return result

    async def analyze(self, claims: list[Claim]) -> list[dict[str, Any]]:
        self.stage("analyze")
        sem = asyncio.Semaphore(self.ctx.params.claim_concurrency)

        async def one(c: Claim) -> dict[str, Any] | None:
            async with sem:
                try:
                    return await self.debate(c)
                except Exception as e:  # noqa: BLE001 - one bad claim must not kill the run
                    log.exception("claim %s failed", c.id)
                    self.claim_errors.append({"claim_id": c.id, "error": f"{type(e).__name__}: {e}"[:500]})
                    self.bus.emit("error", {"message": f"{c.id}: {type(e).__name__}: {e}"[:1000],
                                            "where": "analyze"}, role="system", claim_id=c.id)
                    return None

        results = [r for r in await asyncio.gather(*(one(c) for c in claims)) if r]
        self.bus.emit("stage.finished", {"stage": "analyze", "detail": {"claims": len(results),
                                                                        "failed": len(self.claim_errors)}})
        return results

    # -- stage 4 -------------------------------------------------------------------

    async def report(self, claims: list[dict[str, Any]], run_dir: Path, write_summary: bool = True) -> dict[str, Any]:
        ctx = self.ctx
        self.stage("report")
        agg = aggregate(claims, self.grounding_checks)
        summary_text = None
        if claims and write_summary:
            brief = [{"id": c["id"], "claim": c["text"], "type": c["type"], "status": c["status"],
                      "verdict": c["verdict"], "correction": c.get("correction"),
                      "critic_suggested": c.get("critic_suggested")} for c in claims]
            agg_for_llm = {k: v for k, v in agg.items() if k in ("verdict_counts", "status_counts", "grounding",
                                                                 "needs_review")}
            system, user = load_prompt("summary", title=self.target["title"], aggregate=jsonish(agg_for_llm),
                                       claims=jsonish(brief))
            try:
                out = await ctx.llm.structured(role="analyst", model=ctx.settings.analyst_model, system=system,
                                               user=user, schema=Summary, tool_name="submit_summary",
                                               fixture_key="summary")
                summary_text = out.summary
            except Exception as e:  # noqa: BLE001
                self.bus.emit("error", {"message": f"summary failed: {e}"[:1000], "where": "report"}, role="system")
        metrics = self.bus.metrics_snapshot()
        pages = max(self.target.get("pages") or 0, 0.1)
        report = {
            "run_id": ctx.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target": self.target,
            "corpus": self.corpus,
            "params": vars(ctx.params),
            "models": ctx.settings.models(),
            "aggregate": agg,
            "grounding_rate": agg["grounding"]["rate"],
            "summary": summary_text,
            "claims": claims,
            "claim_errors": self.claim_errors,
            "grounding_checks": self.grounding_checks,
            "metrics": metrics,
            "business": {
                "cost_usd": metrics["cost_usd"],
                "cost_per_page_usd": round(metrics["cost_usd"] / pages, 4),
                "elapsed_s": metrics["elapsed_s"],
                "claims_total": len(claims),
                "needs_review": agg["needs_review"]["count"],
                "needs_review_share": agg["needs_review"]["share"],
            },
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
        try:
            vmap = await asyncio.to_thread(ctx.store.vector_map, ctx.scopes)
            (run_dir / "vector_map.json").write_text(json.dumps(vmap, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("vector map snapshot failed: %s", e)
        self.bus.emit("stage.finished", {"stage": "report", "detail": {
            "needs_review": agg["needs_review"]["count"], "verdict_counts": agg["verdict_counts"]}})
        return report


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    agg = report["aggregate"]
    return {"title": report["target"].get("title"), "claims": len(report["claims"]),
            "verdict_counts": agg["verdict_counts"], "status_counts": agg["status_counts"],
            "needs_review": agg["needs_review"]["count"], "grounding_rate": report["grounding_rate"],
            "claim_errors": len(report["claim_errors"]), "cost_usd": report["metrics"]["cost_usd"],
            "elapsed_s": report["metrics"]["elapsed_s"]}


async def run_pipeline(*, run_id: str, settings: Settings, store: QdrantStore, bus: EventBus, doc: ParsedDoc,
                       filename: str, params: RunParams | None = None, llm: LLM | None = None,
                       corpus_dir: Path | None = None, callback_url: str | None = None,
                       write_summary: bool = True, claims_override: list[Claim] | None = None
                       ) -> dict[str, Any] | None:
    """`claims_override` skips the Scout (the eval reuses the same claims to compare analyst models fairly)."""
    params = params or RunParams()
    run_dir = settings.runs_dir / run_id
    ctx = RunCtx(run_id=run_id, settings=settings, bus=bus, llm=llm or LLM(settings, bus=bus), store=store,
                 params=params)
    if ctx.llm.bus is None:
        ctx.llm.bus = bus
    pipe = Pipeline(ctx)
    report = None
    try:
        await pipe.ingest(doc, filename, corpus_dir or settings.corpus_dir)
        if claims_override is not None:
            pipe.stage("scout")
            claims = [c.model_copy() for c in claims_override]
            bus.emit("scout.claims", {"claims": [c.model_dump(mode="json") for c in claims], "reused": True},
                     role="scout")
            bus.emit("stage.finished", {"stage": "scout", "detail": {"claims": len(claims), "reused": True}})
        else:
            claims = await pipe.run_scout(doc)
        results = await pipe.analyze(claims)
        report = await pipe.report(results, run_dir, write_summary=write_summary)
        bus.emit("run.finished", {"summary": report_summary(report)}, role="system")
    except Exception as e:  # noqa: BLE001
        log.exception("run %s failed", run_id)
        bus.emit("error", {"message": f"{type(e).__name__}: {e}"[:2000], "where": "pipeline", "fatal": True},
                 role="system")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        bus.close()
        try:
            await asyncio.to_thread(store.delete_scope, run_id)  # the document is not kept in the index
        except Exception as e:  # noqa: BLE001
            log.warning("could not drop document points of %s: %s", run_id, e)
    if callback_url and report is not None:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                await http.post(callback_url, json=report)
        except httpx.HTTPError as e:
            log.warning("callback to %s failed: %s", callback_url, e)
    return report
