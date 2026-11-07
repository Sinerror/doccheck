"""Deterministic async pipeline: ingest → scout → index → analyze (analyst ⇄ critic) → report.

Plain asyncio, no agent framework. Every step is an explicit function call and every
observable effect is an event on the bus.
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
from .models import AnalystVerdict, Claim, MetaReview
from .rag.chunking import chunk_document
from .rag.qdrant_store import Hit, QdrantStore
from .report import aggregate, debate_status, is_blocking, max_severity, render_markdown
from .roles import analyst, critic, scout
from .sources import arxiv
from .sources.pdf import ParsedPDF, abstract_like, parse_pdf

log = logging.getLogger(__name__)

MAX_DEBATE_CONTEXT = 24


class Pipeline:
    def __init__(self, ctx: RunCtx):
        self.ctx = ctx
        self.grounding_checks: list[dict[str, Any]] = []
        self.papers: list[dict[str, Any]] = []
        self.target: dict[str, Any] = {}
        self.claim_errors: list[dict[str, str]] = []

    @property
    def bus(self) -> EventBus:
        return self.ctx.bus

    async def stage(self, name: str):
        self.bus.emit("stage.started", {"stage": name})

    # -- stage 0 -------------------------------------------------------------------

    async def ingest(self, arxiv_id: str | None, pdf_bytes: bytes | None, filename: str | None,
                     parsed_override: ParsedPDF | None = None) -> Any:
        ctx = self.ctx
        if arxiv_id:
            meta = await arxiv.fetch_metadata(ctx.http, arxiv_id) or {}
            self.target = {"paper_id": f"arxiv:{arxiv_id}", "arxiv_id": arxiv_id,
                           "title": meta.get("title") or arxiv_id, "authors": meta.get("authors") or [],
                           "year": meta.get("year"), "venue": "arXiv", "url": f"https://arxiv.org/abs/{arxiv_id}",
                           "abstract": meta.get("abstract") or "", "doi": None}
        else:
            digest = hashlib.sha1(pdf_bytes or b"").hexdigest()[:12]
            self.target = {"paper_id": f"upload:{digest}", "arxiv_id": None, "title": filename or "uploaded.pdf",
                           "authors": [], "year": None, "venue": None, "url": None, "abstract": "", "doi": None}
        self.bus.emit("run.started", {
            "target": {k: v for k, v in self.target.items() if k != "abstract"},
            "params": vars(ctx.params),
            "models": {"scout": ctx.settings.scout_model, "analyst": ctx.settings.analyst_model,
                       "critic": ctx.settings.critic_model},
        }, role="system")
        await self.stage("ingest")
        if parsed_override is not None:  # e.g. scripts/eval_planted.py feeds a deliberately altered text
            parsed = parsed_override
            self.target["abstract"] = abstract_like(parsed)
        else:
            if pdf_bytes is None:
                pdf_bytes = await arxiv.download_pdf(ctx.http, arxiv_id)
            parsed = await asyncio.to_thread(parse_pdf, pdf_bytes)
        if not arxiv_id and parsed.title:
            self.target["title"] = parsed.title
        if not self.target["abstract"]:
            self.target["abstract"] = abstract_like(parsed)
        ctx.title = self.target["title"]
        ctx.target_paper_id = self.target["paper_id"]
        chunks = chunk_document(parsed.sections)
        n = await asyncio.to_thread(ctx.store.upsert_chunks, ctx.run_id, self.target, chunks, "target")
        total = await asyncio.to_thread(ctx.store.count, ctx.run_id)
        self.bus.emit("rag.indexed", {"paper_id": self.target["paper_id"], "chunks": n, "total_points": total,
                                      "source": "target", "title": self.target["title"]}, role="system")
        self.bus.emit("stage.finished", {"stage": "ingest", "detail": {
            "title": self.target["title"], "pages": parsed.page_count, "sections": parsed.headings, "chunks": n}})
        return parsed

    # -- stage 1 -------------------------------------------------------------------

    async def run_scout(self, parsed) -> list[Claim]:
        ctx = self.ctx
        await self.stage("scout")
        claims = await scout.extract_claims(ctx, parsed, self.target["title"], self.target["abstract"])
        candidates = await scout.search_literature(ctx, claims, self.target)
        self.papers = await scout.rank_papers(ctx, claims, candidates)
        await scout.fetch_fulltexts(ctx, self.papers)
        self.bus.emit("stage.finished", {"stage": "scout", "detail": {
            "claims": len(claims), "candidates": len(candidates), "selected": len(self.papers),
            "fulltext": sum(1 for p in self.papers if p.get("parsed"))}})
        return claims

    # -- stage: index ----------------------------------------------------------------

    async def index(self) -> None:
        ctx = self.ctx
        await self.stage("index")
        for p in self.papers:
            parsed = p.get("parsed")
            if parsed is not None:
                sections = parsed.sections
            elif p.get("abstract"):
                sections = [("Abstract", p["abstract"])]
            else:
                sections = [("Title", p.get("title") or "")]
            chunks = chunk_document(sections)
            n = await asyncio.to_thread(ctx.store.upsert_chunks, ctx.run_id, p, chunks, "related")
            p["chunks"] = n
            p["fulltext"] = parsed is not None
            total = await asyncio.to_thread(ctx.store.count, ctx.run_id)
            self.bus.emit("rag.indexed", {"paper_id": p["paper_id"], "chunks": n, "total_points": total,
                                          "source": "related", "title": p.get("title")}, role="system")
        await asyncio.to_thread(ctx.store.basis, ctx.run_id)  # fit + cache the PCA basis for query projection
        total = await asyncio.to_thread(ctx.store.count, ctx.run_id)
        self.bus.emit("stage.finished", {"stage": "index", "detail": {"papers": len(self.papers) + 1,
                                                                      "points": total}})

    # -- stage 2/3: debate -------------------------------------------------------------

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
            crit, surfaced = await critic.critique(ctx, claim, verdict, g, context_hits, round_)
            pool.update(surfaced)
            cg = await self._ground(claim.id, [(ce.chunk_id, ce.quote) for i in crit.issues
                                               for ce in i.counter_evidence], pool, "critic", round_)
            sev = max_severity(crit.model_dump(mode="json"))
            self.bus.emit("debate.round", {"claim_id": claim.id, "round": round_, "stance": crit.stance.value,
                                           "max_severity": sev}, role="system", claim_id=claim.id)
            history.append({"round": round_, "analyst": verdict.model_dump(mode="json"), "grounding": g,
                            "critic": crit.model_dump(mode="json"), "critic_grounding": cg,
                            "context_chunk_ids": [h.chunk_id for h in context_hits]})
            rounds.append({"stance": crit.stance.value, "max_severity": sev})
            if not is_blocking(crit.stance.value, sev) or round_ >= ctx.params.debate_rounds:
                break
            for h in surfaced.values():
                if h.chunk_id not in {x.chunk_id for x in context_hits} and len(context_hits) < MAX_DEBATE_CONTEXT:
                    context_hits.append(h)
            verdict = await analyst.revise(ctx, claim, context_hits, verdict, round_, g, crit)
            round_ += 1
        status = debate_status(rounds)
        last_crit = history[-1]["critic"]
        result = {**claim.model_dump(mode="json"), "status": status, "verdict": verdict.verdict.value,
                  "confidence": verdict.confidence, "critic_suggested": last_crit.get("suggested_verdict"),
                  "critic_stance": last_crit["stance"], "history": history}
        self.bus.emit("claim.final", {"claim_id": claim.id, "status": status, "verdict": verdict.verdict.value,
                                      "history": [{"round": h["round"], "analyst_verdict": h["analyst"]["verdict"],
                                                   "critic_stance": h["critic"]["stance"],
                                                   "suggested_verdict": h["critic"].get("suggested_verdict"),
                                                   "issues": len(h["critic"]["issues"])} for h in history]},
                      role="system", claim_id=claim.id)
        return result

    async def analyze(self, claims: list[Claim]) -> list[dict[str, Any]]:
        await self.stage("analyze")
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

    async def report(self, claims: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
        ctx = self.ctx
        await self.stage("report")
        agg = aggregate(claims, self.grounding_checks)
        summary_claims = [{
            "id": c["id"], "claim": c["text"], "type": c["type"], "status": c["status"],
            "final_verdict": c["verdict"], "confidence": c["confidence"],
            "critic_final_stance": c["critic_stance"], "critic_suggested": c["critic_suggested"],
            "critic_issues": [f"[{i['severity']}] {i['type']}: {i['detail'][:300]}"
                              for i in c["history"][-1]["critic"]["issues"]],
            "analyst_concerns": c["history"][-1]["analyst"].get("concerns", [])[:4],
        } for c in claims]
        meta_text = None
        if claims:
            agg_for_llm = {k: v for k, v in agg.items() if k not in ("score_formula", "claim_scores")}
            system, user = load_prompt("meta_review", title=self.target["title"], aggregate=jsonish(agg_for_llm),
                                       claims=jsonish(summary_claims))
            try:
                mr = await ctx.llm.structured(role="analyst", model=ctx.settings.analyst_model, system=system,
                                              user=user, schema=MetaReview, tool_name="submit_meta_review",
                                              fixture_key="meta-review")
                meta_text = mr.meta_review
            except Exception as e:  # noqa: BLE001
                self.bus.emit("error", {"message": f"meta-review failed: {e}", "where": "report"}, role="system")
        report = {
            "run_id": ctx.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target": {k: v for k, v in self.target.items()},
            "params": vars(ctx.params),
            "models": {"scout": ctx.settings.scout_model, "analyst": ctx.settings.analyst_model,
                       "critic": ctx.settings.critic_model},
            "aggregate": agg,
            "review_score": agg["review_score"],
            "grounding_rate": agg["grounding"]["rate"],
            "meta_review": meta_text,
            "claims": claims,
            "claim_errors": self.claim_errors,
            "papers": [{k: p.get(k) for k in ("paper_id", "title", "year", "venue", "url", "authors", "relevance",
                                              "chunks", "fulltext", "found_for")} for p in self.papers],
            "grounding_checks": self.grounding_checks,
        }
        report["metrics"] = self.bus.metrics_snapshot()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
        try:
            vmap = await asyncio.to_thread(ctx.store.vector_map, ctx.run_id)
            (run_dir / "vector_map.json").write_text(json.dumps(vmap), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("vector map snapshot failed: %s", e)
        self.bus.emit("stage.finished", {"stage": "report", "detail": {"review_score": agg["review_score"]}})
        return report


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    agg = report["aggregate"]
    return {"review_score": report["review_score"], "grounding_rate": report["grounding_rate"],
            "status_counts": agg["status_counts"], "verdict_counts": agg["verdict_counts"],
            "critic_issues": agg["critic_issues"], "claims": len(report["claims"]),
            "claim_errors": len(report["claim_errors"]), "title": report["target"].get("title"),
            "cost_usd": report["metrics"]["cost_usd"]}


async def run_pipeline(*, run_id: str, settings: Settings, store: QdrantStore, bus: EventBus,
                       params: RunParams | None = None, arxiv_id: str | None = None,
                       pdf_bytes: bytes | None = None, filename: str | None = None,
                       llm: LLM | None = None, callback_url: str | None = None,
                       parsed: ParsedPDF | None = None) -> dict[str, Any] | None:
    if not arxiv_id and pdf_bytes is None and parsed is None:
        raise ValueError("need arxiv_id or pdf_bytes")
    params = params or RunParams()
    run_dir = settings.runs_dir / run_id
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": arxiv.HEADERS["User-Agent"]}) as http:
        ctx = RunCtx(run_id=run_id, settings=settings, bus=bus,
                     llm=llm or LLM(settings, bus=bus), store=store, params=params, http=http)
        if ctx.llm.bus is None:
            ctx.llm.bus = bus
        pipe = Pipeline(ctx)
        report = None
        try:
            parsed_doc = await pipe.ingest(arxiv_id, pdf_bytes, filename, parsed_override=parsed)
            claims = await pipe.run_scout(parsed_doc)
            await pipe.index()
            results = await pipe.analyze(claims)
            report = await pipe.report(results, run_dir)
            bus.emit("run.finished", {"summary": report_summary(report)}, role="system")
        except Exception as e:  # noqa: BLE001
            log.exception("run %s failed", run_id)
            bus.emit("error", {"message": f"{type(e).__name__}: {e}"[:2000], "where": "pipeline", "fatal": True},
                     role="system")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            bus.close()
        if callback_url and report is not None:
            try:
                await http.post(callback_url, json=report, timeout=30)
            except httpx.HTTPError as e:
                log.warning("callback to %s failed: %s", callback_url, e)
        return report
