"""The Analyst ⇄ Critic debate of one claim as a LangGraph StateGraph.

Same roles, prompts, grounding check and events as the hand-written loop in orchestrator.Pipeline.debate;
only the control flow moves into a graph:

    analyze → ground → critic ─┬─ blocking challenge and rounds left → revise → ground → …
                               └─ otherwise → END
    (with use_critic=False: analyze → ground → END)

Enable with DOCCHECK_ORCHESTRATOR=langgraph. Compared in README ("LangGraph vs own orchestrator").
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

from .models import AnalystVerdict, Claim, Critique
from .report import is_blocking, max_severity
from .roles import analyst, critic

if TYPE_CHECKING:
    from .orchestrator import Pipeline

MAX_DEBATE_CONTEXT = 20


class DebateState(TypedDict, total=False):
    claim: Claim
    pool: dict[str, Any]
    context_hits: list[Any]
    verdict: AnalystVerdict
    round: int
    grounding: list[dict[str, Any]]
    critique: Critique
    surfaced: dict[str, Any]
    history: list[dict[str, Any]]
    rounds: list[dict[str, Any]]


def build_debate_graph(pipe: "Pipeline"):
    ctx = pipe.ctx

    async def analyze_node(s: DebateState) -> DebateState:
        hits = await analyst.gather_context(ctx, s["claim"])
        v = await analyst.analyze(ctx, s["claim"], hits)
        return {"pool": {h.chunk_id: h for h in hits}, "context_hits": list(hits), "verdict": v, "round": 0,
                "history": [], "rounds": []}

    async def ground_node(s: DebateState) -> DebateState:
        g = await pipe._ground(s["claim"].id, [(e.chunk_id, e.quote) for e in s["verdict"].evidence], s["pool"],
                               "analyst", s["round"])
        entry = {"round": s["round"], "analyst": s["verdict"].model_dump(mode="json"), "grounding": g,
                 "context_chunk_ids": [h.chunk_id for h in s["context_hits"]]}
        return {"grounding": g, "history": s["history"] + [entry]}

    async def critic_node(s: DebateState) -> DebateState:
        claim, round_ = s["claim"], s["round"]
        crit, surfaced = await critic.critique(ctx, claim, s["verdict"], s["grounding"], s["context_hits"], round_)
        pool = {**s["pool"], **surfaced}
        cg = await pipe._ground(claim.id, [(ce.chunk_id, ce.quote) for i in crit.issues for ce in i.counter_evidence],
                                pool, "critic", round_)
        sev = max_severity(crit.model_dump(mode="json"))
        pipe.bus.emit("debate.round", {"claim_id": claim.id, "round": round_, "stance": crit.stance.value,
                                       "max_severity": sev}, role="system", claim_id=claim.id)
        history = list(s["history"])
        history[-1] = {**history[-1], "critic": crit.model_dump(mode="json"), "critic_grounding": cg}
        return {"critique": crit, "surfaced": surfaced, "pool": pool, "history": history,
                "rounds": s["rounds"] + [{"stance": crit.stance.value, "max_severity": sev}]}

    async def revise_node(s: DebateState) -> DebateState:
        hits = list(s["context_hits"])
        for h in s["surfaced"].values():
            if h.chunk_id not in {x.chunk_id for x in hits} and len(hits) < MAX_DEBATE_CONTEXT:
                hits.append(h)
        v = await analyst.revise(ctx, s["claim"], hits, s["verdict"], s["round"], s["grounding"], s["critique"])
        return {"verdict": v, "context_hits": hits, "round": s["round"] + 1}

    def after_ground(s: DebateState) -> str:
        return "critic" if ctx.params.use_critic else END

    def after_critic(s: DebateState) -> str:
        last = s["rounds"][-1]
        if not is_blocking(last["stance"], last["max_severity"]) or s["round"] >= ctx.params.debate_rounds:
            return END
        return "revise"

    g = StateGraph(DebateState)
    g.add_node("analyze", analyze_node)
    g.add_node("ground", ground_node)
    g.add_node("critic", critic_node)
    g.add_node("revise", revise_node)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "ground")
    g.add_conditional_edges("ground", after_ground, {"critic": "critic", END: END})
    g.add_conditional_edges("critic", after_critic, {"revise": "revise", END: END})
    g.add_edge("revise", "ground")
    return g.compile()


async def run_debate_graph(pipe: "Pipeline", claim: Claim) -> tuple[AnalystVerdict, list[dict], list[dict]]:
    graph = build_debate_graph(pipe)
    out = await graph.ainvoke({"claim": claim}, {"recursion_limit": 50})
    return out["verdict"], out["history"], out["rounds"]
