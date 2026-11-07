import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, type Report } from "../lib/api";
import { fmtElapsed, fmtUsd } from "../lib/format";
import { StatusBadge, VerdictBadge } from "./Badges";
import { CountUp } from "./CountUp";

function Kpi({ label, children, sub }: { label: string; children: React.ReactNode; sub?: string }) {
  return (
    <div className="panel px-4 py-3">
      <div className="panel-title">{label}</div>
      <div className="mt-1 font-mono text-2xl text-ink">{children}</div>
      {sub && <div className="mt-0.5 font-mono text-[10.5px] text-ink-3">{sub}</div>}
    </div>
  );
}

export function ReportView({ runId, onClose }: { runId: string; onClose: () => void }) {
  const [report, setReport] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.report(runId).then(setReport).catch((e) => setErr(String(e.message ?? e)));
  }, [runId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const exportMd = async () => {
    try {
      const md = await api.reportMarkdown(runId);
      const url = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `triad-${runId}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(`export failed: ${String(e)}`);
    }
  };

  const agg = report?.aggregate;
  const issues = agg ? Object.entries(agg.critic_issues.by_type).sort((a, b) => b[1] - a[1]) : [];
  const maxIssue = Math.max(1, ...issues.map(([, v]) => v));

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 backdrop-blur-sm" onClick={onClose}>
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="panel w-full max-w-5xl p-5"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Review report"
      >
        <div className="mb-4 flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="panel-title">Review report · {runId}</div>
            <h2 className="mt-1 text-lg font-medium text-ink">{(report?.target.title as string) ?? "…"}</h2>
          </div>
          <button onClick={exportMd} disabled={!report} className="rounded-md border border-cyan/60 px-3 py-1.5 font-mono text-[11px] uppercase text-cyan hover:bg-cyan/10 disabled:opacity-40">
            Export .md
          </button>
          <button onClick={onClose} className="rounded-md border border-line px-3 py-1.5 font-mono text-[11px] text-ink-2 hover:text-ink" aria-label="Close report">
            ✕
          </button>
        </div>

        {err && <div className="mb-3 rounded border border-critical/60 px-3 py-2 font-mono text-[12px] text-red">{err}</div>}
        {!report && !err && <div className="py-10 text-center font-mono text-[12px] text-ink-3">loading report…</div>}

        {report && agg && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Kpi label="review score" sub="0–10, formula below">
                <span className="text-amber">
                  <CountUp value={agg.review_score ?? 0} format={(n) => n.toFixed(1)} />
                </span>
                <span className="text-sm text-ink-3"> /10</span>
              </Kpi>
              <Kpi label="grounding rate" sub={`${agg.grounding.ok}/${agg.grounding.checks} quotes verbatim`}>
                <CountUp value={(agg.grounding.rate ?? 0) * 100} format={(n) => `${n.toFixed(1)}%`} />
              </Kpi>
              <Kpi label="claims" sub={`${agg.status_counts.accepted ?? 0} accepted · ${agg.status_counts.revised ?? 0} revised · ${agg.status_counts.disputed ?? 0} disputed`}>
                {report.claims.length}
              </Kpi>
              <Kpi label="cost · time" sub={`${report.metrics.llm_calls} LLM calls`}>
                {fmtUsd(report.metrics.cost_usd)} <span className="text-sm text-ink-3">· {fmtElapsed(report.metrics.elapsed_s)}</span>
              </Kpi>
            </div>

            <section className="panel p-4">
              <div className="panel-title mb-2">Meta-review</div>
              <p className="text-[13.5px] leading-relaxed text-ink-2">{report.meta_review ?? "not available"}</p>
            </section>

            <section className="panel overflow-x-auto p-4">
              <div className="panel-title mb-2">Claims</div>
              <table className="w-full text-left text-[12.5px]">
                <thead className="font-mono text-[10px] uppercase tracking-wider text-ink-3">
                  <tr>
                    <th className="py-1 pr-3">#</th>
                    <th className="py-1 pr-3">claim</th>
                    <th className="py-1 pr-3">verdict</th>
                    <th className="py-1 pr-3">status</th>
                    <th className="py-1 pr-3">critic</th>
                    <th className="py-1 text-right">score</th>
                  </tr>
                </thead>
                <tbody>
                  {report.claims.map((c) => (
                    <tr key={c.id} className="border-t border-line/60 align-top">
                      <td className="py-2 pr-3 font-mono text-cyan">{c.id}</td>
                      <td className="py-2 pr-3 text-ink-2">{c.text}</td>
                      <td className="py-2 pr-3"><VerdictBadge verdict={c.verdict} /></td>
                      <td className="py-2 pr-3"><StatusBadge status={c.status} /></td>
                      <td className="py-2 pr-3 font-mono text-[11px] text-ink-3">
                        {c.critic_stance}
                        {c.critic_suggested && c.critic_suggested !== c.verdict ? ` → ${c.critic_suggested}` : ""}
                      </td>
                      <td className="py-2 text-right font-mono text-ink">{agg.claim_scores[c.id]?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {report.claim_errors.length > 0 && (
                <div className="mt-2 font-mono text-[11px] text-red">
                  {report.claim_errors.map((e) => `${e.claim_id}: ${e.error}`).join(" · ")}
                </div>
              )}
            </section>

            <div className="grid gap-3 md:grid-cols-2">
              <section className="panel p-4">
                <div className="panel-title mb-2">What the critic caught · {agg.critic_issues.total}</div>
                {issues.length ? (
                  <ul className="space-y-1.5">
                    {issues.map(([k, v]) => (
                      <li key={k} className="flex items-center gap-2 font-mono text-[11.5px]">
                        <span className="w-44 shrink-0 text-ink-2">{k.replace(/_/g, " ")}</span>
                        <span className="h-2.5 rounded-r" style={{ width: `${(v / maxIssue) * 60}%`, background: "#f87171" }} />
                        <span className="text-ink">{v}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="font-mono text-[11.5px] text-ink-3">no issues raised</div>
                )}
                <div className="mt-2 font-mono text-[10.5px] text-ink-3">
                  severity: {Object.entries(agg.critic_issues.by_severity).map(([k, v]) => `${k} ${v}`).join(" · ") || "—"}
                </div>
              </section>
              <section className="panel p-4">
                <div className="panel-title mb-2">Grounding by role</div>
                <ul className="space-y-1 font-mono text-[11.5px]">
                  {Object.entries(agg.grounding.by_role).map(([role, g]) => (
                    <li key={role} className="flex justify-between text-ink-2">
                      <span>{role}</span>
                      <span>
                        {g.ok}/{g.checks} <span className="text-ink-3">({((g.ok / Math.max(1, g.checks)) * 100).toFixed(0)}%)</span>
                      </span>
                    </li>
                  ))}
                </ul>
                <div className="panel-title mb-1 mt-4">Score formula</div>
                <p className="font-mono text-[10.5px] leading-relaxed text-ink-3">{agg.score_formula}</p>
              </section>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
