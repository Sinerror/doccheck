import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, type Report } from "../lib/api";
import { fmtElapsed, fmtUsd } from "../lib/format";
import { CLAIM_TYPE_RU } from "../lib/colors";
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
      a.download = `doccheck-${runId}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(`экспорт не удался: ${String(e)}`);
    }
  };

  const agg = report?.aggregate;
  const review = agg?.needs_review.claims ?? {};
  const issues = agg ? Object.entries(agg.critic_issues.by_type).sort((a, b) => b[1] - a[1]) : [];
  const maxIssue = Math.max(1, ...issues.map(([, v]) => v));
  const fixes = report?.claims.filter((c) => ["CONTRADICTED", "OVERSTATED", "PARTIALLY_SUPPORTED", "UNVERIFIABLE"].includes(c.verdict)) ?? [];

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 backdrop-blur-sm" onClick={onClose}>
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="panel w-full max-w-5xl p-5"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Отчёт о проверке"
      >
        <div className="mb-4 flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="panel-title">Отчёт о проверке · {runId}</div>
            <h2 className="mt-1 text-lg font-medium text-ink">{report?.target.title ?? "…"}</h2>
            {report?.document_context && <div className="mt-0.5 text-[12px] text-ink-3">{report.document_context}</div>}
          </div>
          <button onClick={exportMd} disabled={!report} className="rounded-md border border-cyan/60 px-3 py-1.5 font-mono text-[11px] uppercase text-cyan hover:bg-cyan/10 disabled:opacity-40">
            Скачать .md
          </button>
          <button onClick={onClose} className="rounded-md border border-line px-3 py-1.5 font-mono text-[11px] text-ink-2 hover:text-ink" aria-label="Закрыть">
            ✕
          </button>
        </div>

        {err && <div className="mb-3 rounded border border-critical/60 px-3 py-2 font-mono text-[12px] text-red">{err}</div>}
        {!report && !err && <div className="py-10 text-center font-mono text-[12px] text-ink-3">загрузка отчёта…</div>}

        {report && agg && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Kpi label="человеку на проверку" sub={`из ${report.claims.length} утверждений`}>
                <span className="text-amber">
                  <CountUp value={agg.needs_review.count} format={(n) => n.toFixed(0)} />
                </span>
                <span className="text-sm text-ink-3"> / {report.claims.length}</span>
              </Kpi>
              <Kpi label="цитаты проверены кодом" sub={`${agg.grounding.ok}/${agg.grounding.checks} дословно`}>
                <CountUp value={(agg.grounding.rate ?? 0) * 100} format={(n) => `${n.toFixed(0)}%`} />
              </Kpi>
              <Kpi label="спор аналитика и критика" sub={`${agg.status_counts.accepted ?? 0} согласен · ${agg.status_counts.revised ?? 0} пересмотрено`}>
                {agg.status_counts.disputed ?? 0} <span className="text-sm text-ink-3">спорных</span>
              </Kpi>
              <Kpi label="стоимость · время" sub={`${report.metrics.llm_calls} вызовов LLM · ${fmtUsd(report.business.cost_per_page_usd)}/стр.`}>
                {fmtUsd(report.metrics.cost_usd)} <span className="text-sm text-ink-3">· {fmtElapsed(report.metrics.elapsed_s)}</span>
              </Kpi>
            </div>

            <section className="panel p-4">
              <div className="panel-title mb-2">Итог</div>
              <p className="text-[13.5px] leading-relaxed text-ink-2">{report.summary ?? "резюме недоступно"}</p>
            </section>

            {fixes.length > 0 && (
              <section className="panel p-4">
                <div className="panel-title mb-2">Что исправить или уточнить</div>
                <ul className="space-y-2">
                  {fixes.map((c) => (
                    <li key={c.id} className="rounded-md border border-line/70 px-3 py-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-[12px] text-cyan">{c.id}</span>
                        <VerdictBadge verdict={c.verdict} />
                        {c.status === "disputed" && <StatusBadge status="disputed" />}
                      </div>
                      <div className="mt-1 text-[13px] text-ink">«{c.quote || c.text}»</div>
                      {c.correction && <div className="mt-1 text-[12.5px] text-green-400">→ по источникам: {c.correction}</div>}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <section className="panel overflow-x-auto p-4">
              <div className="panel-title mb-2">Все утверждения</div>
              <table className="w-full text-left text-[12.5px]">
                <thead className="font-mono text-[10px] uppercase tracking-wider text-ink-3">
                  <tr>
                    <th className="py-1 pr-3">#</th>
                    <th className="py-1 pr-3">утверждение</th>
                    <th className="py-1 pr-3">вердикт</th>
                    <th className="py-1 pr-3">спор</th>
                    <th className="py-1">человеку</th>
                  </tr>
                </thead>
                <tbody>
                  {report.claims.map((c) => (
                    <tr key={c.id} className="border-t border-line/60 align-top">
                      <td className="py-2 pr-3 font-mono text-cyan">{c.id}</td>
                      <td className="py-2 pr-3 text-ink-2">
                        {c.text}
                        <div className="font-mono text-[10px] text-ink-3">{CLAIM_TYPE_RU[c.type] ?? c.type} · уверенность {c.confidence.toFixed(2)}</div>
                      </td>
                      <td className="py-2 pr-3"><VerdictBadge verdict={c.verdict} /></td>
                      <td className="py-2 pr-3"><StatusBadge status={c.status} /></td>
                      <td className="py-2 font-mono text-[11px] text-amber">{review[c.id]?.join("; ") ?? ""}</td>
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
                <div className="panel-title mb-2">Что нашёл критик · {agg.critic_issues.total}</div>
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
                  <div className="font-mono text-[11.5px] text-ink-3">замечаний нет</div>
                )}
              </section>
              <section className="panel p-4">
                <div className="panel-title mb-2">Источники и модели</div>
                <div className="font-mono text-[11px] leading-relaxed text-ink-3">
                  {report.corpus.docs.length} документов · {report.corpus.chunks} фрагментов · {report.corpus.embed_model}
                  <br />
                  {Object.entries(report.models).map(([k, v]) => `${k}: ${v}`).join(" · ")}
                </div>
                <div className="panel-title mb-1 mt-4">Правило отбора для человека</div>
                <p className="font-mono text-[10.5px] leading-relaxed text-ink-3">{agg.needs_review.rule}</p>
              </section>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
