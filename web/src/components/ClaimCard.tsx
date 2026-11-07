import { AnimatePresence, motion } from "framer-motion";
import type { Evidence, Issue } from "../types/events";
import { paperOf, type ClaimState, type RoundState } from "../state/runReducer";
import { SEVERITY_COLOR } from "../lib/colors";
import { RoleDot, StatusBadge, VerdictBadge } from "./Badges";
import { Typewriter } from "./Typewriter";

/** Attach a critic issue to the evidence item it is about: shared chunk id, or the quote/chunk id quoted in the detail. */
function issueTargets(issue: Issue, ev: Evidence[]): number[] {
  const hits = new Set<number>();
  ev.forEach((e, i) => {
    if (issue.detail.includes(e.chunk_id)) hits.add(i);
    const head = e.quote.slice(0, 40);
    if (head.length >= 20 && issue.detail.includes(head)) hits.add(i);
    if (issue.type === "ungrounded_quote" && issue.counter_evidence?.some((c) => c.chunk_id === e.chunk_id)) hits.add(i);
  });
  return [...hits];
}

function IssueNote({ issue }: { issue: Issue }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      className="mt-1 rounded-md border-l-2 border-red bg-red/8 px-2 py-1 text-[11.5px] leading-snug text-ink-2"
      style={{ background: "rgba(248,113,113,0.07)" }}
    >
      <span className="mr-1.5 font-mono text-[10px] uppercase tracking-wide" style={{ color: SEVERITY_COLOR[issue.severity] }}>
        ⚠ {issue.severity} · {issue.type.replace(/_/g, " ")}
      </span>
      {issue.detail}
      {issue.counter_evidence?.map((c, i) => (
        <div key={i} className="mt-0.5 font-mono text-[10.5px] text-red/90">
          ↳ {c.chunk_id}: “{c.quote}”
        </div>
      ))}
    </motion.div>
  );
}

function EvidenceRow({ e, round, issues }: { e: Evidence; round: RoundState; issues: Issue[] }) {
  const g = round.grounding.find((x) => x.chunk_id === e.chunk_id && x.quote === e.quote);
  const ok = g?.ok;
  const stanceColor = e.stance === "supports" ? "#0ca30c" : e.stance === "contradicts" ? "#d03b3b" : "#6b7685";
  return (
    <li className="rounded-md border border-line/70 bg-bg/40 px-2 py-1.5">
      <div className="flex items-start gap-2">
        <span
          className="mt-px shrink-0 font-mono text-[11px]"
          style={{ color: g === undefined ? "#6b7685" : ok ? "#0ca30c" : "#d03b3b" }}
          title={g ? `grounding partial_ratio=${g.score} (${ok ? "verbatim" : "UNGROUNDED"})` : "grounding pending"}
        >
          {g === undefined ? "…" : ok ? "✓" : "✗"}
        </span>
        <div className="min-w-0 flex-1">
          <div className={`text-[12px] leading-snug ${ok === false ? "text-red line-through decoration-red/60" : "text-ink"}`}>“{e.quote}”</div>
          <div className="mt-0.5 flex flex-wrap gap-x-2 font-mono text-[10px] text-ink-3">
            <span style={{ color: stanceColor }}>{e.stance}</span>
            <span title={paperOf(e.chunk_id)}>{e.chunk_id}</span>
            {g && <span>match {Math.round(g.score)}</span>}
          </div>
          {issues.map((iss, i) => (
            <IssueNote key={i} issue={iss} />
          ))}
        </div>
      </div>
    </li>
  );
}

export function ClaimCard({ c }: { c: ClaimState }) {
  const withAnalyst = c.rounds.filter((r) => r.analyst);
  const latest = withAnalyst[withAnalyst.length - 1];
  const verdicts = withAnalyst.map((r) => r.analyst!.verdict);
  const critique = latest?.critic;
  const evidence = latest?.analyst?.evidence ?? [];
  const attached = new Map<number, Issue[]>();
  const loose: Issue[] = [];
  for (const iss of critique?.issues ?? []) {
    const t = issueTargets(iss, evidence);
    if (t.length) t.forEach((i) => attached.set(i, [...(attached.get(i) ?? []), iss]));
    else loose.push(iss);
  }
  const working = c.phase === "analyst" || c.phase === "critic";
  const roleColor = c.phase === "critic" ? "#f87171" : "#a78bfa";
  const streamingAnalyst = c.phase === "analyst";
  const streamingCritic = c.phase === "critic";
  const glow = c.final?.status === "disputed" ? "glow-red" : working ? (c.phase === "critic" ? "glow-red" : "glow-violet") : "";

  return (
    <motion.article layout className={`panel flex min-w-0 flex-col gap-2 p-3 ${glow}`}>
      <header className="flex items-start gap-2">
        <span className="font-mono text-[12px] font-medium text-cyan">{c.id}</span>
        <div className="min-w-0 flex-1">
          <p className="text-[13px] leading-snug text-ink">{c.text}</p>
          <div className="mt-1 font-mono text-[10px] text-ink-3">
            {c.type} · {c.section}
          </div>
        </div>
        {c.final ? (
          <StatusBadge status={c.final.status} />
        ) : (
          <span className="flex items-center gap-1.5 font-mono text-[10px] text-ink-3">
            {working && <RoleDot color={roleColor} pulsing />}
            {c.phase === "waiting" ? "queued" : `${c.phase} · r${c.round}`}
          </span>
        )}
      </header>

      {verdicts.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <AnimatePresence initial={false}>
            {verdicts.map((v, i) => (
              <motion.span key={`${i}-${v}`} initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-1.5">
                {i > 0 && <span className="font-mono text-[10px] text-ink-3">→</span>}
                <VerdictBadge verdict={v} struck={i < verdicts.length - 1} size={i === verdicts.length - 1 ? "md" : "sm"} />
              </motion.span>
            ))}
          </AnimatePresence>
          {latest?.analyst && <span className="font-mono text-[10px] text-ink-3">conf {latest.analyst.confidence.toFixed(2)}</span>}
        </div>
      )}

      {/* analyst stream */}
      {(c.analystText || c.analystThinking) && (
        <section className="rounded-md bg-violet/5 px-2 py-1.5" style={{ background: "rgba(167,139,250,0.05)" }}>
          <div className="mb-0.5 flex items-center gap-1.5 font-mono text-[10px] text-violet">
            <RoleDot color="#a78bfa" pulsing={streamingAnalyst} /> ANALYST{c.analystText ? "" : " · thinking"}
          </div>
          <div className="scroll-thin max-h-36 overflow-y-auto whitespace-pre-wrap text-[12px] leading-relaxed text-ink-2">
            {c.analystText ? (
              <Typewriter text={c.analystText} active={streamingAnalyst} />
            ) : (
              <Typewriter text={c.analystThinking} active={streamingAnalyst} className="italic text-ink-3" />
            )}
          </div>
          {latest?.analyst?.response_to_critic && (
            <div className="scroll-thin mt-1 max-h-24 overflow-y-auto border-t border-line/60 pt-1 text-[11.5px] text-ink-2">
              <span className="font-mono text-[10px] text-violet">↩ to critic: </span>
              {latest.analyst.response_to_critic}
            </div>
          )}
        </section>
      )}

      {evidence.length > 0 && (
        <ul className="space-y-1">
          {evidence.map((e, i) => (
            <EvidenceRow key={`${e.chunk_id}-${i}`} e={e} round={latest!} issues={attached.get(i) ?? []} />
          ))}
        </ul>
      )}

      {/* critic stream */}
      {(streamingCritic || critique || c.criticTools.length > 0) && (
        <section className="rounded-md px-2 py-1.5" style={{ background: "rgba(248,113,113,0.06)" }}>
          <div className="mb-0.5 flex items-center gap-1.5 font-mono text-[10px] text-red">
            <RoleDot color="#f87171" pulsing={streamingCritic} /> CRITIC
            {critique && (
              <span className="ml-1 rounded border px-1 text-[9.5px]" style={{ borderColor: critique.stance === "AGREE" ? "#0ca30c" : "#f87171" }}>
                {critique.stance}
                {critique.suggested_verdict ? ` → ${critique.suggested_verdict}` : ""}
              </span>
            )}
          </div>
          {c.criticTools.length > 0 && (
            <ul className="mb-1 space-y-px font-mono text-[10px] text-ink-3">
              {c.criticTools.slice(-4).map((t) => (
                <li key={t.seq} className="truncate" title={t.summary}>
                  ⌕ {t.summary}
                </li>
              ))}
            </ul>
          )}
          {c.criticText && (
            <div className="scroll-thin max-h-28 overflow-y-auto whitespace-pre-wrap text-[12px] leading-relaxed text-ink-2">
              <Typewriter text={c.criticText} active={streamingCritic} />
            </div>
          )}
          {loose.map((iss, i) => (
            <IssueNote key={i} issue={iss} />
          ))}
        </section>
      )}

      {c.rounds
        .filter((r) => r !== latest && r.critic && r.critic.issues.length > 0)
        .map((r) => (
          <details key={`hist-${r.round}`} className="rounded-md border border-line/60 px-2 py-1">
            <summary className="cursor-pointer font-mono text-[10px] text-red/90">
              r{r.round} critic {r.critic!.stance.toLowerCase()} · {r.critic!.issues.length} issue{r.critic!.issues.length > 1 ? "s" : ""} on{" "}
              {r.analyst?.verdict ?? "?"}
              {r.critic!.suggested_verdict ? ` → suggested ${r.critic!.suggested_verdict}` : ""}
            </summary>
            {r.critic!.issues.map((iss, i) => (
              <IssueNote key={i} issue={iss} />
            ))}
          </details>
        ))}

      {c.rounds.length > 1 && (
        <footer className="flex flex-wrap gap-x-3 gap-y-1 border-t border-line/60 pt-1.5 font-mono text-[10px] text-ink-3">
          {c.rounds.map((r) => (
            <span key={r.round}>
              r{r.round}: {r.analyst?.verdict ?? "…"}
              {r.stance && (
                <span style={{ color: r.stance === "AGREE" ? "#0ca30c" : "#f87171" }}>
                  {" "}
                  · {r.stance.toLowerCase()}
                  {r.maxSeverity ? `(${r.maxSeverity})` : ""}
                </span>
              )}
            </span>
          ))}
        </footer>
      )}
      {c.final?.status === "disputed" && (
        <div className="rounded-md border border-critical/60 px-2 py-1 text-[11.5px] text-ink-2">
          ⚑ Disputed: analyst holds <b className="text-ink">{c.final.verdict}</b>, critic suggests{" "}
          <b className="text-ink">{critique?.suggested_verdict ?? "a different verdict"}</b>. Both positions shown.
        </div>
      )}
      {c.error && <div className="rounded-md border border-critical/60 px-2 py-1 font-mono text-[11px] text-red">error: {c.error}</div>}
    </motion.article>
  );
}

export function ClaimLanes({ claims }: { claims: ClaimState[] }) {
  if (!claims.length)
    return (
      <div className="panel flex h-40 items-center justify-center font-mono text-[12px] text-ink-3">
        waiting for scout.claims…
      </div>
    );
  return (
    <div className="grid grid-cols-1 gap-3 2xl:grid-cols-2">
      {claims.map((c) => (
        <ClaimCard key={c.id} c={c} />
      ))}
    </div>
  );
}
