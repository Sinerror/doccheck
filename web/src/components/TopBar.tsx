import type { RunState } from "../state/runReducer";
import type { StreamInfo, StreamMode } from "../state/useRunStream";
import { fmtCompact, fmtElapsed, fmtInt, fmtUsd } from "../lib/format";
import { CountUp } from "./CountUp";

function Stat({ label, value, format, accent, title }: { label: string; value: number; format: (n: number) => string; accent?: string; title?: string }) {
  return (
    <div className="flex min-w-[78px] flex-col" title={title}>
      <span className="panel-title !text-[9.5px]">{label}</span>
      <span className="font-mono text-[17px] leading-tight" style={{ color: accent ?? "var(--color-ink)" }}>
        <CountUp value={value} format={format} />
      </span>
    </div>
  );
}

export function TopBar({ state, mode, info, onHome, onShowReport, canReport }: {
  state: RunState;
  mode: StreamMode;
  info: StreamInfo;
  onHome: () => void;
  onShowReport: () => void;
  canReport: boolean;
}) {
  const m = state.metrics;
  // elapsed comes from the backend's own clock (metrics.elapsed_s) or event timestamps, never a local timer
  const elapsed = Math.max(m?.elapsed_s ?? 0, state.firstTs && state.lastTs ? state.lastTs - state.firstTs : 0);
  const title = (state.target?.title as string) ?? state.runId;
  const live = mode === "live";
  const badgeColor = live ? "#f87171" : "#a78bfa";
  const statusText =
    state.status === "finished" ? "готово" : state.status === "failed" ? "ошибка" : info.error ? info.error : info.connected ? "поток событий" : info.ended ? "завершено" : "подключение…";

  return (
    <header className="panel flex flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3">
      <button onClick={onHome} className="font-mono text-sm tracking-[0.3em] text-cyan hover:text-ink" title="К списку прогонов">
        DOCCHECK
      </button>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span
            className="inline-flex items-center gap-1.5 rounded border px-1.5 py-px font-mono text-[10px] font-medium tracking-widest"
            style={{ borderColor: badgeColor, color: badgeColor }}
          >
            {(live && state.status === "running") || (!live && info.connected) ? (
              <span className="h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: badgeColor }} />
            ) : null}
            {live ? "LIVE" : "REPLAY"}
          </span>
          <span className="font-mono text-[10px] text-ink-3">{state.runId}</span>
          <span className="font-mono text-[10px] text-ink-3">· {statusText}</span>
        </div>
        <h1 className="truncate text-[15px] font-medium text-ink" title={title}>
          {title}
        </h1>
      </div>
      <div className="flex flex-wrap items-end gap-x-5 gap-y-2">
        <Stat label="источники" value={state.paperOrder.length} format={fmtInt} />
        <Stat label="фрагменты" value={state.totalPoints} format={fmtInt} accent="#22d3ee" title="Точки в Qdrant: корпус + документ на проверке" />
        <Stat label="утверждения" value={state.claimOrder.length} format={fmtInt} />
        <Stat label="поиски" value={state.retrievalCount} format={fmtInt} title="Гибридные запросы к Qdrant (аналитик + инструменты критика)" />
        <Stat
          label="агенты"
          value={Object.keys(state.agents).length}
          format={fmtInt}
          accent="#a78bfa"
          title={`Agent instances spawned (agent.started): ${Object.values(state.agents)
            .map((a) => `${a.role}${a.claim_id ? `·${a.claim_id}` : ""}`)
            .join(", ") || "none yet"}`}
        />
        <Stat label="вызовы LLM" value={m?.llm_calls ?? 0} format={fmtInt} />
        <Stat label="токены" value={(m?.tokens_in ?? 0) + (m?.tokens_out ?? 0)} format={fmtCompact} title={`${m?.tokens_in ?? 0} in / ${m?.tokens_out ?? 0} out`} />
        <Stat label="стоимость" value={m?.cost_usd ?? 0} format={fmtUsd} accent="#fbbf24" title="Оценка по прайсу роутера; локальная модель — $0" />
        <Stat label="время" value={elapsed} format={fmtElapsed} />
        <button
          disabled={!canReport}
          onClick={onShowReport}
          className="rounded-md border border-amber/60 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-amber transition enabled:hover:bg-amber/10 disabled:cursor-not-allowed disabled:border-line disabled:text-ink-3"
          title={canReport ? "Открыть отчёт" : "Доступен после run.finished"}
        >
          Отчёт
        </button>
      </div>
    </header>
  );
}
