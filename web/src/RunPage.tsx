import { useEffect, useRef, useState } from "react";
import { useRunStream, type StreamMode } from "./state/useRunStream";
import { TopBar } from "./components/TopBar";
import { PipelineGraph } from "./components/PipelineGraph";
import { PaperList } from "./components/PaperList";
import { ClaimLanes } from "./components/ClaimCard";
import { VectorSpace } from "./components/VectorSpace";
import { EvidenceGraph } from "./components/EvidenceGraph";
import { EventLog } from "./components/EventLog";
import { ReportView } from "./components/ReportView";
import { STAGES } from "./state/runReducer";

export function RunPage({ runId, mode, speed, onHome, onMode }: {
  runId: string;
  mode: StreamMode;
  speed: number;
  onHome: () => void;
  onMode: (m: StreamMode, speed?: number) => void;
}) {
  const [nonce, setNonce] = useState(0);
  const [state, info] = useRunStream(runId, mode, speed, nonce);
  const [showReport, setShowReport] = useState(false);
  const autoOpened = useRef(false);

  // open the report once the run finishes (live, or when a replay reaches run.finished)
  useEffect(() => {
    if (state.status === "finished" && !autoOpened.current) {
      autoOpened.current = true;
      const t = setTimeout(() => setShowReport(true), 1200);
      return () => clearTimeout(t);
    }
  }, [state.status]);
  useEffect(() => {
    autoOpened.current = false;
    setShowReport(false);
  }, [runId, mode, nonce]);

  const claims = state.claimOrder.map((id) => state.claims[id]);
  const fatal = state.errors.find((e) => e.fatal);

  return (
    <div className="mx-auto flex max-w-[1920px] flex-col gap-3 p-3 pb-16">
      <TopBar state={state} mode={mode} info={info} onHome={onHome} onShowReport={() => setShowReport(true)} canReport={state.status === "finished"} />

      <div className="flex flex-wrap items-center gap-2 font-mono text-[10.5px]">
        {STAGES.map((s) => (
          <span
            key={s}
            className="rounded border px-2 py-0.5 uppercase tracking-wider"
            style={{
              borderColor: state.stages[s] === "active" ? "#22d3ee" : state.stages[s] === "done" ? "#2a3442" : "#1a2029",
              color: state.stages[s] === "active" ? "#22d3ee" : state.stages[s] === "done" ? "#a9b4c2" : "#4a5361",
            }}
          >
            {state.stages[s] === "done" ? "✓ " : state.stages[s] === "active" ? "● " : ""}
            {s}
          </span>
        ))}
        <span className="ml-auto flex items-center gap-2 text-ink-3">
          {mode === "replay" ? (
            <>
              replay speed
              {[1, 4, 16].map((sp) => (
                <button
                  key={sp}
                  onClick={() => (sp === speed ? setNonce((n) => n + 1) : onMode("replay", sp))}
                  className={`rounded border px-1.5 ${sp === speed ? "border-violet text-violet" : "border-line hover:text-ink"}`}
                  title={sp === speed ? "restart replay" : `replay at ${sp}×`}
                >
                  {sp}×
                </button>
              ))}
              <button onClick={() => setNonce((n) => n + 1)} className="rounded border border-line px-1.5 hover:text-ink">
                ↺ restart
              </button>
            </>
          ) : (
            <button onClick={() => onMode("replay", 4)} className="rounded border border-line px-1.5 hover:text-ink" disabled={state.status === "running"}>
              open as replay
            </button>
          )}
        </span>
      </div>

      {fatal && (
        <div className="panel border-critical/70 px-4 py-2 font-mono text-[12px] text-red">
          run failed: {fatal.message}
        </div>
      )}
      {info.error && mode === "replay" && !info.ended && (
        <div className="panel px-4 py-2 font-mono text-[12px] text-red">
          {info.error} — <button className="underline" onClick={() => setNonce((n) => n + 1)}>restart</button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_380px]">
        <aside className="panel flex flex-col gap-2 p-3 lg:sticky lg:top-3 lg:max-h-[calc(100vh-90px)]">
          <div className="panel-title">Pipeline</div>
          <PipelineGraph state={state} />
          <PaperList state={state} />
        </aside>

        <main className="min-w-0">
          <ClaimLanes claims={claims} />
        </main>

        <aside className="flex flex-col gap-3 lg:col-span-2 xl:sticky xl:top-3 xl:col-span-1 xl:h-[calc(100vh-90px)]">
          <div className="panel flex min-h-[300px] flex-1 flex-col p-3">
            <VectorSpace state={state} />
          </div>
          <div className="panel flex min-h-[280px] flex-1 flex-col p-3">
            <EvidenceGraph state={state} />
          </div>
        </aside>
      </div>

      <EventLog events={state.events} />
      {showReport && <ReportView runId={runId} onClose={() => setShowReport(false)} />}
    </div>
  );
}
