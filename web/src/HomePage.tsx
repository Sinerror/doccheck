import { useEffect, useState } from "react";
import { api, type Health, type RunInfo } from "./lib/api";
import { fmtUsd, shortTitle } from "./lib/format";

const RUN_STATUS: Record<string, { color: string; icon: string }> = {
  finished: { color: "#0ca30c", icon: "✓" },
  failed: { color: "#d03b3b", icon: "✕" },
  running: { color: "#f87171", icon: "●" },
  unknown: { color: "#6b7685", icon: "?" },
};

function RunStatus({ status }: { status: string }) {
  const s = RUN_STATUS[status] ?? RUN_STATUS.unknown;
  return (
    <span className="inline-flex items-center gap-1 rounded-full border px-2 py-px font-mono text-[10px] uppercase tracking-wider text-ink" style={{ borderColor: `${s.color}99` }}>
      <span style={{ color: s.color }}>{s.icon}</span>
      {status}
    </span>
  );
}

export function HomePage({ onOpen }: { onOpen: (runId: string, mode: "live" | "replay") => void }) {
  const [arxiv, setArxiv] = useState("");
  const [maxClaims, setMaxClaims] = useState(4);
  const [maxPapers, setMaxPapers] = useState(10);
  const [rounds, setRounds] = useState(2);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [runsErr, setRunsErr] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  const load = () => {
    api.runs().then((r) => (setRuns(r), setRunsErr(null))).catch((e) => setRunsErr(String(e.message ?? e)));
    api.health().then(setHealth).catch(() => setHealth(null));
  };
  useEffect(load, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!file && !arxiv.trim()) {
      setErr("enter an arXiv id / URL or choose a PDF");
      return;
    }
    setBusy(true);
    try {
      const params = { max_claims: maxClaims, max_papers: maxPapers, debate_rounds: rounds };
      const { run_id } = file
        ? await api.createRunPdf(file, Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])))
        : await api.createRun({ ...(arxiv.includes("/") ? { url: arxiv.trim() } : { arxiv_id: arxiv.trim() }), ...params });
      onOpen(run_id, "live");
    } catch (e2) {
      setErr(String((e2 as Error).message ?? e2));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col gap-6 px-4 py-10">
      <header>
        <div className="font-mono text-sm tracking-[0.4em] text-cyan">TRIAD</div>
        <h1 className="mt-2 text-2xl font-medium text-ink">Scout → Analyst ⇄ Critic</h1>
        <p className="mt-1 max-w-2xl text-[13.5px] text-ink-2">
          Agentic review of a scientific paper: claims are extracted, checked against retrieved literature, every quote is
          verified verbatim, and an adversarial critic challenges the analyst until they agree or the dispute is shown.
        </p>
        <div className="mt-3 flex flex-wrap gap-3 font-mono text-[10.5px]">
          {health ? (
            <>
              <span style={{ color: health.qdrant ? "#0ca30c" : "#d03b3b" }}>{health.qdrant ? "✓" : "✕"} qdrant · {health.collection} · {health.points ?? "?"} pts</span>
              <span style={{ color: health.llm_key ? "#0ca30c" : "#d03b3b" }}>{health.llm_key ? "✓" : "✕"} llm key</span>
              <span className="text-ink-3">{Object.entries(health.models).map(([k, v]) => `${k}=${v}`).join(" · ")}</span>
            </>
          ) : (
            <span className="text-red">✕ backend unreachable (is uvicorn running on :8000?)</span>
          )}
        </div>
      </header>

      <form onSubmit={submit} className="panel flex flex-col gap-3 p-4">
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={arxiv}
            onChange={(e) => setArxiv(e.target.value)}
            placeholder="arXiv id or URL, e.g. 1706.03762"
            className="min-w-0 flex-1 rounded-md border border-line bg-bg px-3 py-2 font-mono text-[14px] text-ink outline-none focus:border-cyan"
            disabled={busy || !!file}
            aria-label="arXiv id or URL"
          />
          <button
            type="submit"
            disabled={busy || (health !== null && !health.ok)}
            className="rounded-md bg-cyan/90 px-5 py-2 font-mono text-[13px] font-medium uppercase tracking-wider text-bg transition hover:bg-cyan disabled:opacity-40"
          >
            {busy ? "starting…" : "Run"}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-4 font-mono text-[11px] text-ink-3">
          <label className="flex items-center gap-1.5">
            max_claims
            <input type="number" min={1} max={20} value={maxClaims} onChange={(e) => setMaxClaims(+e.target.value)} className="w-14 rounded border border-line bg-bg px-1.5 py-0.5 text-ink" />
          </label>
          <label className="flex items-center gap-1.5">
            max_papers
            <input type="number" min={0} max={40} value={maxPapers} onChange={(e) => setMaxPapers(+e.target.value)} className="w-14 rounded border border-line bg-bg px-1.5 py-0.5 text-ink" />
          </label>
          <label className="flex items-center gap-1.5">
            debate_rounds
            <input type="number" min={0} max={4} value={rounds} onChange={(e) => setRounds(+e.target.value)} className="w-14 rounded border border-line bg-bg px-1.5 py-0.5 text-ink" />
          </label>
          <label className="flex items-center gap-1.5">
            or PDF
            <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="max-w-[220px] text-ink-2 file:mr-2 file:rounded file:border file:border-line file:bg-panel-2 file:px-2 file:text-ink" />
          </label>
        </div>
        {err && <div className="font-mono text-[12px] text-red">✕ {err}</div>}
      </form>

      <section className="panel p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="panel-title">Past runs</span>
          <button onClick={load} className="font-mono text-[10.5px] text-ink-3 hover:text-ink">↻ refresh</button>
        </div>
        {runsErr && <div className="font-mono text-[12px] text-red">✕ {runsErr}</div>}
        {runs === null && !runsErr && <div className="font-mono text-[12px] text-ink-3">loading…</div>}
        {runs && !runs.length && <div className="font-mono text-[12px] text-ink-3">no runs yet — start one above</div>}
        <ul className="divide-y divide-line/60">
          {runs?.map((r) => (
            <li key={r.run_id} className="flex flex-wrap items-center gap-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] text-ink" title={r.title ?? ""}>{shortTitle(r.title, 90)}</div>
                <div className="font-mono text-[10.5px] text-ink-3">
                  {r.run_id}
                  {r.created_at ? ` · ${new Date(r.created_at * 1000).toLocaleString()}` : ""}
                  {r.review_score != null ? ` · score ${r.review_score}` : ""}
                  {r.grounding_rate != null ? ` · grounding ${(r.grounding_rate * 100).toFixed(0)}%` : ""}
                  {r.cost_usd != null ? ` · ${fmtUsd(r.cost_usd)}` : ""}
                </div>
              </div>
              <RunStatus status={r.status} />
              {r.status === "running" && (
                <button onClick={() => onOpen(r.run_id, "live")} className="rounded border border-red/70 px-2.5 py-1 font-mono text-[11px] text-red hover:bg-red/10">
                  ● Live
                </button>
              )}
              <button onClick={() => onOpen(r.run_id, "replay")} className="rounded border border-violet/70 px-2.5 py-1 font-mono text-[11px] text-violet hover:bg-violet/10">
                ▶ Replay
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
