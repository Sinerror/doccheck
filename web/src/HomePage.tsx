import { useEffect, useState } from "react";
import { api, type CorpusDoc, type Example, type Health, type RunInfo } from "./lib/api";
import { fmtUsd, shortTitle } from "./lib/format";

const RUN_STATUS: Record<string, { color: string; icon: string; label: string }> = {
  finished: { color: "#0ca30c", icon: "✓", label: "готово" },
  failed: { color: "#d03b3b", icon: "✕", label: "ошибка" },
  running: { color: "#f87171", icon: "●", label: "идёт" },
  unknown: { color: "#6b7685", icon: "?", label: "?" },
};

function RunStatus({ status }: { status: string }) {
  const s = RUN_STATUS[status] ?? RUN_STATUS.unknown;
  return (
    <span className="inline-flex items-center gap-1 rounded-full border px-2 py-px font-mono text-[10px] uppercase tracking-wider text-ink" style={{ borderColor: `${s.color}99` }}>
      <span style={{ color: s.color }}>{s.icon}</span>
      {s.label}
    </span>
  );
}

type Source = "example" | "text" | "file";

export function HomePage({ onOpen }: { onOpen: (runId: string, mode: "live" | "replay") => void }) {
  const [source, setSource] = useState<Source>("example");
  const [examples, setExamples] = useState<Example[]>([]);
  const [example, setExample] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [maxClaims, setMaxClaims] = useState(10);
  const [rounds, setRounds] = useState(2);
  const [useCritic, setUseCritic] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [runsErr, setRunsErr] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [corpus, setCorpus] = useState<CorpusDoc[]>([]);

  const load = () => {
    api.runs().then((r) => (setRuns(r), setRunsErr(null))).catch((e) => setRunsErr(String(e.message ?? e)));
    api.health().then(setHealth).catch(() => setHealth(null));
  };
  useEffect(() => {
    load();
    api.examples().then((ex) => (setExamples(ex), ex.length && setExample(ex[1]?.name ?? ex[0].name))).catch(() => undefined);
    api.corpus().then(setCorpus).catch(() => undefined);
  }, []);

  const preview = examples.find((e) => e.name === example);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    const params = { max_claims: maxClaims, debate_rounds: rounds, use_critic: useCritic };
    setBusy(true);
    try {
      let run_id: string;
      if (source === "file") {
        if (!file) throw new Error("выберите файл");
        ({ run_id } = await api.createRunFile(file, Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)]))));
      } else if (source === "text") {
        if (!text.trim()) throw new Error("вставьте текст документа");
        ({ run_id } = await api.createRun({ text, title: "Вставленный текст", ...params }));
      } else {
        ({ run_id } = await api.createRun({ example, ...params }));
      }
      onOpen(run_id, "live");
    } catch (e2) {
      setErr(String((e2 as Error).message ?? e2));
    } finally {
      setBusy(false);
    }
  };

  const tab = (s: Source, label: string) => (
    <button
      type="button"
      onClick={() => setSource(s)}
      className={`rounded-md border px-3 py-1 font-mono text-[11.5px] ${source === s ? "border-cyan text-cyan" : "border-line text-ink-3 hover:text-ink"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col gap-6 px-4 py-10">
      <header>
        <div className="font-mono text-sm tracking-[0.4em] text-cyan">DOCCHECK</div>
        <h1 className="mt-2 text-2xl font-medium text-ink">Проверка документа по источникам компании</h1>
        <p className="mt-1 max-w-3xl text-[13.5px] text-ink-2">
          Ответ клиенту, КП или памятка → Scout выписывает проверяемые утверждения → Analyst сверяет каждое с тарифами, офертой и
          регламентами → каждая цитата проверяется кодом → Critic другой модели ищет опровержения через поиск по корпусу. Спорное
          уходит человеку.
        </p>
        <div className="mt-3 flex flex-wrap gap-3 font-mono text-[10.5px]">
          {health ? (
            <>
              <span style={{ color: health.qdrant ? "#0ca30c" : "#d03b3b" }}>{health.qdrant ? "✓" : "✕"} qdrant · {health.collection} · {health.points ?? "?"} pts</span>
              <span style={{ color: health.llm_key ? "#0ca30c" : "#d03b3b" }}>{health.llm_key ? "✓ ключ LLM" : "✕ нет ключа LLM — доступен только Replay"}</span>
              <span className="text-ink-3">{Object.entries(health.models).map(([k, v]) => `${k}=${v}`).join(" · ")}</span>
            </>
          ) : (
            <span className="text-red">✕ бэкенд недоступен (запущен ли uvicorn на :8000?)</span>
          )}
        </div>
      </header>

      <form onSubmit={submit} className="panel flex flex-col gap-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          {tab("example", "Пример из набора")}
          {tab("text", "Вставить текст")}
          {tab("file", "Загрузить файл")}
          <button
            type="submit"
            disabled={busy || (health !== null && !health.ok)}
            className="ml-auto rounded-md bg-cyan/90 px-5 py-2 font-mono text-[13px] font-medium uppercase tracking-wider text-bg transition hover:bg-cyan disabled:opacity-40"
          >
            {busy ? "запуск…" : "Проверить"}
          </button>
        </div>
        {source === "example" && (
          <div className="flex flex-col gap-2">
            <select value={example} onChange={(e) => setExample(e.target.value)} className="rounded-md border border-line bg-bg px-2 py-1.5 text-[13px] text-ink" aria-label="Пример">
              {examples.map((e) => (
                <option key={e.name} value={e.name}>
                  {e.title} ({e.name})
                </option>
              ))}
            </select>
            {preview && <pre className="scroll-thin max-h-40 overflow-y-auto whitespace-pre-wrap rounded-md border border-line/60 bg-bg/60 p-2 text-[12px] text-ink-2">{preview.preview}…</pre>}
          </div>
        )}
        {source === "text" && (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            placeholder="Вставьте ответ клиенту, коммерческое предложение или памятку…"
            className="rounded-md border border-line bg-bg px-3 py-2 text-[13px] text-ink outline-none focus:border-cyan"
          />
        )}
        {source === "file" && (
          <input type="file" accept=".md,.txt,.html,.htm,.docx,.pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-[12px] text-ink-2 file:mr-2 file:rounded file:border file:border-line file:bg-panel-2 file:px-2 file:text-ink" />
        )}
        <div className="flex flex-wrap items-center gap-4 font-mono text-[11px] text-ink-3">
          <label className="flex items-center gap-1.5">
            макс. утверждений
            <input type="number" min={1} max={20} value={maxClaims} onChange={(e) => setMaxClaims(+e.target.value)} className="w-14 rounded border border-line bg-bg px-1.5 py-0.5 text-ink" />
          </label>
          <label className="flex items-center gap-1.5">
            раундов спора
            <input type="number" min={0} max={4} value={rounds} onChange={(e) => setRounds(+e.target.value)} className="w-14 rounded border border-line bg-bg px-1.5 py-0.5 text-ink" />
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={useCritic} onChange={(e) => setUseCritic(e.target.checked)} /> критик (дороже, точнее)
          </label>
        </div>
        {err && <div className="font-mono text-[12px] text-red">✕ {err}</div>}
      </form>

      <section className="panel p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="panel-title">Прогоны</span>
          <button onClick={load} className="font-mono text-[10.5px] text-ink-3 hover:text-ink">↻ обновить</button>
        </div>
        {runsErr && <div className="font-mono text-[12px] text-red">✕ {runsErr}</div>}
        {runs === null && !runsErr && <div className="font-mono text-[12px] text-ink-3">загрузка…</div>}
        {runs && !runs.length && <div className="font-mono text-[12px] text-ink-3">прогонов пока нет</div>}
        <ul className="divide-y divide-line/60">
          {runs?.map((r) => (
            <li key={r.run_id} className="flex flex-wrap items-center gap-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] text-ink" title={r.title ?? ""}>{shortTitle(r.title, 90)}</div>
                <div className="font-mono text-[10.5px] text-ink-3">
                  {r.run_id}
                  {r.created_at ? ` · ${new Date(r.created_at * 1000).toLocaleString("ru-RU")}` : ""}
                  {r.claims != null ? ` · утверждений ${r.claims}, на проверку ${r.needs_review}` : ""}
                  {r.grounding_rate != null ? ` · цитаты ${(r.grounding_rate * 100).toFixed(0)}%` : ""}
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

      {corpus.length > 0 && (
        <section className="panel p-4">
          <div className="panel-title mb-2">Источники компании · {corpus.length} документов (синтетический корпус вымышленной «Сорока CRM»)</div>
          <ul className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
            {corpus.map((d) => (
              <li key={d.doc_id} className="truncate text-[12px] text-ink-2" title={d.sections.join(" · ")}>
                <span className="font-mono text-[10.5px] text-ink-3">{d.doc_id}</span> — {d.title}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
