import { useEffect, useMemo, useRef, useState } from "react";
import { EVENT_TYPES, type TriadEvent } from "../types/events";
import { ROLE_COLOR } from "../lib/colors";
import { fmtTime } from "../lib/format";

const TYPE_COLOR: Record<string, string> = {
  error: "#d03b3b",
  "run.finished": "#fbbf24",
  "run.started": "#fbbf24",
  "claim.final": "#fbbf24",
  "debate.round": "#fbbf24",
  "grounding.check": "#0ca30c",
  "rag.retrieval": "#22d3ee",
  "rag.indexed": "#22d3ee",
};

const MAX_ROWS = 400;

export function EventLog({ events }: { events: TriadEvent[] }) {
  const [open, setOpen] = useState(false);
  const [type, setType] = useState<string>("all");
  const [role, setRole] = useState<string>("all");
  const [hidePartials, setHidePartials] = useState(true);
  const [follow, setFollow] = useState(true);
  const [expanded, setExpanded] = useState<number | null>(null);
  const body = useRef<HTMLDivElement>(null);

  const rows = useMemo(() => {
    const f = events.filter(
      (e) =>
        (type === "all" || e.type === type) &&
        (role === "all" || (e.role ?? "none") === role) &&
        !(hidePartials && type !== "agent.partial" && e.type === "agent.partial"),
    );
    return f.length > MAX_ROWS ? f.slice(f.length - MAX_ROWS) : f;
  }, [events, type, role, hidePartials]);

  useEffect(() => {
    if (open && follow && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [rows, open, follow]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const e of events) c[e.type] = (c[e.type] ?? 0) + 1;
    return c;
  }, [events]);
  const errors = counts["error"] ?? 0;

  return (
    <section className="panel fixed inset-x-3 bottom-3 z-40 flex flex-col overflow-hidden" style={{ height: open ? "38vh" : 38, background: "#0d1118" }}>
      <button onClick={() => setOpen((o) => !o)} className="flex h-[38px] shrink-0 items-center gap-3 px-3 text-left">
        <span className="font-mono text-[11px] text-cyan">{open ? "▾" : "▸"} EVENT LOG</span>
        <span className="font-mono text-[10.5px] text-ink-3">
          {events.length} events · seq {events.length ? events[events.length - 1].seq : 0}
        </span>
        {errors > 0 && <span className="font-mono text-[10.5px] text-red">⚠ {errors} error</span>}
        {!open && events.length > 0 && (
          <span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-ink-3">
            last: {events[events.length - 1].type} {JSON.stringify(events[events.length - 1].data).slice(0, 120)}
          </span>
        )}
      </button>
      {open && (
        <>
          <div className="flex shrink-0 flex-wrap items-center gap-2 border-y border-line px-3 py-1.5 font-mono text-[10.5px]">
            <label className="flex items-center gap-1 text-ink-3">
              type
              <select value={type} onChange={(e) => setType(e.target.value)} className="rounded border border-line bg-bg px-1 py-0.5 text-ink">
                <option value="all">all</option>
                {EVENT_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t} ({counts[t] ?? 0})
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-ink-3">
              role
              <select value={role} onChange={(e) => setRole(e.target.value)} className="rounded border border-line bg-bg px-1 py-0.5 text-ink">
                {["all", "scout", "analyst", "critic", "system", "none"].map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-ink-3">
              <input type="checkbox" checked={hidePartials} onChange={(e) => setHidePartials(e.target.checked)} /> hide agent.partial
            </label>
            <label className="flex items-center gap-1 text-ink-3">
              <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow
            </label>
            <span className="ml-auto text-ink-3">showing {rows.length}{rows.length === MAX_ROWS ? ` (last ${MAX_ROWS})` : ""}</span>
          </div>
          <div ref={body} className="scroll-thin min-h-0 flex-1 overflow-y-auto bg-[#05070a] px-3 py-1 font-mono text-[11px] leading-[1.55]">
            {rows.map((e) => (
              <div key={e.seq} className="cursor-pointer hover:bg-white/[0.03]" onClick={() => setExpanded(expanded === e.seq ? null : e.seq)}>
                <div className="flex gap-2 whitespace-nowrap">
                  <span className="w-12 shrink-0 text-right text-ink-3">{e.seq}</span>
                  <span className="shrink-0 text-ink-3">{fmtTime(e.ts)}</span>
                  <span className="w-32 shrink-0" style={{ color: TYPE_COLOR[e.type] ?? "#a9b4c2" }}>
                    {e.type}
                  </span>
                  <span className="w-14 shrink-0" style={{ color: ROLE_COLOR[e.role ?? ""] ?? "#6b7685" }}>
                    {e.role ?? "-"}
                  </span>
                  <span className="w-8 shrink-0 text-cyan">{e.claim_id ?? ""}</span>
                  <span className="truncate text-ink-2">{JSON.stringify(e.data)}</span>
                </div>
                {expanded === e.seq && (
                  <pre className="my-1 ml-14 whitespace-pre-wrap break-all rounded border border-line bg-panel p-2 text-[10.5px] text-ink-2">
                    {JSON.stringify(e, null, 2)}
                  </pre>
                )}
              </div>
            ))}
            {!rows.length && <div className="py-2 text-ink-3">no events{events.length ? " match the filter" : " yet"}</div>}
          </div>
        </>
      )}
    </section>
  );
}
