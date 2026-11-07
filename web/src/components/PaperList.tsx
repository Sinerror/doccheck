import { AnimatePresence, motion } from "framer-motion";
import type { RunState } from "../state/runReducer";
import { paperColor } from "../lib/colors";

export function PaperList({ state }: { state: RunState }) {
  const target = Object.values(state.papers).find((p) => p.source === "target");
  const papers = state.paperOrder.map((id) => state.papers[id]);
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="panel-title">Corpus</span>
        <span className="font-mono text-[10px] text-ink-3">{papers.length} related</span>
      </div>
      <ul className="scroll-thin min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {target && (
          <li className="flex items-start gap-2 rounded-md border border-cyan/30 bg-cyan/5 px-2 py-1.5">
            <span className="mt-1 h-2 w-2 shrink-0 rotate-45" style={{ background: paperColor(-1, "target") }} />
            <div className="min-w-0">
              <div className="truncate text-[12px] text-ink" title={target.title}>{target.title}</div>
              <div className="font-mono text-[10px] text-ink-3">target{target.chunks ? ` · ${target.chunks} chunks` : ""}</div>
            </div>
          </li>
        )}
        <AnimatePresence initial={false}>
          {papers.map((p) => (
            <motion.li
              key={p.paper_id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.35 }}
              className="flex items-start gap-2 rounded-md px-2 py-1 hover:bg-panel-2"
            >
              <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: paperColor(p.order) }} />
              <div className="min-w-0 flex-1">
                <a
                  href={p.url ?? undefined}
                  target="_blank"
                  rel="noreferrer"
                  className="line-clamp-2 text-[12px] leading-snug text-ink-2 hover:text-ink"
                  title={p.title}
                >
                  {p.title}
                </a>
                <div className="font-mono text-[10px] text-ink-3">
                  {p.year ?? "n.d."}
                  {p.venue ? ` · ${p.venue}` : ""}
                  {p.chunks !== undefined ? ` · ${p.chunks} ch` : " · pending"}
                </div>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
        {!papers.length && <li className="px-2 py-3 font-mono text-[11px] text-ink-3">waiting for scout.paper_found…</li>}
      </ul>
    </div>
  );
}
