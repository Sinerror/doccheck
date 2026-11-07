import { AnimatePresence, motion } from "framer-motion";
import type { RunState } from "../state/runReducer";
import { paperColor } from "../lib/colors";

/** The document under check and the company sources it is checked against (from rag.indexed events). */
export function PaperList({ state }: { state: RunState }) {
  const target = Object.values(state.papers).find((p) => p.source === "target");
  const docs = state.paperOrder.map((id) => state.papers[id]);
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="panel-title">Источники</span>
        <span className="font-mono text-[10px] text-ink-3">
          {docs.length} док.{state.corpusReused ? " · индекс из кэша" : ""}
        </span>
      </div>
      <ul className="scroll-thin min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {target && (
          <li className="flex items-start gap-2 rounded-md border border-cyan/30 bg-cyan/5 px-2 py-1.5">
            <span className="mt-1 h-2 w-2 shrink-0 rotate-45" style={{ background: paperColor(-1, "target") }} />
            <div className="min-w-0">
              <div className="truncate text-[12px] text-ink" title={target.title}>{target.title}</div>
              <div className="font-mono text-[10px] text-ink-3">на проверке{target.chunks ? ` · ${target.chunks} фрагм.` : ""}</div>
            </div>
          </li>
        )}
        <AnimatePresence initial={false}>
          {docs.map((p) => (
            <motion.li
              key={p.paper_id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.35 }}
              className="flex items-start gap-2 rounded-md px-2 py-1 hover:bg-panel-2"
            >
              <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: paperColor(p.order) }} />
              <div className="min-w-0 flex-1">
                <div className="line-clamp-2 text-[12px] leading-snug text-ink-2" title={p.title}>{p.title}</div>
                <div className="font-mono text-[10px] text-ink-3">
                  {p.paper_id}
                  {p.chunks !== undefined ? ` · ${p.chunks} фрагм.` : ""}
                </div>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
        {!docs.length && <li className="px-2 py-3 font-mono text-[11px] text-ink-3">индексация источников…</li>}
      </ul>
    </div>
  );
}
