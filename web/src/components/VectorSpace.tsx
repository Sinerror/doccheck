import { useEffect, useMemo, useRef, useState } from "react";
import type { RunState, RetrievalState } from "../state/runReducer";
import { api, type VectorPoint } from "../lib/api";
import { OTHER, PAPER_SLOTS, ROLE_COLOR, TARGET, paperColor } from "../lib/colors";
import { shortTitle } from "../lib/format";

const FADE_MS = 1500;
const PAD = 14;

interface Flash {
  r: RetrievalState;
  born: number;
}

export function VectorSpace({ state }: { state: RunState }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [points, setPoints] = useState<VectorPoint[]>([]);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [size, setSize] = useState({ w: 300, h: 260 });
  const [hover, setHover] = useState<{ x: number; y: number; p: VectorPoint } | null>(null);
  const flashes = useRef<Flash[]>([]);
  const seen = useRef(0);
  const raf = useRef<number | null>(null);

  // (re)load the map whenever the corpus grows; the server refits the PCA basis on the current points
  const indexedKey = state.totalPoints;
  useEffect(() => {
    if (!indexedKey && state.status !== "finished") return;
    let cancelled = false;
    const t = setTimeout(() => {
      api
        .vectorMap(state.runId)
        .then((p) => !cancelled && (setPoints(p), setLoadErr(null)))
        .catch((e) => !cancelled && setLoadErr(String(e.message ?? e)));
    }, 400);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [state.runId, indexedKey, state.status]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.max(200, e.contentRect.width), h: Math.max(200, e.contentRect.height) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // only show chunks of papers already indexed in this stream (replay reveals the map as it was built)
  const indexed = useMemo(() => new Set(Object.values(state.papers).filter((p) => p.chunks !== undefined).map((p) => p.paper_id)), [state.papers]);
  const visible = useMemo(() => points.filter((p) => indexed.has(p.paper_id)), [points, indexed]);
  const byId = useMemo(() => new Map(visible.map((p) => [p.id, p])), [visible]);

  const toPx = (x: number, y: number): [number, number] => [PAD + ((x + 1) / 2) * (size.w - 2 * PAD), PAD + ((1 - y) / 2) * (size.h - 2 * PAD)];

  // register new retrievals as flashes
  useEffect(() => {
    const fresh = state.retrievals.filter((r) => r.seq > seen.current);
    if (fresh.length) {
      seen.current = fresh[fresh.length - 1].seq;
      const now = performance.now();
      flashes.current = [...flashes.current, ...fresh.map((r) => ({ r, born: now }))].slice(-12);
    }
  }, [state.retrievals]);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr;
    cv.height = size.h * dpr;
    const ctx = cv.getContext("2d")!;

    const draw = () => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.w, size.h);
      // faint crosshair at the PCA origin
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.beginPath();
      const [ox, oy] = toPx(0, 0);
      ctx.moveTo(PAD, oy); ctx.lineTo(size.w - PAD, oy); ctx.moveTo(ox, PAD); ctx.lineTo(ox, size.h - PAD);
      ctx.stroke();

      for (const p of visible) {
        const paper = state.papers[p.paper_id];
        const [x, y] = toPx(p.x, p.y);
        const isT = p.source === "target";
        ctx.fillStyle = paperColor(paper?.order, p.source);
        ctx.globalAlpha = isT ? 0.95 : 0.75;
        ctx.beginPath();
        ctx.arc(x, y, isT ? 2.6 : 1.8, 0, Math.PI * 2);
        ctx.fill();
        if (isT) {
          ctx.globalAlpha = 0.35;
          ctx.strokeStyle = TARGET;
          ctx.beginPath();
          ctx.arc(x, y, 4.5, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      const now = performance.now();
      flashes.current = flashes.current.filter((f) => now - f.born < FADE_MS);
      for (const f of flashes.current) {
        if (!f.r.query_xy) continue;
        const a = 1 - (now - f.born) / FADE_MS;
        const color = ROLE_COLOR[f.r.by_role] ?? "#fff";
        const [qx, qy] = toPx(f.r.query_xy[0], f.r.query_xy[1]);
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.strokeStyle = color;
        f.r.point_ids.forEach((id, i) => {
          const p = byId.get(id);
          if (!p) return;
          const [x, y] = toPx(p.x, p.y);
          ctx.globalAlpha = a * (0.85 - i * 0.05);
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(x, y);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.stroke();
        });
        ctx.globalAlpha = a;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(qx, qy, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillStyle = color;
        ctx.fillText(`${f.r.by_role}${f.r.claim_id ? ` · ${f.r.claim_id}` : ""}`, Math.min(qx + 7, size.w - 90), Math.max(qy - 7, 12));
        ctx.globalAlpha = 1;
      }
      raf.current = flashes.current.length ? requestAnimationFrame(draw) : null;
    };
    if (raf.current != null) cancelAnimationFrame(raf.current);
    draw();
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
      raf.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, size, state.retrievals, byId, state.papers]);

  const onMove = (e: React.MouseEvent) => {
    const rect = canvas.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: VectorPoint | null = null;
    let bd = 64;
    for (const p of visible) {
      const [x, y] = toPx(p.x, p.y);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd) {
        bd = d;
        best = p;
      }
    }
    setHover(best ? { x: mx, y: my, p: best } : null);
  };

  const legend = state.paperOrder.slice(0, PAPER_SLOTS.length).map((id) => state.papers[id]);
  const last = state.retrievals[state.retrievals.length - 1];

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="panel-title">Vector space · PCA</span>
        <span className="font-mono text-[10px] text-ink-3">{visible.length} pts</span>
      </div>
      <div ref={wrap} className="relative min-h-[220px] flex-1 overflow-hidden rounded-md border border-line/60 bg-[#05070a]">
        <canvas ref={canvas} style={{ width: size.w, height: size.h }} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        {hover && (
          <div
            className="pointer-events-none absolute z-10 max-w-[240px] rounded border border-line bg-panel px-2 py-1 text-[11px] text-ink shadow-lg"
            style={{ left: Math.min(hover.x + 10, size.w - 250), top: hover.y + 10 }}
          >
            <div className="leading-snug">{shortTitle(hover.p.title, 90)}</div>
            <div className="font-mono text-[10px] text-ink-3">{hover.p.chunk_id ?? hover.p.paper_id}</div>
          </div>
        )}
        {!visible.length && (
          <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-ink-3">
            {loadErr ? `vector map unavailable: ${loadErr}` : "waiting for rag.indexed…"}
          </div>
        )}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-2.5 gap-y-0.5 font-mono text-[9.5px] text-ink-3">
        <span className="flex items-center gap-1"><span className="h-2 w-2 rotate-45" style={{ background: TARGET }} />документ</span>
        {legend.map((p) => (
          <span key={p.paper_id} className="flex max-w-[150px] items-center gap-1 truncate" title={p.title}>
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: paperColor(p.order) }} />
            {shortTitle(p.title, 22)}
          </span>
        ))}
        {state.paperOrder.length > PAPER_SLOTS.length && (
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full" style={{ background: OTHER }} />other</span>
        )}
      </div>
      {last && (
        <div className="mt-1 truncate font-mono text-[10px]" style={{ color: ROLE_COLOR[last.by_role] }} title={last.query}>
          ⌕ {last.by_role}: {last.query}
        </div>
      )}
    </div>
  );
}
