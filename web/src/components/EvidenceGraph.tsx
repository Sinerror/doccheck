import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import { paperOf, type RunState } from "../state/runReducer";
import { paperColor } from "../lib/colors";
import { shortTitle } from "../lib/format";

interface GNode {
  id: string;
  title: string;
  color: string;
  target: boolean;
  fx?: number;
  fy?: number;
  x?: number;
  y?: number;
}
interface GLink {
  source: string;
  target: string;
  stance: "supports" | "contradicts";
  count: number;
  curvature: number;
}

const GOOD = "#0ca30c";
const BAD = "#d03b3b";

/** Edges come from the analyst's latest evidence per claim (stance supports/contradicts) plus the critic's
 *  counter-evidence (counted as contradicts). Context-only quotes and quotes from the target itself add no edge. */
function buildLinks(state: RunState, targetId: string): GLink[] {
  const counts = new Map<string, number>();
  const add = (chunkId: string, stance: "supports" | "contradicts") => {
    const pid = paperOf(chunkId);
    if (pid === targetId || !state.papers[pid]) return;
    const k = `${pid}|${stance}`;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  };
  for (const id of state.claimOrder) {
    const c = state.claims[id];
    const rounds = c.rounds.filter((r) => r.analyst);
    const latest = rounds[rounds.length - 1];
    for (const e of latest?.analyst?.evidence ?? []) {
      if (e.stance === "supports" || e.stance === "contradicts") add(e.chunk_id, e.stance);
    }
    for (const r of c.rounds) for (const iss of r.critic?.issues ?? []) for (const ce of iss.counter_evidence ?? []) add(ce.chunk_id, "contradicts");
  }
  const links: GLink[] = [];
  for (const [k, count] of counts) {
    const [pid, stance] = k.split("|") as [string, "supports" | "contradicts"];
    const both = counts.has(`${pid}|supports`) && counts.has(`${pid}|contradicts`);
    links.push({ source: pid, target: targetId, stance, count, curvature: both ? (stance === "supports" ? 0.25 : -0.25) : 0 });
  }
  return links;
}

export function EvidenceGraph({ state }: { state: RunState }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 300, h: 240 });
  const nodesRef = useRef(new Map<string, GNode>());
  const fg = useRef<ForceGraphMethods | undefined>(undefined);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.max(200, e.contentRect.width), h: Math.max(180, e.contentRect.height) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const target = Object.values(state.papers).find((p) => p.source === "target");
  const targetId = target?.paper_id ?? "target";
  const links = useMemo(() => buildLinks(state, targetId), [state.claims, state.papers, targetId]); // eslint-disable-line react-hooks/exhaustive-deps

  const data = useMemo(() => {
    const map = nodesRef.current;
    const ids = new Set<string>();
    if (target) ids.add(target.paper_id);
    state.paperOrder.forEach((id) => ids.add(id));
    for (const id of ids) {
      const p = state.papers[id];
      if (!map.has(id)) {
        const isT = p.source === "target";
        map.set(id, { id, title: p.title, color: paperColor(p.order, p.source), target: isT, ...(isT ? { fx: 0, fy: 0 } : {}) });
      }
    }
    // keep node objects stable so the simulation does not restart from scratch
    return { nodes: [...ids].map((id) => map.get(id)!), links: links.map((l) => ({ ...l })) };
  }, [state.paperOrder, state.papers, target, links]);

  // keep papers on a ring around the target: unlinked nodes would otherwise drift far apart under charge
  useEffect(() => {
    const g = fg.current;
    if (!g) return;
    let nodes: GNode[] = [];
    const ring = Object.assign(
      (alpha: number) => {
        for (const n of nodes) {
          if (n.target || n.x === undefined || n.y === undefined) continue;
          const r = Math.hypot(n.x, n.y) || 1;
          const k = ((90 - r) / r) * alpha * 0.25;
          (n as GNode & { vx: number; vy: number }).vx += n.x * k;
          (n as GNode & { vx: number; vy: number }).vy += n.y * k;
        }
      },
      { initialize: (ns: GNode[]) => (nodes = ns) },
    );
    g.d3Force("ring", ring as never);
    (g.d3Force("charge") as unknown as { strength?: (v: number) => void } | undefined)?.strength?.(-25);
    g.d3ReheatSimulation();
  }, [data.nodes.length]);

  const sup = links.filter((l) => l.stance === "supports").reduce((a, l) => a + l.count, 0);
  const con = links.filter((l) => l.stance === "contradicts").reduce((a, l) => a + l.count, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="panel-title">Evidence graph</span>
        <span className="font-mono text-[10px]">
          <span style={{ color: GOOD }}>✓ {sup} supports</span> <span className="text-ink-3">·</span>{" "}
          <span style={{ color: BAD }}>✕ {con} contradicts</span>
        </span>
      </div>
      <div ref={wrap} className="relative min-h-[200px] flex-1 overflow-hidden rounded-md border border-line/60 bg-[#05070a]">
        {data.nodes.length > 0 ? (
          <ForceGraph2D
            ref={fg}
            // zoom is clamped by maxZoom, so fitting one or two nodes stays readable
            onEngineStop={() => fg.current?.zoomToFit(400, 40)}
            width={size.w}
            height={size.h}
            graphData={data}
            backgroundColor="rgba(0,0,0,0)"
            nodeId="id"
            nodeLabel={(n) => (n as GNode).title}
            nodeCanvasObject={(n, ctx, scale) => {
              const node = n as GNode;
              const r = node.target ? 7 : 4.5;
              ctx.beginPath();
              ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, Math.PI * 2);
              ctx.fillStyle = node.color;
              ctx.shadowColor = node.color;
              ctx.shadowBlur = node.target ? 16 : 6;
              ctx.fill();
              ctx.shadowBlur = 0;
              if (node.target || scale > 2.2) {
                ctx.font = `${node.target ? 11 : 9}px Inter, sans-serif`;
                ctx.fillStyle = "#a9b4c2";
                ctx.textAlign = "center";
                ctx.fillText(shortTitle(node.title, node.target ? 32 : 24), node.x ?? 0, (node.y ?? 0) + r + 10);
              }
            }}
            nodePointerAreaPaint={(n, color, ctx) => {
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(n.x ?? 0, n.y ?? 0, 8, 0, Math.PI * 2);
              ctx.fill();
            }}
            linkColor={(l) => ((l as GLink).stance === "supports" ? GOOD : BAD)}
            linkWidth={(l) => 0.8 + 1.4 * (l as GLink).count}
            linkCurvature={(l) => (l as GLink).curvature}
            linkLabel={(l) => `${(l as GLink).count} ${(l as GLink).stance} quote(s)`}
            linkDirectionalParticles={(l) => Math.min(3, (l as GLink).count)}
            linkDirectionalParticleWidth={2}
            linkDirectionalParticleColor={(l) => ((l as GLink).stance === "supports" ? GOOD : BAD)}
            cooldownTicks={120}
            minZoom={0.3}
            maxZoom={1.6}
            d3VelocityDecay={0.3}
            enableZoomInteraction={true}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-ink-3">waiting for evidence…</div>
        )}
      </div>
    </div>
  );
}
