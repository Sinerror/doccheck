import { useEffect, useMemo, useRef, useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  Handle,
  Position,
  ReactFlow,
  useNodesState,
  getBezierPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import type { ClaimState, EdgeId, RunState } from "../state/runReducer";
import { STATUS_STYLE } from "../lib/colors";

// The fan-out mirrors what actually runs: one Scout, one Qdrant corpus, and for every extracted claim its own
// Analyst instance and Critic instance debating in parallel; all instances come from agent.started events.

type NodeData = { label: string; sub: string; color: string; active: boolean; count?: string; doneColor?: string; small?: boolean };
type EdgeData = { latest: number; color: string; label?: string; dim?: boolean }; // latest = seq of the newest event on this edge

function StageNode({ data }: NodeProps<Node<NodeData>>) {
  const border = data.active ? data.color : data.doneColor ?? "#1f2733";
  return (
    <div
      className={`relative rounded-lg border text-center transition-shadow duration-300 ${data.small ? "w-[96px] px-1.5 py-1" : "w-[120px] px-2.5 py-1.5"}`}
      style={{
        borderColor: border,
        background: "#0d1118",
        boxShadow: data.active ? `0 0 0 1px ${data.color}66, 0 0 20px ${data.color}55` : "none",
        opacity: data.active || data.doneColor ? 1 : 0.85,
      }}
    >
      {data.active && <span className="absolute inset-0 animate-pulse rounded-lg" style={{ boxShadow: `inset 0 0 14px ${data.color}40` }} />}
      <Handle type="target" position={Position.Top} id="t" />
      <Handle type="source" position={Position.Bottom} id="b" />
      <Handle type="source" position={Position.Right} id="r" />
      <Handle type="target" position={Position.Left} id="l" />
      <Handle type="target" position={Position.Right} id="rt" />
      <Handle type="source" position={Position.Left} id="ls" />
      <div className={`font-mono font-medium tracking-wider ${data.small ? "text-[9.5px]" : "text-[11px]"}`} style={{ color: data.color }}>
        {data.label}
      </div>
      <div className="truncate font-mono text-[8.5px] text-ink-3" title={data.sub}>
        {data.sub}
      </div>
      {data.count && <div className="truncate font-mono text-[8.5px] text-ink-2">{data.count}</div>}
    </div>
  );
}

function Particle({ path, color }: { path: string; color: string }) {
  const motion = useRef<SVGAnimateMotionElement>(null);
  const fade = useRef<SVGAnimateElement>(null);
  useEffect(() => {
    // begin="indefinite" + beginElement(): SMIL begin times are document-relative, so start explicitly
    motion.current?.beginElement();
    fade.current?.beginElement();
  }, []);
  return (
    <circle r={3} fill={color} opacity={0} style={{ filter: `drop-shadow(0 0 4px ${color})` }}>
      <animateMotion ref={motion} dur="1.1s" begin="indefinite" fill="freeze" path={path} />
      <animate ref={fade} attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.8;1" dur="1.1s" begin="indefinite" fill="freeze" />
    </circle>
  );
}

const PARTICLE_LIFE_MS = 1100; // matches the SMIL dur above
const PARTICLE_SPACING_MS = 300; // bursts of ~20 events/s collapse into one particle per 300 ms
const MAX_PARTICLES = Math.ceil(PARTICLE_LIFE_MS / PARTICLE_SPACING_MS) + 1;

/** Each edge owns its particles: a new event spawns one (rate-limited), and every particle always plays its full
 *  flight before it is removed, so overlapping traffic overlaps instead of restarting the animation. */
function useParticles(latest: number): number[] {
  const [ids, setIds] = useState<number[]>([]);
  const lastSeq = useRef(latest); // events already present at mount (e.g. a finished replay) don't fire
  const lastSpawn = useRef(-Infinity);
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());

  useEffect(() => {
    if (latest === lastSeq.current) return;
    lastSeq.current = latest;
    const now = performance.now();
    if (now - lastSpawn.current < PARTICLE_SPACING_MS) return;
    lastSpawn.current = now;
    setIds((cur) => [...cur, latest].slice(-MAX_PARTICLES));
    const t = setTimeout(() => {
      timers.current.delete(t);
      setIds((cur) => cur.filter((x) => x !== latest));
    }, PARTICLE_LIFE_MS + 50);
    timers.current.add(t);
  }, [latest]);

  useEffect(() => {
    const ts = timers.current;
    return () => ts.forEach(clearTimeout);
  }, []);
  return ids;
}

function ParticleEdge(props: EdgeProps<Edge<EdgeData>>) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data } = props;
  const [path, lx, ly] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, curvature: 0.3 });
  const color = data?.color ?? "#2a3442";
  const particles = useParticles(data?.latest ?? 0);
  return (
    <>
      <BaseEdge path={path} style={{ stroke: data?.dim ? "#18202b" : "#2a3442", strokeWidth: 1.2 }} />
      {particles.map((id) => (
        <Particle key={id} path={path} color={color} />
      ))}
      {data?.label && (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan pointer-events-none absolute rounded border border-amber/50 bg-bg/90 px-1 font-mono text-[8.5px] text-amber"
            style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}
          >
            {data.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

function sameData(a: NodeData, b: NodeData): boolean {
  return a.label === b.label && a.sub === b.sub && a.color === b.color && a.active === b.active &&
    a.count === b.count && a.doneColor === b.doneColor && a.small === b.small;
}

const nodeTypes = { stage: StageNode };
const edgeTypes = { particle: ParticleEdge };

const ROW_Y0 = 172;
const ROW_H = 64;
const COLOR = { scout: "#22d3ee", analyst: "#a78bfa", critic: "#f87171", report: "#fbbf24" };

function shortModel(m?: string) {
  return m ?? "—";
}

export function PipelineGraph({ state }: { state: RunState }) {
  const st = state.stages;
  const m = state.models ?? {};
  const claims: (ClaimState | null)[] = state.claimOrder.length ? state.claimOrder.map((id) => state.claims[id]) : [null];
  const rows = claims.length;
  const recentRetrieval =
    state.retrievals.length > 0 && state.lastTs !== undefined && state.lastTs - state.retrievals[state.retrievals.length - 1].ts < 2;
  const finals = claims.filter((c) => c?.final).length;
  const meta = state.agents["analyst:meta"];
  const reportY = ROW_Y0 + rows * ROW_H + 6;

  const desired: Node<NodeData>[] = useMemo(() => {
    const out: Node<NodeData>[] = [
      {
        id: "scout", type: "stage", position: { x: 54, y: 0 }, draggable: false,
        data: { label: "SCOUT", sub: shortModel(m.scout), color: COLOR.scout, active: st.scout === "active" || state.activeRoles.scout > 0,
          count: state.claimOrder.length ? `${state.claimOrder.length} claims · ${state.paperOrder.length} papers` : undefined,
          doneColor: st.scout === "done" ? "#22d3ee55" : undefined },
      },
      {
        id: "qdrant", type: "stage", position: { x: 54, y: 84 }, draggable: false,
        data: { label: "QDRANT", sub: "dense + bm25 · RRF", color: COLOR.scout, active: st.index === "active" || st.ingest === "active" || recentRetrieval,
          count: state.totalPoints ? `${state.totalPoints} chunks` : undefined },
      },
    ];
    claims.forEach((c, i) => {
      const y = ROW_Y0 + i * ROW_H;
      const cid = c?.id ?? "?";
      const done = c?.final ? STATUS_STYLE[c.final.status]?.color : undefined;
      const rounds = c ? c.rounds.filter((r) => r.analyst).length : 0;
      const aInst = c ? state.agents[`analyst:${c.id}`] : undefined;
      const cInst = c ? state.agents[`critic:${c.id}`] : undefined;
      out.push({
        id: `a-${cid}`, type: "stage", position: { x: 0, y }, draggable: false,
        data: { small: true, label: c ? `ANALYST·${cid}` : "ANALYST ×N", sub: c ? shortModel(m.analyst) : "one per claim",
          color: COLOR.analyst, active: c?.phase === "analyst", doneColor: done,
          count: aInst ? `${aInst.calls} call${aInst.calls > 1 ? "s" : ""}${rounds > 1 ? ` · v${rounds}` : ""}` : c ? "queued" : undefined },
      });
      out.push({
        id: `c-${cid}`, type: "stage", position: { x: 132, y }, draggable: false,
        data: { small: true, label: c ? `CRITIC·${cid}` : "CRITIC ×N", sub: c ? shortModel(m.critic) : "one per claim",
          color: COLOR.critic, active: c?.phase === "critic", doneColor: done,
          count: c?.final ? `${STATUS_STYLE[c.final.status]?.icon ?? ""} ${c.final.status}` : cInst ? `${c?.criticTools.length ?? 0} tool calls` : c ? "queued" : undefined },
      });
    });
    out.push({
      id: "report", type: "stage", position: { x: 54, y: reportY }, draggable: false,
      data: { label: "REPORT", sub: meta ? `meta-review · ${shortModel(meta.model)}` : "aggregate + meta-review", color: COLOR.report,
        active: st.report === "active" || state.status === "finished",
        count: state.claimOrder.length ? `${finals}/${state.claimOrder.length} claims final` : undefined },
    });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st, m.scout, m.analyst, m.critic, state.activeRoles, state.agents, recentRetrieval, state.totalPoints, state.claims, state.claimOrder, state.paperOrder.length, finals, state.status, reportY]);

  // React Flow keeps the measured size of each node only on the node objects it is given back
  // (onNodesChange writes `measured`). Rebuilding fresh node objects on every event dropped `measured`, so React
  // Flow re-measured all nodes and unmounted/remounted every edge up to ~20×/s: the whole graph flickered and
  // particles restarted. Keep node objects stable and only patch `data` / `position` when they really change.
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NodeData>>(desired);
  useEffect(() => {
    setNodes((prev) => {
      const byId = new Map(prev.map((n) => [n.id, n]));
      let changed = prev.length !== desired.length;
      const next = desired.map((d, i) => {
        const p = byId.get(d.id);
        if (!p) {
          changed = true;
          return d;
        }
        const same = p.position.x === d.position.x && p.position.y === d.position.y && sameData(p.data, d.data);
        if (!same) changed = true;
        if (prev[i]?.id !== d.id) changed = true;
        return same ? p : { ...p, position: d.position, data: d.data };
      });
      return changed ? next : prev;
    });
  }, [desired, setNodes]);

  const edges: Edge<EdgeData>[] = useMemo(() => {
    const latest = (edge: EdgeId, claim?: string) => {
      for (let i = state.pulses.length - 1; i >= 0; i--) {
        const p = state.pulses[i];
        if (p.edge === edge && (claim === undefined || p.claim === claim)) return p.seq;
      }
      return 0;
    };
    const out: Edge<EdgeData>[] = [
      { id: "scout-qdrant", source: "scout", target: "qdrant", sourceHandle: "b", targetHandle: "t", type: "particle",
        data: { color: COLOR.scout, latest: latest("scout-qdrant") } },
    ];
    for (const c of claims) {
      const cid = c?.id ?? "?";
      const dim = !!c?.final;
      const rounds = c ? Math.max(0, ...c.rounds.filter((r) => r.stance).map((r) => r.round + 1)) : 0;
      out.push(
        { id: `q-a-${cid}`, source: "qdrant", target: `a-${cid}`, sourceHandle: "b", targetHandle: "t", type: "particle",
          data: { dim, color: COLOR.analyst, latest: c ? latest("qdrant-analyst", c.id) : 0 } },
        { id: `q-c-${cid}`, source: "qdrant", target: `c-${cid}`, sourceHandle: "b", targetHandle: "t", type: "particle",
          data: { dim, color: COLOR.critic, latest: c ? latest("qdrant-critic", c.id) : 0 } },
        { id: `a-c-${cid}`, source: `a-${cid}`, target: `c-${cid}`, sourceHandle: "r", targetHandle: "l", type: "particle",
          data: { color: COLOR.analyst, latest: c ? latest("analyst-critic", c.id) : 0, label: rounds ? `⇄ r${rounds}` : undefined } },
        { id: `c-a-${cid}`, source: `c-${cid}`, target: `a-${cid}`, sourceHandle: "ls", targetHandle: "rt", type: "particle",
          data: { color: COLOR.critic, latest: c ? latest("critic-analyst", c.id) : 0 } },
        { id: `c-r-${cid}`, source: `c-${cid}`, target: "report", sourceHandle: "b", targetHandle: "t", type: "particle",
          data: { color: COLOR.report, latest: c ? latest("critic-report", c.id) : 0 } },
      );
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.pulses, state.claims, state.claimOrder]);

  const height = Math.max(340, reportY + 70);
  return (
    <div className="w-full shrink-0" style={{ height: Math.min(height, 620) }}>
      <ReactFlow
        key={rows /* refit when the fan-out grows */}
        nodes={nodes}
        onNodesChange={onNodesChange}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.08 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
        proOptions={{ hideAttribution: true }}
      />
    </div>
  );
}
