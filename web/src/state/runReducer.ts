// Pure reducer: the whole UI state is a fold over the backend event stream.
// Nothing here invents data — every field is derived from a TriadEvent.
import type {
  AnalystRevision,
  AnalystVerdict,
  ClaimWithQueries,
  Critique,
  GroundingCheck,
  Metrics,
  RagRetrieval,
  ScoutPaperFound,
  TriadEvent,
} from "../types/events";

export type StageName = "ingest" | "scout" | "index" | "analyze" | "report";
export const STAGES: StageName[] = ["ingest", "scout", "index", "analyze", "report"];
export type StageStatus = "pending" | "active" | "done";
export type Role = "scout" | "analyst" | "critic" | "system";

export interface RoundState {
  round: number;
  analyst?: AnalystVerdict & Partial<Pick<AnalystRevision, "response_to_critic">>;
  critic?: Critique;
  grounding: GroundingCheck[]; // analyst evidence checks
  criticGrounding: GroundingCheck[];
  stance?: "AGREE" | "CHALLENGE";
  maxSeverity?: string | null;
}

export interface ClaimState {
  id: string;
  text: string;
  type: string;
  section: string;
  phase: "waiting" | "analyst" | "critic" | "done";
  activeRole?: Role;
  round: number;
  analystText: string;
  analystThinking: string;
  criticText: string;
  criticTools: { name: string; summary: string; seq: number }[];
  rounds: RoundState[];
  final?: { status: "accepted" | "revised" | "disputed"; verdict: string };
  error?: string;
}

export interface PaperState extends ScoutPaperFound {
  chunks?: number;
  source: "target" | "related";
  order: number; // order of appearance → stable color slot
}

export interface RetrievalState extends RagRetrieval {
  seq: number;
  ts: number;
  claim_id?: string | null;
}

/** One unit of traffic on a pipeline edge, for particle animation. */
export interface Pulse {
  edge: EdgeId;
  seq: number;
  claim?: string | null;
}
export type EdgeId = "scout-qdrant" | "qdrant-analyst" | "qdrant-critic" | "analyst-critic" | "critic-analyst" | "critic-report";

/** One spawned agent instance: a role bound to a claim (or the run, for scout / meta-review). */
export interface AgentInstance {
  key: string;
  role: Role;
  claim_id: string | null;
  model: string;
  calls: number;
}

export interface RunState {
  runId: string;
  status: "connecting" | "running" | "finished" | "failed";
  target?: Record<string, unknown>;
  params?: Record<string, unknown>;
  models?: Record<string, string>;
  stages: Record<StageName, StageStatus>;
  claims: Record<string, ClaimState>;
  claimOrder: string[];
  papers: Record<string, PaperState>;
  paperOrder: string[];
  queries: number;
  candidatesScanned: number;
  totalPoints: number;
  retrievals: RetrievalState[];
  retrievalCount: number;
  metrics?: Metrics;
  maxDebateRound: number;
  activeRoles: Record<Role, number>; // how many agents of each role are currently working
  agents: Record<string, AgentInstance>; // distinct (role, claim) instances seen in agent.started
  pulses: Pulse[];
  errors: { seq: number; message: string; where?: string | null; fatal: boolean; claim_id?: string | null }[];
  summary?: Record<string, unknown>;
  events: TriadEvent[];
  firstTs?: number;
  lastTs?: number;
  lastSeq: number;
}

export function initialState(runId: string): RunState {
  return {
    runId,
    status: "connecting",
    stages: { ingest: "pending", scout: "pending", index: "pending", analyze: "pending", report: "pending" },
    claims: {},
    claimOrder: [],
    papers: {},
    paperOrder: [],
    queries: 0,
    candidatesScanned: 0,
    totalPoints: 0,
    retrievals: [],
    retrievalCount: 0,
    maxDebateRound: 0,
    activeRoles: { scout: 0, analyst: 0, critic: 0, system: 0 },
    agents: {},
    pulses: [],
    errors: [],
    events: [],
    lastSeq: 0,
  };
}

const MAX_PULSES = 120;
const MAX_RETRIEVALS = 40;

function pulse(s: RunState, edge: EdgeId, seq: number, claim?: string | null): Pulse[] {
  const next = [...s.pulses, { edge, seq, claim }];
  return next.length > MAX_PULSES ? next.slice(next.length - MAX_PULSES) : next;
}

function updateClaim(s: RunState, id: string | null | undefined, fn: (c: ClaimState) => ClaimState): RunState {
  if (!id || !s.claims[id]) return s;
  return { ...s, claims: { ...s.claims, [id]: fn(s.claims[id]) } };
}

function upsertRound(c: ClaimState, round: number, fn: (r: RoundState) => RoundState): ClaimState {
  const rounds = [...c.rounds];
  let i = rounds.findIndex((r) => r.round === round);
  if (i < 0) {
    rounds.push({ round, grounding: [], criticGrounding: [] });
    rounds.sort((a, b) => a.round - b.round);
    i = rounds.findIndex((r) => r.round === round);
  }
  rounds[i] = fn(rounds[i]);
  return { ...c, rounds };
}

function paperOf(chunkId: string): string {
  const i = chunkId.lastIndexOf("#");
  return i > 0 ? chunkId.slice(0, i) : chunkId;
}
export { paperOf };

export function reduce(state: RunState, ev: TriadEvent): RunState {
  if (ev.seq <= state.lastSeq) return state; // idempotent against reconnect duplicates
  let s: RunState = {
    ...state,
    events: [...state.events, ev],
    lastSeq: ev.seq,
    firstTs: state.firstTs ?? ev.ts,
    lastTs: ev.ts,
    status: state.status === "connecting" ? "running" : state.status,
  };

  switch (ev.type) {
    case "run.started": {
      const d = ev.data;
      s = { ...s, target: d.target, params: d.params, models: d.models };
      const pid = String(d.target.paper_id ?? "target");
      s.papers = {
        ...s.papers,
        [pid]: {
          paper_id: pid,
          title: String(d.target.title ?? pid),
          year: (d.target.year as number) ?? null,
          url: (d.target.url as string) ?? null,
          source: "target",
          order: -1,
        },
      };
      break;
    }
    case "stage.started":
      s = { ...s, stages: { ...s.stages, [ev.data.stage]: "active" } };
      break;
    case "stage.finished":
      s = { ...s, stages: { ...s.stages, [ev.data.stage]: "done" } };
      break;
    case "scout.claims": {
      const claims = { ...s.claims };
      const order: string[] = [];
      for (const raw of ev.data.claims) {
        const c = raw as unknown as ClaimWithQueries;
        order.push(c.id);
        claims[c.id] = {
          id: c.id,
          text: c.text,
          type: c.type,
          section: c.section,
          phase: "waiting",
          round: 0,
          analystText: "",
          analystThinking: "",
          criticText: "",
          criticTools: [],
          rounds: [],
        };
      }
      s = { ...s, claims, claimOrder: order };
      break;
    }
    case "scout.query":
      s = { ...s, queries: s.queries + 1, candidatesScanned: s.candidatesScanned + (ev.data.hits ?? 0) };
      break;
    case "scout.paper_found": {
      const order = s.paperOrder.length;
      s = {
        ...s,
        papers: { ...s.papers, [ev.data.paper_id]: { ...ev.data, source: "related", order } },
        paperOrder: [...s.paperOrder, ev.data.paper_id],
      };
      break;
    }
    case "rag.indexed": {
      const p = s.papers[ev.data.paper_id];
      s = {
        ...s,
        totalPoints: ev.data.total_points,
        papers: p ? { ...s.papers, [p.paper_id]: { ...p, chunks: ev.data.chunks } } : s.papers,
        pulses: pulse(s, "scout-qdrant", ev.seq),
      };
      break;
    }
    case "rag.retrieval": {
      const r: RetrievalState = { ...ev.data, seq: ev.seq, ts: ev.ts, claim_id: ev.claim_id };
      const rs = [...s.retrievals, r];
      s = {
        ...s,
        retrievals: rs.length > MAX_RETRIEVALS ? rs.slice(rs.length - MAX_RETRIEVALS) : rs,
        retrievalCount: s.retrievalCount + 1,
        pulses: pulse(s, ev.data.by_role === "critic" ? "qdrant-critic" : "qdrant-analyst", ev.seq, ev.claim_id),
      };
      break;
    }
    case "agent.started": {
      const role = ev.data.role;
      const cid = ev.data.claim_id ?? null;
      const key = `${role}:${cid ?? (role === "analyst" ? "meta" : "run")}`;
      const prev = s.agents[key];
      s = {
        ...s,
        activeRoles: { ...s.activeRoles, [role]: s.activeRoles[role] + 1 },
        agents: { ...s.agents, [key]: { key, role, claim_id: cid, model: ev.data.model, calls: (prev?.calls ?? 0) + 1 } },
      };
      s = updateClaim(s, ev.data.claim_id, (c) => ({
        ...c,
        phase: role === "critic" ? "critic" : "analyst",
        activeRole: role,
        round: ev.data.round ?? 0,
        ...(role === "analyst" ? { analystText: "", analystThinking: "" } : { criticText: "", criticTools: [] }),
      }));
      break;
    }
    case "agent.partial": {
      const d = ev.data;
      s = updateClaim(s, d.claim_id, (c) => {
        if (d.role === "critic") return { ...c, criticText: d.text };
        if (d.kind === "thinking") return { ...c, analystThinking: d.text };
        return { ...c, analystText: d.text };
      });
      break;
    }
    case "agent.output": {
      const d = ev.data;
      const role = d.role;
      s = { ...s, activeRoles: { ...s.activeRoles, [role]: Math.max(0, s.activeRoles[role] - 1) } };
      const round = d.round ?? 0;
      if (role === "analyst" && d.claim_id) {
        const out = d.output as unknown as AnalystVerdict;
        s = updateClaim(s, d.claim_id, (c) => upsertRound({ ...c, analystText: out.reasoning }, round, (r) => ({ ...r, analyst: out })));
        s = { ...s, pulses: pulse(s, "analyst-critic", ev.seq, d.claim_id) };
      } else if (role === "critic" && d.claim_id) {
        const out = d.output as unknown as Critique;
        s = updateClaim(s, d.claim_id, (c) => upsertRound({ ...c, criticText: out.reasoning }, round, (r) => ({ ...r, critic: out })));
      }
      break;
    }
    case "tool.call":
      s = updateClaim(s, ev.claim_id, (c) => ({
        ...c,
        criticTools: [...c.criticTools, { name: ev.data.name, summary: ev.data.summary, seq: ev.seq }],
      }));
      break;
    case "tool.result":
      s = updateClaim(s, ev.claim_id, (c) => {
        const tools = [...c.criticTools];
        for (let i = tools.length - 1; i >= 0; i--) {
          if (tools[i].name === ev.data.name && !tools[i].summary.includes(" → ")) {
            tools[i] = { ...tools[i], summary: `${tools[i].summary} → ${ev.data.summary}` };
            break;
          }
        }
        return { ...c, criticTools: tools };
      });
      break;
    case "grounding.check": {
      const g = ev.data;
      s = updateClaim(s, g.claim_id, (c) =>
        upsertRound(c, g.round ?? 0, (r) =>
          g.by_role === "critic" ? { ...r, criticGrounding: [...r.criticGrounding, g] } : { ...r, grounding: [...r.grounding, g] },
        ),
      );
      break;
    }
    case "debate.round": {
      const d = ev.data;
      s = updateClaim(s, d.claim_id, (c) => upsertRound(c, d.round, (r) => ({ ...r, stance: d.stance, maxSeverity: d.max_severity })));
      s = { ...s, maxDebateRound: Math.max(s.maxDebateRound, d.round) };
      if (d.stance === "CHALLENGE") s = { ...s, pulses: pulse(s, "critic-analyst", ev.seq, d.claim_id) };
      break;
    }
    case "claim.final":
      s = updateClaim(s, ev.data.claim_id, (c) => ({
        ...c,
        phase: "done",
        activeRole: undefined,
        final: { status: ev.data.status, verdict: ev.data.verdict },
      }));
      s = { ...s, pulses: pulse(s, "critic-report", ev.seq, ev.data.claim_id) };
      break;
    case "metrics":
      s = { ...s, metrics: ev.data };
      break;
    case "run.finished":
      s = { ...s, status: "finished", summary: ev.data.summary };
      break;
    case "error": {
      const d = ev.data;
      s = { ...s, errors: [...s.errors, { seq: ev.seq, message: d.message, where: d.where, fatal: !!d.fatal, claim_id: ev.claim_id }] };
      if (d.fatal) s = { ...s, status: "failed" };
      else if (ev.claim_id && d.where === "analyze") s = updateClaim(s, ev.claim_id, (c) => ({ ...c, phase: "done", error: d.message }));
      break;
    }
  }
  return s;
}

/** Latest analyst verdict (and the one before it, for the revision diff). */
export function verdictHistory(c: ClaimState): string[] {
  return c.rounds.filter((r) => r.analyst).map((r) => r.analyst!.verdict);
}

export function groundingFor(r: RoundState, chunkId: string, quote: string): GroundingCheck | undefined {
  return r.grounding.find((g) => g.chunk_id === chunkId && g.quote === quote);
}
