// Thin typed wrappers over the backend REST API (proxied by Vite to :8000, same origin in docker).

export interface RunInfo {
  run_id: string;
  status: "finished" | "failed" | "running" | "unknown";
  title: string | null;
  created_at: number | null;
  params?: Record<string, unknown>;
  grounding_rate?: number | null;
  claims?: number;
  needs_review?: number;
  verdict_counts?: Record<string, number>;
  status_counts?: Record<string, number>;
  cost_usd?: number | null;
}

export interface VectorPoint {
  id: string;
  x: number;
  y: number;
  paper_id: string; // = doc_id (kept as paper_id for the canvas components)
  source: "target" | "related";
  title: string | null;
  section?: string | null;
  chunk_id?: string;
}

export interface Health {
  ok: boolean;
  qdrant: boolean;
  llm_key: boolean;
  collection: string;
  points?: number;
  analyst_local?: boolean;
  models: Record<string, string>;
  qdrant_error?: string;
}

export interface Example {
  name: string;
  title: string;
  chars: number;
  preview: string;
}

export interface CorpusDoc {
  doc_id: string;
  title: string;
  sections: string[];
  chars: number;
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

interface RawPoint {
  id: string;
  x: number;
  y: number;
  doc_id: string;
  source: "corpus" | "document";
  title: string | null;
  section?: string | null;
  chunk_id?: string;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),
  runs: () => fetch("/api/runs").then((r) => json<RunInfo[]>(r)),
  examples: () => fetch("/api/examples").then((r) => json<Example[]>(r)),
  example: (name: string) => fetch(`/api/examples/${encodeURIComponent(name)}`).then((r) => json<{ name: string; text: string }>(r)),
  corpus: () => fetch("/api/corpus").then((r) => json<CorpusDoc[]>(r)),
  report: (id: string) => fetch(`/api/runs/${encodeURIComponent(id)}/report`).then((r) => json<Report>(r)),
  reportMarkdown: (id: string) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/report.md`).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.text();
    }),
  vectorMap: (id: string) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/vector-map`)
      .then((r) => json<RawPoint[]>(r))
      .then((pts) =>
        pts.map<VectorPoint>((p) => ({ ...p, paper_id: p.doc_id, source: p.source === "document" ? "target" : "related" })),
      ),
  createRun: (body: Record<string, unknown>) =>
    fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(
      (r) => json<{ run_id: string }>(r),
    ),
  createRunFile: (file: File, params: Record<string, string>) => {
    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(params)) fd.append(k, v);
    return fetch("/api/runs", { method: "POST", body: fd }).then((r) => json<{ run_id: string }>(r));
  },
};

// Shape of runs/{id}/report.json (backend doccheck/orchestrator.py: Pipeline.report)
export interface ReportClaim {
  id: string;
  text: string;
  quote: string;
  type: string;
  section: string;
  status: "accepted" | "revised" | "disputed";
  verdict: string;
  verdict_ru: string;
  confidence: number;
  correction: string | null;
  critic_suggested: string | null;
  critic_stance: string | null;
  history: unknown[];
}

export interface Report {
  run_id: string;
  target: { title?: string; filename?: string; chars?: number; pages?: number };
  document_context?: string;
  corpus: { docs: { doc_id: string; title: string }[]; chunks: number; embed_model: string; reused: boolean };
  models: Record<string, string>;
  grounding_rate: number | null;
  summary: string | null;
  aggregate: {
    verdict_counts: Record<string, number>;
    status_counts: Record<string, number>;
    needs_review: { count: number; share: number | null; claims: Record<string, string[]>; rule: string };
    grounding: { rate: number | null; checks: number; ok: number; by_role: Record<string, { checks: number; ok: number }> };
    critic_issues: { total: number; by_type: Record<string, number>; by_severity: Record<string, number> };
  };
  claims: ReportClaim[];
  claim_errors: { claim_id: string; error: string }[];
  metrics: { cost_usd: number; llm_calls: number; tokens_in: number; tokens_out: number; elapsed_s: number };
  business: { cost_usd: number; cost_per_page_usd: number; elapsed_s: number; claims_total: number; needs_review: number; needs_review_share: number | null };
}
