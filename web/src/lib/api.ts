// Thin typed wrappers over the backend REST API (proxied by Vite to :8000).

export interface RunInfo {
  run_id: string;
  status: "finished" | "failed" | "running" | "unknown";
  title: string | null;
  created_at: number | null;
  params?: Record<string, unknown>;
  review_score?: number | null;
  grounding_rate?: number | null;
  status_counts?: Record<string, number>;
  cost_usd?: number | null;
}

export interface VectorPoint {
  id: string;
  x: number;
  y: number;
  paper_id: string;
  source: "target" | "related";
  title: string | null;
  chunk_id?: string;
}

export interface Health {
  ok: boolean;
  qdrant: boolean;
  llm_key: boolean;
  collection: string;
  points?: number;
  models: Record<string, string>;
  qdrant_error?: string;
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

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),
  runs: () => fetch("/api/runs").then((r) => json<RunInfo[]>(r)),
  report: (id: string) => fetch(`/api/runs/${encodeURIComponent(id)}/report`).then((r) => json<Report>(r)),
  reportMarkdown: (id: string) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/report.md`).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.text();
    }),
  vectorMap: (id: string) => fetch(`/api/runs/${encodeURIComponent(id)}/vector-map`).then((r) => json<VectorPoint[]>(r)),
  createRun: (body: Record<string, unknown>) =>
    fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(
      (r) => json<{ run_id: string }>(r),
    ),
  createRunPdf: (file: File, params: Record<string, string>) => {
    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(params)) fd.append(k, v);
    return fetch("/api/runs", { method: "POST", body: fd }).then((r) => json<{ run_id: string }>(r));
  },
};

// Shape of runs/{id}/report.json (backend triad/orchestrator.py: Pipeline.report)
export interface ReportClaim {
  id: string;
  text: string;
  type: string;
  section: string;
  status: "accepted" | "revised" | "disputed";
  verdict: string;
  confidence: number;
  critic_suggested: string | null;
  critic_stance: string;
  history: unknown[];
}

export interface Report {
  run_id: string;
  target: Record<string, unknown>;
  models: Record<string, string>;
  review_score: number | null;
  grounding_rate: number | null;
  meta_review: string | null;
  aggregate: {
    review_score: number | null;
    score_formula: string;
    claim_scores: Record<string, number>;
    verdict_counts: Record<string, number>;
    status_counts: Record<string, number>;
    grounding: { rate: number | null; checks: number; ok: number; by_role: Record<string, { checks: number; ok: number }> };
    critic_issues: { total: number; by_type: Record<string, number>; by_severity: Record<string, number> };
  };
  claims: ReportClaim[];
  claim_errors: { claim_id: string; error: string }[];
  metrics: { cost_usd: number; llm_calls: number; tokens_in: number; tokens_out: number; elapsed_s: number };
}
