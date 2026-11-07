import { STATUS_STYLE, VERDICT_STYLE } from "../lib/colors";

export function VerdictBadge({ verdict, struck = false, size = "sm" }: { verdict: string; struck?: boolean; size?: "sm" | "md" }) {
  const v = VERDICT_STYLE[verdict] ?? { label: verdict, color: "#8b95a3", icon: "·" };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border font-mono uppercase tracking-wide ${
        size === "md" ? "px-2 py-0.5 text-[11px]" : "px-1.5 py-px text-[10px]"
      } ${struck ? "line-through opacity-50" : ""}`}
      style={{ borderColor: `${v.color}80`, color: "#e6edf3", background: `${v.color}22` }}
      title={verdict}
    >
      <span style={{ color: v.color }}>{v.icon}</span>
      {v.label}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLE[status] ?? { color: "#8b95a3", icon: "·", label: status };
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border px-2 py-px font-mono text-[10px] uppercase tracking-wider text-ink"
      style={{ borderColor: `${s.color}99`, background: `${s.color}26` }}
    >
      <span style={{ color: s.color }}>{s.icon}</span>
      {s.label}
    </span>
  );
}

export function RoleDot({ color, pulsing }: { color: string; pulsing?: boolean }) {
  return (
    <span className="relative inline-flex h-2 w-2">
      {pulsing && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: color }} />}
      <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: color }} />
    </span>
  );
}
