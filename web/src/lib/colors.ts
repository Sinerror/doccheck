// Source-document identity colors: validated 8-slot dark categorical palette, assigned in order of
// appearance (rag.indexed), never cycled. Documents past slot 8 fold into "Other".
export const PAPER_SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const OTHER = "#5b6472";
export const TARGET = "#22d3ee";

export function paperColor(order: number | undefined, source?: string): string {
  if (source === "target" || order === -1) return TARGET;
  if (order === undefined || order < 0 || order >= PAPER_SLOTS.length) return OTHER;
  return PAPER_SLOTS[order];
}

export const ROLE_COLOR: Record<string, string> = {
  scout: "#22d3ee",
  analyst: "#a78bfa",
  critic: "#f87171",
  system: "#fbbf24",
};

export const VERDICT_STYLE: Record<string, { label: string; color: string; icon: string }> = {
  SUPPORTED: { label: "Подтверждено", color: "#0ca30c", icon: "✓" },
  PARTIALLY_SUPPORTED: { label: "Частично", color: "#fab219", icon: "◐" },
  OVERSTATED: { label: "Преувеличено", color: "#ec835a", icon: "⇡" },
  CONTRADICTED: { label: "Противоречит", color: "#d03b3b", icon: "✕" },
  UNVERIFIABLE: { label: "Не найдено", color: "#8b95a3", icon: "?" },
};

export const STATUS_STYLE: Record<string, { color: string; icon: string; label: string }> = {
  accepted: { color: "#0ca30c", icon: "✓", label: "критик согласен" },
  revised: { color: "#fbbf24", icon: "↻", label: "пересмотрено" },
  disputed: { color: "#d03b3b", icon: "⚑", label: "спорное" },
};

export const CLAIM_TYPE_RU: Record<string, string> = {
  number: "число",
  deadline: "срок",
  condition: "условие",
  reference: "ссылка на пункт",
  feature: "состав тарифа",
  other: "другое",
};

export const SEVERITY_COLOR: Record<string, string> = { low: "#8b95a3", medium: "#ec835a", high: "#d03b3b" };
