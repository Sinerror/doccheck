export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}
export function fmtCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 10_000) return `${(n / 1000).toFixed(1)}k`;
  return fmtInt(n);
}
export function fmtUsd(n: number): string {
  return `$${n.toFixed(n < 1 ? 4 : 2)}`;
}
export function fmtElapsed(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}
export function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false });
}
export function shortTitle(t: string | null | undefined, n = 60): string {
  if (!t) return "untitled";
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}
