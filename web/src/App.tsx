import { useEffect, useState } from "react";
import { HomePage } from "./HomePage";
import { RunPage } from "./RunPage";
import type { StreamMode } from "./state/useRunStream";

// Minimal hash router: #/  ·  #/run/<id>?mode=live|replay&speed=4
type Route = { page: "home" } | { page: "run"; runId: string; mode: StreamMode; speed: number };

function parse(hash: string): Route {
  const m = hash.match(/^#\/run\/([^?]+)(?:\?(.*))?$/);
  if (!m) return { page: "home" };
  const q = new URLSearchParams(m[2] ?? "");
  const mode = q.get("mode") === "live" ? "live" : "replay";
  const speed = Number(q.get("speed") ?? 4) || 4;
  return { page: "run", runId: decodeURIComponent(m[1]), mode, speed };
}

export function go(runId: string | null, mode: StreamMode = "replay", speed = 4) {
  window.location.hash = runId ? `#/run/${encodeURIComponent(runId)}?mode=${mode}${mode === "replay" ? `&speed=${speed}` : ""}` : "#/";
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  if (route.page === "run")
    return (
      <RunPage
        key={`${route.runId}-${route.mode}-${route.speed}`}
        runId={route.runId}
        mode={route.mode}
        speed={route.speed}
        onHome={() => go(null)}
        onMode={(mode, speed) => go(route.runId, mode, speed ?? route.speed)}
      />
    );
  return <HomePage onOpen={(id, mode) => go(id, mode)} />;
}
