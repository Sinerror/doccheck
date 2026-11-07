import { useEffect, useReducer, useRef, useState } from "react";
import type { DocCheckEvent } from "../types/events";
import { initialState, reduce, type RunState } from "./runReducer";

export type StreamMode = "live" | "replay";

type Action = { kind: "reset"; runId: string } | { kind: "events"; events: DocCheckEvent[] };

function reducer(state: RunState, a: Action): RunState {
  if (a.kind === "reset") return initialState(a.runId);
  let s = state;
  for (const ev of a.events) s = reduce(s, ev);
  return s;
}

export interface StreamInfo {
  connected: boolean;
  ended: boolean;
  error: string | null;
}

/** Subscribes to the backend SSE stream (live or replay) and folds events into RunState.
 *  Events are flushed once per animation frame so bursts of `agent.partial` don't thrash React. */
export function useRunStream(runId: string, mode: StreamMode, speed = 4, nonce = 0): [RunState, StreamInfo] {
  const [state, dispatch] = useReducer(reducer, runId, initialState);
  const [info, setInfo] = useState<StreamInfo>({ connected: false, ended: false, error: null });
  const buffer = useRef<DocCheckEvent[]>([]);
  const frame = useRef<number | null>(null);

  useEffect(() => {
    dispatch({ kind: "reset", runId });
    setInfo({ connected: false, ended: false, error: null });
    buffer.current = [];
    const url =
      mode === "replay"
        ? `/api/runs/${encodeURIComponent(runId)}/replay?speed=${speed}`
        : `/api/runs/${encodeURIComponent(runId)}/events`;
    const es = new EventSource(url);
    let ended = false;

    const flush = () => {
      frame.current = null;
      if (buffer.current.length) {
        const events = buffer.current;
        buffer.current = [];
        dispatch({ kind: "events", events });
      }
    };

    es.onopen = () => setInfo((i) => ({ ...i, connected: true, error: null }));
    es.onmessage = (m) => {
      try {
        buffer.current.push(JSON.parse(m.data) as DocCheckEvent);
      } catch {
        return;
      }
      if (frame.current == null) frame.current = requestAnimationFrame(flush);
    };
    es.addEventListener("end", () => {
      ended = true;
      es.close();
      flush();
      setInfo({ connected: false, ended: true, error: null });
    });
    es.onerror = () => {
      if (ended) return;
      if (mode === "replay") {
        // a replay restarts from scratch on reconnect; stop instead of duplicating timing
        es.close();
        setInfo({ connected: false, ended: false, error: "replay stream interrupted" });
      } else {
        setInfo((i) => ({ ...i, connected: false, error: "connection lost, retrying…" }));
      }
    };
    return () => {
      es.close();
      if (frame.current != null) cancelAnimationFrame(frame.current);
      frame.current = null;
    };
  }, [runId, mode, speed, nonce]);

  return [state, info];
}
