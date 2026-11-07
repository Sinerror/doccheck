import { useEffect, useRef, useState } from "react";

/** Reveals streamed text character by character, catching up faster when it falls behind.
 *  The animation loop runs only while the shown text lags the real streamed text. */
export function Typewriter({ text, active, className }: { text: string; active: boolean; className?: string }) {
  const [n, setN] = useState(text.length);
  const nRef = useRef(text.length);
  const prev = useRef(text);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (!text.startsWith(prev.current)) nRef.current = 0; // a new stream (next round) restarts the reveal
    prev.current = text;
    const goal = text.length;
    const step = () => {
      const cur = nRef.current;
      const next = cur >= goal ? goal : cur + Math.max(1, Math.ceil((goal - cur) / 14));
      nRef.current = next;
      setN(next);
      raf.current = next < goal ? requestAnimationFrame(step) : null;
    };
    if (raf.current != null) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
      raf.current = null;
    };
  }, [text]);

  const shown = text.slice(0, Math.min(n, text.length));
  const typing = shown.length < text.length;
  return <span className={`${className ?? ""} ${active || typing ? "caret" : ""}`}>{shown}</span>;
}
