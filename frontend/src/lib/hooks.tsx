import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api, type Health } from "./api";

/* ---------------- server health (shared, polled) ---------------- */

interface HealthCtx {
  health: Health | null;
  error: string | null;
  refresh: () => void;
}
const HealthContext = createContext<HealthCtx>({ health: null, error: null, refresh: () => {} });

export function HealthProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number>();

  const refresh = useCallback(() => {
    api.health()
      .then((h) => { setHealth(h); setError(null); })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    refresh();
    const tick = () => {
      // poll fast while the models load, slowly afterwards
      const fast = !health || health.models === "loading" || health.models === "idle" || error;
      timer.current = window.setTimeout(() => { refresh(); tick(); }, fast ? 3000 : 20000);
    };
    tick();
    return () => window.clearTimeout(timer.current);
  }, [refresh, health?.models, error]); // eslint-disable-line react-hooks/exhaustive-deps

  return <HealthContext.Provider value={{ health, error, refresh }}>{children}</HealthContext.Provider>;
}

export const useHealth = () => useContext(HealthContext);

/* ---------------- one-shot async loader ---------------- */

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [n, setN] = useState(0);
  useEffect(() => {
    let live = true;
    setLoading(true);
    fn()
      .then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e: Error) => { if (live) setError(e.message); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [...deps, n]); // eslint-disable-line react-hooks/exhaustive-deps
  return { data, setData, error, loading, reload: () => setN((x) => x + 1) };
}

/* ---------------- media query ---------------- */

export function useMedia(query: string) {
  const [match, setMatch] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setMatch(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, [query]);
  return match;
}
