import { useEffect, useRef, useState } from "react";
import { ChevronRight, Loader2, Search } from "lucide-react";
import { api, type GuidelineHit } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { Empty, Notice, Skeleton } from "../components/ui";

const QUESTIONS = [
  "Can my mother sign the form for me?",
  "What if I tick two programs?",
  "How do I write my CNIC number?",
  "Can I write N/A in a box?",
  "How is the percentage calculated?",
  "I made a mistake while writing",
];

export default function Guidelines() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<GuidelineHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sections = useAsync(() => api.guidelines(), []);
  const input = useRef<HTMLInputElement>(null);
  const asked = useRef("");

  const ask = async (text: string) => {
    const t = text.trim();
    if (t.length < 2) return;
    asked.current = t;
    setQ(t);
    setBusy(true);
    setError(null);
    try {
      const r = await api.searchGuidelines(t, 4);
      if (asked.current === t) setHits(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => { input.current?.focus(); }, []);
  const best = hits?.[0]?.score ?? 1;

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>Guidelines</h1>
          <p>Ask a question in your own words. FormSense finds the guideline passage that answers it, the same way it cites a guideline for every problem it finds.</p>
        </div>
      </div>

      <section className="panel panel-pad">
        <form onSubmit={(e) => { e.preventDefault(); ask(q); }}>
          <div className="search">
            <Search size={18} />
            <input ref={input} className="input" value={q} onChange={(e) => setQ(e.target.value)} maxLength={300}
                   placeholder="e.g. Can my father sign instead of me?" aria-label="Ask about the form guidelines" />
          </div>
        </form>
        <div className="chips" aria-label="Example questions">
          {QUESTIONS.map((x) => <button key={x} onClick={() => ask(x)}>{x}</button>)}
        </div>

        <div style={{ marginTop: 20 }}>
          {busy && !hits && <div style={{ display: "grid", gap: 10 }}>{[0, 1].map((i) => <Skeleton key={i} h={92} r={12} />)}</div>}
          {error && <Notice kind="error">{error}</Notice>}
          {hits && (
            <div aria-live="polite" style={{ opacity: busy ? .55 : 1, transition: "opacity .2s" }}>
              <p className="muted" style={{ fontSize: ".86rem", marginBottom: 10, display: "flex", gap: 8, alignItems: "center" }}>
                {busy && <Loader2 size={14} className="spin" />} Best matches for “{asked.current}”
              </p>
              {hits.length === 0 && <p className="muted">No guideline matches that question.</p>}
              {hits.map((h, i) => (
                <article key={h.section + i} className={`hit${i === 0 ? " best" : ""}`}>
                  <div className="hit-head">
                    <h3>{h.section.split(">").pop()?.trim()}</h3>
                    <span className="score" title="How closely this passage matches the question">
                      <span className="bar"><span style={{ width: `${Math.round(100 * Math.min(1, h.score / best))}%` }} /></span>
                      {i === 0 ? "Best match" : "Related"}
                    </span>
                  </div>
                  <p className="faint" style={{ marginTop: 2, fontSize: ".8rem" }}>{h.section.split(">")[0]?.trim()}</p>
                  <p>{h.text.replace(/^#+.*$/gm, "").trim()}</p>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="section panel accordion" aria-labelledby="all-title">
        <div className="panel-head"><h2 id="all-title">All guidelines</h2>{sections.data && <span className="muted">{sections.data.length} sections</span>}</div>
        {sections.loading && !sections.data ? (
          <div style={{ padding: 18, display: "grid", gap: 10 }}>{[0, 1, 2, 3].map((i) => <Skeleton key={i} h={22} />)}</div>
        ) : sections.error ? (
          <Empty title="The guidelines couldn't be loaded">{sections.error}</Empty>
        ) : (
          sections.data?.map((s) => (
            <details key={s.section}>
              <summary>{s.section.split(">").pop()?.trim()} <ChevronRight size={17} /></summary>
              <div className="acc-body">{s.text.replace(/^#+.*$/gm, "").trim()}</div>
            </details>
          ))
        )}
      </section>
    </main>
  );
}
