import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { api, urls, type Sample } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { useToast } from "./Toast";
import { Skeleton } from "./ui";

const PROFILE: Record<string, string> = {
  scan_clean: "Clean scan", scan_noisy: "Noisy scan", phone_good: "Phone photo",
  phone_bad: "Blurry phone photo", phone_shadow: "Photo with shadow",
};

export default function SampleGrid({ limit = 8, options }: { limit?: number; options?: { use_trocr: boolean; explain: boolean; use_rag: boolean } }) {
  const { data, loading, error } = useAsync(() => api.samples(), []);
  const [busy, setBusy] = useState<string | null>(null);
  const nav = useNavigate();
  const toast = useToast();

  const pick = async (s: Sample) => {
    setBusy(s.name);
    try {
      const job = await api.checkSample(s.name, options ?? {});
      nav(`/forms/${job.id}`);
    } catch (e) {
      toast("error", (e as Error).message);
      setBusy(null);
    }
  };

  if (loading) {
    return (
      <div className="samples">
        {Array.from({ length: Math.min(limit, 6) }).map((_, i) => (
          <div key={i} className="sample" style={{ cursor: "default" }}>
            <Skeleton h={170} r={0} />
            <div className="sample-body"><Skeleton w="60%" /><Skeleton w="85%" h={11} /></div>
          </div>
        ))}
      </div>
    );
  }
  if (error) return <p className="muted">Sample forms could not be loaded: {error}</p>;
  if (!data || data.length === 0) {
    return (
      <p className="muted">
        No sample forms yet. In the backend folder run{" "}
        <code>python -m formsense.generator.generate --count 30 --split dev --seed 1 --out data/synthetic/dev</code>
      </p>
    );
  }
  return (
    <div className="samples">
      {data.slice(0, limit).map((s) => (
        <button key={s.name} className="sample" onClick={() => pick(s)} disabled={busy !== null}
                aria-label={`Check sample ${s.form_id}`}>
          <img src={urls.sampleThumb(s.name)} alt="" loading="lazy" />
          <div className="sample-body">
            <span className="sample-name" style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {busy === s.name && <Loader2 size={14} className="spin" />} {s.form_id}
            </span>
            <span className="sample-meta">{PROFILE[s.profile ?? ""] ?? s.profile ?? "Sample"}</span>
            <span className="sample-meta">
              {s.issues === 0 ? "Filled in correctly" : `${s.issues} known problem${s.issues === 1 ? "" : "s"}: ${s.issue_titles[0]?.toLowerCase()}`}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
}
