import { Link, useNavigate } from "react-router-dom";
import { ArrowUpRight, BookCheck, Check, Clock3, FileSearch, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import { api, urls, type FormMeta, type Stats } from "../lib/api";
import { useAsync, useHealth } from "../lib/hooks";
import { plural, timeAgo } from "../lib/format";
import Dropzone from "../components/Dropzone";
import SampleGrid from "../components/SampleGrid";
import { useToast } from "../components/Toast";
import { Empty, JobPill, Skeleton, VerdictPill } from "../components/ui";

function StatTiles({ stats }: { stats: Stats | null }) {
  const passRate = stats && stats.done ? Math.round((100 * stats.verdicts.pass) / stats.done) : null;
  const tiles = [
    { label: "Forms checked", icon: FileSearch, value: stats?.done ?? 0, note: stats?.processing ? `${stats.processing} being checked now` : "in the history" },
    { label: "Problems caught", icon: ShieldCheck, value: stats?.issues_found ?? 0, note: "errors and warnings" },
    { label: "Average check", icon: Clock3, value: stats?.avg_seconds != null ? `${stats.avg_seconds}s` : "–", note: "per form, on this computer" },
    { label: "Ready to accept", icon: BookCheck, value: passRate != null ? `${passRate}%` : "–", note: "of checked forms had no problems" },
  ];
  return (
    <div className="stats">
      {tiles.map(({ label, icon: Icon, value, note }) => (
        <div className="stat" key={label}>
          <div className="stat-label"><Icon size={16} /> {label}</div>
          {stats ? <div className="stat-value tnum">{value}</div> : <Skeleton w={70} h={34} />}
          <div className="stat-note">{note}</div>
        </div>
      ))}
    </div>
  );
}

function Recent({ forms }: { forms: FormMeta[] | null }) {
  if (!forms) {
    return (
      <div className="recent">
        {[0, 1, 2].map((i) => (
          <div key={i} className="recent-item"><Skeleton w={46} h={62} r={6} /><div style={{ display: "grid", gap: 6 }}><Skeleton w="55%" /><Skeleton w="35%" h={11} /></div><span /></div>
        ))}
      </div>
    );
  }
  if (forms.length === 0) {
    return <Empty title="No forms checked yet">Drop a form above or try one of the sample forms below. Every check is saved here.</Empty>;
  }
  return (
    <div className="recent">
      {forms.slice(0, 6).map((f) => {
        const n = f.summary ? f.summary.issues.error + f.summary.issues.warning : 0;
        return (
          <Link key={f.id} to={`/forms/${f.id}`} className="recent-item">
            <img className="thumb" src={urls.image(f.id, "thumb", f.updated_at)} alt="" loading="lazy"
                 onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} />
            <div style={{ minWidth: 0 }}>
              <div className="recent-name">{f.filename}</div>
              <div className="recent-meta">
                {timeAgo(f.created_at)}
                {f.status === "done" && f.summary && ` · ${n === 0 ? "no problems" : plural(n, "problem")}`}
                {f.status === "failed" && " · could not be checked"}
              </div>
            </div>
            {f.status === "done" && f.summary ? <VerdictPill verdict={f.summary.verdict} /> : <JobPill status={f.status} />}
          </Link>
        );
      })}
    </div>
  );
}

export default function Dashboard() {
  const nav = useNavigate();
  const toast = useToast();
  const { health } = useHealth();
  const stats = useAsync(() => api.stats(), []);
  const forms = useAsync(() => api.forms(), []);
  const s = stats.data;
  const maxDay = Math.max(1, ...(s?.days.map((d) => d.count) ?? [1]));
  const maxRule = Math.max(1, ...(s?.top_rules.map((r) => r.count) ?? [1]));

  return (
    <main className="page">
      <section className="hero" aria-labelledby="hero-title">
        <div>
          <h1 id="hero-title">Check an admission form in seconds.</h1>
          <p className="hero-lede">
            Upload a scan or a phone photo. FormSense reads every box, applies the admission rules,
            and marks exactly what the applicant has to fix.
          </p>
          <ul className="hero-points">
            <li><Check size={17} /> Finds empty boxes, unreadable handwriting and missing signatures</li>
            <li><Check size={17} /> Catches answers that contradict each other, like a CNIC that doesn't match the gender</li>
            <li><Check size={17} /> Quotes the guideline behind every problem{health?.llm.available ? " and explains it in plain English" : ""}</li>
          </ul>
        </div>
        <Dropzone variant="hero" onError={(m) => toast("error", m)} onFile={(file) => nav("/check", { state: { file } })} />
      </section>

      <section className="section" aria-label="Numbers">
        <StatTiles stats={s} />
      </section>

      <div className="grid-2 section">
        <section className="panel" aria-labelledby="recent-title">
          <div className="panel-head">
            <h2 id="recent-title">Recent checks</h2>
            <Link to="/history" className="btn btn-ghost btn-sm">All history <ArrowUpRight size={15} /></Link>
          </div>
          <Recent forms={forms.data} />
        </section>

        <section className="panel" aria-labelledby="problems-title">
          <div className="panel-head"><h2 id="problems-title">Most common problems</h2></div>
          {!s ? (
            <div className="bars">{[0, 1, 2].map((i) => <Skeleton key={i} h={22} />)}</div>
          ) : s.top_rules.length === 0 ? (
            <Empty title="Nothing to count yet">The problems FormSense finds most often will show up here.</Empty>
          ) : (
            <div className="bars">
              {s.top_rules.map((r) => (
                <div className="bar-row" key={r.id}>
                  <div className="bar-row-head"><span>{r.title}</span><span className="tnum muted">{r.count}</span></div>
                  <div className="bar"><span style={{ width: `${(100 * r.count) / maxRule}%` }} /></div>
                </div>
              ))}
              <div style={{ marginTop: 8 }}>
                <div className="bar-row-head" style={{ marginBottom: 8 }}><span className="muted">Checks in the last 7 days</span></div>
                <div className="sparkline" aria-label="Checks per day, last 7 days">
                  {s.days.map((d, i) => (
                    <div key={d.date} className={i === s.days.length - 1 ? "today" : ""} title={`${d.date}: ${d.count}`}
                         style={{ height: `${Math.max(8, (100 * d.count) / maxDay)}%` }} />
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="section panel panel-pad" aria-labelledby="how-title">
        <div className="section-title"><h2 id="how-title">How a form is checked</h2></div>
        <div className="steps">
          {[
            { icon: ScanLine, t: "Line up the page", d: "The photo is straightened and matched to the blank form, so every answer box is found even on a tilted, blurry photo." },
            { icon: FileSearch, t: "Read every box", d: "Printed and handwritten answers are read, ticks and signatures are detected, and empty or unclear boxes are flagged." },
            { icon: ShieldCheck, t: "Apply the admission rules", d: "19 rules check formats and whether answers agree: CNIC and gender, marks and percentage, names and dates." },
            { icon: Sparkles, t: "Explain what to fix", d: "Each problem comes with the guideline that covers it and a plain-English explanation for the applicant." },
          ].map(({ t, d }, i) => (
            <div className="step" key={t}>
              <div className="step-n">{i + 1}</div>
              <h3>{t}</h3>
              <p>{d}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="samples-title">
        <div className="section-title">
          <div>
            <h2 id="samples-title">Try a sample form</h2>
            <p>Generated forms with known problems, so you can see what FormSense catches.</p>
          </div>
          <Link to="/check" className="btn btn-secondary btn-sm">More samples</Link>
        </div>
        <SampleGrid limit={6} />
      </section>
    </main>
  );
}
