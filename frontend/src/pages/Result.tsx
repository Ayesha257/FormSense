import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AlertTriangle, ArrowLeft, Check, Clock3, Crosshair, Download, FileJson, FileText, ListChecks,
         Loader2, PenLine, Plus, ScanLine, Sparkles, Trash2, TriangleAlert } from "lucide-react";
import { api, urls, type FormResult, type FormState } from "../lib/api";
import { fullDate, plural, STAGE_TEXT, timeAgo, VERDICT } from "../lib/format";
import DocumentViewer, { type ViewMode } from "../components/DocumentViewer";
import { ChecklistTab, ExplanationTab, FieldEditor, IssueList } from "../components/ReviewPanels";
import { useToast } from "../components/Toast";
import { Confirm, Empty, Notice, Skeleton } from "../components/ui";

const STAGES = [
  { key: "queued", title: "Uploaded", note: "Waiting for the reader" },
  { key: "reading", title: "Opening the file", note: "Turning the upload into a page" },
  { key: "analyzing", title: "Reading every box", note: "Straightening, OCR, ticks, signatures, then the 19 rules" },
  { key: "explaining", title: "Writing the explanation", note: "Guidelines and a plain-English summary" },
] as const;

function Processing({ state }: { state: FormState }) {
  const [elapsed, setElapsed] = useState(0);
  const [thumb, setThumb] = useState(false);
  useEffect(() => {
    const start = new Date(state.created_at).getTime();
    const t = window.setInterval(() => setElapsed(Math.round((Date.now() - start) / 1000)), 1000);
    return () => window.clearInterval(t);
  }, [state.created_at]);
  const idx = STAGES.findIndex((s) => s.key === state.stage);
  const explain = state.options?.explain !== false;
  return (
    <div className="panel processing" aria-live="polite">
      <div className="scan-preview">
        <img src={urls.image(state.id, "thumb", state.stage)} alt="" onLoad={() => setThumb(true)}
             onError={() => setThumb(false)} style={{ opacity: thumb ? 1 : 0, transition: "opacity .4s" }} />
        {!thumb && <div style={{ position: "absolute", inset: 0 }}><Skeleton h="100%" r={0} /></div>}
        <div className="scan-line" />
      </div>
      <div>
        <h2>Checking {state.filename}</h2>
        <p className="muted" style={{ marginTop: 6 }}>
          {state.stage === "queued" ? "Another form is being checked. Yours is next." :
            "This usually takes 5 to 20 seconds. The first form after starting the server can take a minute while the reader loads."}
        </p>
        <ol className="stages">
          {STAGES.filter((s) => explain || s.key !== "explaining").map((s, i) => {
            const st = i < idx ? "done" : i === idx ? "active" : "";
            return (
              <li key={s.key} className={st}>
                <span className="stage-icon">{st === "done" ? <Check size={16} /> : st === "active" ? <Loader2 size={16} className="spin" /> : i + 1}</span>
                <span><strong>{s.title}</strong><span>{s.note}</span></span>
              </li>
            );
          })}
        </ol>
        <p className="faint tnum" style={{ marginTop: 10, fontSize: ".84rem" }}><Clock3 size={13} style={{ verticalAlign: -2 }} /> {elapsed}s · {STAGE_TEXT[state.stage]}</p>
      </div>
    </div>
  );
}

function Verdict({ r }: { r: FormResult }) {
  const v = VERDICT[r.verdict];
  const errors = r.counts.issues.error, warnings = r.counts.issues.warning;
  const body = r.verdict === "pass" ? "Every required item is filled in and the answers agree with each other."
    : r.verdict === "unaligned" ? "The page could not be matched to the blank form, so the boxes may be in the wrong place. Retake the photo flat and well lit."
    : r.verdict === "review" ? `${plural(warnings, "warning")} to look at. Nothing is definitely wrong.`
    : `${plural(errors, "error")}${warnings ? ` and ${plural(warnings, "warning")}` : ""} to fix before this form can be accepted.`;
  return (
    <section className={`verdict ${r.verdict}`} aria-label="Verdict">
      <span className="stamp" aria-hidden="true">{v.stamp}</span>
      <div>
        <h2>{v.title}</h2>
        <p>{body}</p>
      </div>
      <div className="verdict-counts">
        <div><b className="tnum">{r.counts.fields.ok}</b><span>look right</span></div>
        <div><b className="tnum" style={{ color: errors ? "var(--red)" : undefined }}>{errors}</b><span>errors</span></div>
        <div><b className="tnum" style={{ color: warnings ? "var(--amber)" : undefined }}>{warnings}</b><span>warnings</span></div>
      </div>
    </section>
  );
}

type Tab = "issues" | "fields" | "checklist" | "explain";

export default function Result() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const toast = useToast();
  const [state, setState] = useState<FormState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("issues");
  const [mode, setMode] = useState<ViewMode>("marks");
  const [activeIssue, setActiveIssue] = useState<string | null>(null);
  const [focusField, setFocusField] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const timer = useRef<number>();
  const announced = useRef(false);

  const load = useCallback(async () => {
    try {
      const s = await api.form(id);
      setState(s);
      setError(null);
      if (s.status === "queued" || s.status === "processing") {
        timer.current = window.setTimeout(load, 1200);
      } else if (s.status === "done" && announced.current) {
        toast("success", "Check finished.");
      }
      announced.current = s.status !== "done";
    } catch (e) {
      setError((e as Error).message);
    }
  }, [id, toast]);

  useEffect(() => {
    announced.current = false;
    setState(null); setActiveIssue(null); setFocusField(null); setTab("issues");
    load();
    return () => window.clearTimeout(timer.current);
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  const remove = async () => {
    setConfirmDelete(false);
    try {
      await api.deleteForm(id);
      toast("success", "Deleted from the history.");
      nav("/history");
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };

  if (error) {
    return (
      <main className="page">
        <div className="panel">
          <Empty title="This form can't be shown" action={<Link className="btn btn-primary" to="/history">Open the history</Link>}>{error}</Empty>
        </div>
      </main>
    );
  }
  if (!state) {
    return (
      <main className="page">
        <Skeleton w="40%" h={34} /><div style={{ height: 14 }} /><Skeleton h={88} r={18} />
        <div className="workspace" style={{ marginTop: 18 }}><Skeleton h={560} r={18} /><Skeleton h={560} r={18} /></div>
      </main>
    );
  }

  const back = <Link to="/history" className="btn btn-ghost btn-sm" style={{ marginBottom: 12, marginLeft: -10 }}><ArrowLeft size={15} /> History</Link>;

  if (state.status === "queued" || state.status === "processing") {
    return <main className="page">{back}<Processing state={state} /></main>;
  }
  if (state.status === "failed") {
    return (
      <main className="page">
        {back}
        <div className="panel panel-pad" style={{ display: "grid", gap: 14, maxWidth: 680 }}>
          <h1 style={{ fontSize: "1.6rem" }}>{state.filename} could not be checked</h1>
          <Notice kind="error" icon={<AlertTriangle size={18} />}>{state.error ?? "Something went wrong while checking this form."}</Notice>
          <div className="actions">
            <Link className="btn btn-primary" to="/check"><Plus size={16} /> Upload it again</Link>
            <button className="btn btn-danger" onClick={() => setConfirmDelete(true)}><Trash2 size={16} /> Delete</button>
          </div>
        </div>
        <Confirm open={confirmDelete} danger title="Delete this check?" confirm="Delete" body="It will be removed from the history."
                 onClose={() => setConfirmDelete(false)} onConfirm={remove} />
      </main>
    );
  }

  const r = state as FormResult;
  const pickIssue = (key: string | null) => {
    setActiveIssue(key);
    setFocusField(null);
    if (key) { setMode("marks"); setTab("issues"); }
  };
  const pickField = (name: string) => {
    setFocusField(name);
    setActiveIssue(null);
    setTab("fields");
  };
  const fixField = (name: string) => { setTab("fields"); setFocusField(name); setMode("fields"); };
  const errorsN = r.counts.issues.error + r.counts.issues.warning;

  return (
    <main className="page">
      {back}
      <div className="result-head">
        <div>
          <div className="result-title"><h1>{r.filename}</h1></div>
          <div className="result-sub">
            <span title={fullDate(r.created_at)}><Clock3 size={14} /> Checked {timeAgo(r.created_at)}</span>
            {r.seconds != null && <span><ScanLine size={14} /> {r.seconds}s to check</span>}
            {r.alignment.ok && r.alignment.inliers != null && <span><Crosshair size={14} /> Matched to the form at {r.alignment.inliers} points</span>}
            {Object.keys(r.corrections).length > 0 && <span><PenLine size={14} /> {plural(Object.keys(r.corrections).length, "answer")} corrected</span>}
            {r.pages > 1 && <span><FileText size={14} /> Page {r.page} of {r.pages}</span>}
          </div>
        </div>
        <div className="actions">
          <a className="btn btn-ink" href={urls.pdf(r.id)} download><Download size={16} /> Report (PDF)</a>
          <a className="btn btn-secondary" href={urls.json(r.id)} download><FileJson size={16} /> JSON</a>
          <Link className="btn btn-secondary" to="/check"><Plus size={16} /> New check</Link>
          <button className="btn btn-ghost btn-icon" onClick={() => setConfirmDelete(true)} aria-label="Delete this check"><Trash2 size={17} /></button>
        </div>
      </div>

      <Verdict r={r} />

      {r.sample && (
        <div className="truth" role="note">
          <TriangleAlert size={16} />
          <span>Sample with a known answer: <b>{r.sample.true_issues}</b> real problem{r.sample.true_issues === 1 ? "" : "s"} in it.
            FormSense found <b>{r.sample.found}</b>{r.sample.missed.length ? `, missed ${r.sample.missed.length}` : ""}
            {r.sample.false_alarms ? `, and raised ${plural(r.sample.false_alarms, "extra flag")} (usually a misread letter you can correct)` : ", with no false alarms"}.</span>
        </div>
      )}

      <div className="workspace">
        <DocumentViewer result={r} mode={mode} onMode={setMode} activeIssue={activeIssue} focusField={focusField}
                        onPickIssue={(k) => pickIssue(activeIssue === k ? null : k)} onPickField={pickField} />

        <section className="panel" aria-label="Review">
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "issues"} className={tab === "issues" ? "on" : ""} onClick={() => setTab("issues")}>
              <TriangleAlert size={16} /> Problems <span className={`tab-count${errorsN ? " red" : ""}`}>{r.issues.length}</span>
            </button>
            <button role="tab" aria-selected={tab === "fields"} className={tab === "fields" ? "on" : ""} onClick={() => setTab("fields")}>
              <PenLine size={16} /> Answers
            </button>
            <button role="tab" aria-selected={tab === "checklist"} className={tab === "checklist" ? "on" : ""} onClick={() => setTab("checklist")}>
              <ListChecks size={16} /> Checklist
            </button>
            <button role="tab" aria-selected={tab === "explain"} className={tab === "explain" ? "on" : ""} onClick={() => setTab("explain")}>
              <Sparkles size={16} /> Explanation
            </button>
          </div>
          <div className="tab-body">
            {tab === "issues" && (
              <>
                {r.explanation?.summary && r.issues.length > 0 && !r.explanation_stale && (
                  <p className="muted" style={{ marginBottom: 14, fontSize: ".92rem" }}>{r.explanation.summary}</p>
                )}
                <IssueList result={r} active={activeIssue} onPick={pickIssue} onFix={fixField} />
              </>
            )}
            {tab === "fields" && <FieldEditor result={r} focus={focusField} onFocus={setFocusField} onUpdated={(x) => { setState(x); setActiveIssue(null); }} />}
            {tab === "checklist" && <ChecklistTab result={r} />}
            {tab === "explain" && <ExplanationTab result={r} onUpdated={setState} />}
          </div>
        </section>
      </div>

      <Confirm open={confirmDelete} danger title="Delete this check?" confirm="Delete"
               body="The uploaded file, the results and any corrections will be removed from the history."
               onClose={() => setConfirmDelete(false)} onConfirm={remove} />
    </main>
  );
}
