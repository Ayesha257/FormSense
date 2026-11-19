import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AlertTriangle, FileText, Info, Loader2, RefreshCcw, ShieldCheck, X } from "lucide-react";
import { uploadForm } from "../lib/api";
import { useHealth } from "../lib/hooks";
import { fileSize, todayIso } from "../lib/format";
import Dropzone, { validateFile } from "../components/Dropzone";
import SampleGrid from "../components/SampleGrid";
import { useToast } from "../components/Toast";
import { Notice, Switch } from "../components/ui";

export default function Check() {
  const loc = useLocation();
  const nav = useNavigate();
  const toast = useToast();
  const { health, error: serverError } = useHealth();
  const [file, setFile] = useState<File | null>(() => (loc.state as { file?: File } | null)?.file ?? null);
  const [opts, setOpts] = useState({ use_trocr: true, explain: true, use_rag: true, today: todayIso(), page: 1 });
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  // the hero dropzone passes the file through navigation state; clear it so a refresh starts clean
  useEffect(() => { if (loc.state) nav(loc.pathname, { replace: true, state: null }); }, []); // eslint-disable-line

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  const isPdf = file?.name.toLowerCase().endsWith(".pdf");
  const isImage = file && !isPdf && /\.(png|jpe?g|webp|bmp)$/i.test(file.name);

  const choose = (f: File) => {
    const err = validateFile(f, health);
    if (err) { setError(err); return; }
    setError(null);
    setFile(f);
  };

  const submit = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    abort.current = new AbortController();
    try {
      const job = await uploadForm(file, opts, setProgress, abort.current.signal);
      toast("success", "Uploaded. Checking the form now.");
      nav(`/forms/${job.id}`);
    } catch (e) {
      const msg = (e as Error).message;
      setProgress(null);
      if (msg !== "Upload cancelled.") setError(msg);
    }
  };

  const uploading = progress !== null;
  const set = <K extends keyof typeof opts>(k: K, v: (typeof opts)[K]) => setOpts((o) => ({ ...o, [k]: v }));

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>Check a form</h1>
          <p>Upload one filled admission form. A flat, well-lit photo or a scan gives the best results.</p>
        </div>
      </div>

      {serverError && (
        <div style={{ marginBottom: 16 }}>
          <Notice kind="error" icon={<AlertTriangle size={18} />}>{serverError}</Notice>
        </div>
      )}

      <div className="check-layout">
        <section className="panel" aria-label="Form file">
          {!file ? (
            <div className="panel-pad">
              <Dropzone onFile={choose} onError={setError} />
            </div>
          ) : (
            <div className="file-card">
              <div className="file-preview">
                {isImage && preview ? <img src={preview} alt="Preview of the selected form" /> :
                  isPdf && preview ? <object data={`${preview}#toolbar=0&navpanes=0&view=FitH`} type="application/pdf" aria-label="PDF preview"><FileText size={40} /></object> :
                  <FileText size={40} />}
              </div>
              <div style={{ display: "grid", gap: 10, minWidth: 0 }}>
                <div>
                  <div className="file-name">{file.name}</div>
                  <div className="muted" style={{ fontSize: ".86rem" }}>{fileSize(file.size)} · {isPdf ? "PDF document" : "Image"}</div>
                </div>
                {uploading ? (
                  <div style={{ display: "grid", gap: 6 }}>
                    <div className="progress" role="progressbar" aria-valuenow={Math.round((progress ?? 0) * 100)} aria-valuemin={0} aria-valuemax={100}>
                      <span style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
                    </div>
                    <div className="muted" style={{ fontSize: ".84rem" }}>
                      {progress! < 1 ? `Uploading… ${Math.round(progress! * 100)}%` : "Uploaded. Starting the check…"}
                    </div>
                  </div>
                ) : (
                  <div className="actions">
                    <button className="btn btn-primary btn-lg" onClick={submit} disabled={!!serverError}>
                      <ShieldCheck size={19} /> Check this form
                    </button>
                    <button className="btn btn-secondary btn-lg" onClick={() => { setFile(null); setError(null); }}>
                      <RefreshCcw size={17} /> Choose another file
                    </button>
                  </div>
                )}
                {uploading && progress! < 1 && (
                  <button className="btn btn-ghost btn-sm" style={{ justifySelf: "start" }} onClick={() => abort.current?.abort()}>
                    <X size={15} /> Cancel upload
                  </button>
                )}
              </div>
            </div>
          )}
          {error && <div style={{ padding: "0 22px 22px" }}><Notice kind="error" icon={<AlertTriangle size={18} />}>{error}</Notice></div>}
        </section>

        <aside className="panel panel-pad" aria-label="Check settings">
          <h3 style={{ marginBottom: 6 }}>Settings</h3>
          <div className="options">
            <Switch checked={opts.use_trocr} onChange={(v) => set("use_trocr", v)} title="Handwriting reader"
                    hint="A second model re-reads messy handwriting. Slower the first time." />
            <Switch checked={opts.explain} onChange={(v) => set("explain", v)} title="Write an explanation"
                    hint={health?.llm.available ? `Plain-English summary by ${health.llm.model ?? "the AI model"}.` : "Written from the rules, because no AI key is set."} />
            <Switch checked={opts.use_rag} onChange={(v) => set("use_rag", v)} disabled={!opts.explain} title="Use the guidelines"
                    hint="Give the explanation the guideline passages behind each problem." />
            <div className="divider" />
            <label className="field-label" htmlFor="today">Check dates against</label>
            <input id="today" className="input" type="date" value={opts.today} onChange={(e) => set("today", e.target.value)} />
            <span className="faint" style={{ fontSize: ".8rem", marginTop: 4 }}>Used for age and “date in the future” rules.</span>
            {isPdf && (
              <>
                <label className="field-label" htmlFor="page" style={{ marginTop: 12 }}>Page of the PDF to check</label>
                <input id="page" className="input" type="number" min={1} value={opts.page}
                       onChange={(e) => set("page", Math.max(1, Number(e.target.value) || 1))} />
              </>
            )}
          </div>
          {health?.handwriting_reader?.available === false && opts.use_trocr && (
            <div style={{ marginTop: 14 }}>
              <Notice kind="info" icon={<Info size={17} />}>
                The handwriting model isn't downloaded on this computer, so forms are read with the main reader only.
                To add it, restart the server once with internet and without <code>HF_HUB_OFFLINE</code>.
              </Notice>
            </div>
          )}
          {health && health.models !== "ready" && (
            <div style={{ marginTop: 14 }}>
              <Notice kind="info" icon={health.models === "error" ? <Info size={17} /> : <Loader2 size={17} className="spin" />}>
                {health.models === "error" ? "The reader will load when you check the first form." : "The reader is still loading. Your form will wait in the queue for a moment."}
              </Notice>
            </div>
          )}
        </aside>
      </div>

      <section className="section" aria-labelledby="try-title">
        <div className="section-title">
          <div>
            <h2 id="try-title">No form at hand? Try a sample</h2>
            <p>Each sample was generated with known problems, so you can compare what FormSense finds.</p>
          </div>
        </div>
        <SampleGrid limit={24} options={{ use_trocr: opts.use_trocr, explain: opts.explain, use_rag: opts.use_rag }} />
      </section>
    </main>
  );
}
