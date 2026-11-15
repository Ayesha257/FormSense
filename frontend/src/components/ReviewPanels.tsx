import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, BookOpen, Check, CheckCircle2, ChevronRight, Download, Lightbulb, Loader2,
         PenLine, RotateCcw, Sparkles, Undo2, Wand2 } from "lucide-react";
import { api, urls, type Correction, type FieldRow, type FormResult } from "../lib/api";
import { useHealth } from "../lib/hooks";
import { fullDate, plural } from "../lib/format";
import { useToast } from "./Toast";
import { Confirm, FieldPill, Notice, SeverityPill, Switch } from "./ui";

/* ================================================================ Problems */

export function IssueList({ result, active, onPick, onFix }: {
  result: FormResult; active: string | null; onPick: (key: string | null) => void; onFix: (field: string) => void;
}) {
  const refs = useRef<Record<string, HTMLDivElement | null>>({});
  useEffect(() => {
    if (active) refs.current[active]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active]);

  if (result.issues.length === 0) {
    return (
      <div className="empty">
        <div className="stamp" style={{ color: "var(--green)", fontSize: "1rem", marginBottom: 8 }}>NO PROBLEMS</div>
        <h3>Every required item is filled in and the answers agree</h3>
        <p>FormSense found nothing to fix. Check the answers tab if you want to confirm what was read.</p>
      </div>
    );
  }
  return (
    <div role="list">
      {result.issues.map((iss) => (
        <div key={iss.key} ref={(el) => (refs.current[iss.key] = el)} role="listitem"
             className={`issue ${iss.severity}${active === iss.key ? " active" : ""}`}
             onClick={() => onPick(active === iss.key ? null : iss.key)}>
          <div className="issue-n" aria-hidden="true">{iss.n}</div>
          <div style={{ minWidth: 0 }}>
            <div className="issue-head">
              <h3>{iss.title}</h3>
              <SeverityPill severity={iss.severity} />
            </div>
            <div className="issue-fields">{iss.field_labels.join(" · ")}</div>
            <p className="issue-msg">{iss.message}</p>
            {iss.explanation && iss.explanation !== iss.message && (
              <div className="issue-ai"><Sparkles size={15} /><span>{iss.explanation}</span></div>
            )}
            {iss.citation && (
              <details className="cite" onClick={(e) => e.stopPropagation()}>
                <summary><BookOpen size={14} /> Guideline: {iss.citation.source.split("§").pop()?.split(">").pop()?.trim()}</summary>
                <p>{iss.citation.text}</p>
              </details>
            )}
            <div className="issue-actions">
              <button className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); onFix(iss.fields[0]); }}>
                <PenLine size={14} /> Correct the answer
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ================================================================ Answers (review + correct) */

interface Group { title: string; rows: FieldRow[] }

function effective(f: FieldRow, corrections: Record<string, Correction>) {
  const c = corrections[f.name];
  if (f.type === "checkbox") return { checked: c?.checked ?? !!f.checked };
  if (f.type === "signature") return { signed: c?.signed ?? f.status !== "missing" };
  return { value: c?.value ?? f.raw_text ?? "" };
}

export function FieldEditor({ result, focus, onFocus, onUpdated }: {
  result: FormResult; focus: string | null; onFocus: (name: string | null) => void; onUpdated: (r: FormResult) => void;
}) {
  const toast = useToast();
  const [edits, setEdits] = useState<Record<string, Correction>>({});
  const [saving, setSaving] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const refs = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => setEdits({}), [result.id]);
  useEffect(() => {
    if (!focus) return;
    const el = refs.current[focus];
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
    el?.querySelector<HTMLElement>("input, textarea, button")?.focus({ preventScroll: true });
  }, [focus]);

  const groups = useMemo(() => {
    const out: Group[] = [];
    for (const f of result.fields) {
      let g = out.find((x) => x.title === f.section_title);
      if (!g) out.push((g = { title: f.section_title, rows: [] }));
      g.rows.push(f);
    }
    return out;
  }, [result.fields]);

  const base = result.corrections || {};
  const changed = Object.keys(edits).length;
  const setEdit = (f: FieldRow, c: Correction) => {
    const orig = effective(f, base);
    const same = JSON.stringify(orig) === JSON.stringify({ ...orig, ...c });
    setEdits((e) => {
      const next = { ...e };
      if (same) delete next[f.name]; else next[f.name] = c;
      return next;
    });
  };
  const current = (f: FieldRow) => ({ ...effective(f, base), ...edits[f.name] });

  const save = async (corrections: Record<string, Correction>, msg?: string) => {
    setSaving(true);
    try {
      const r = await api.correct(result.id, corrections);
      onUpdated(r);
      setEdits({});
      const n = r.issues.length;
      toast("success", msg ?? (n === 0 ? "Re-checked: no problems left." : `Re-checked: ${plural(n, "problem")} left.`));
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const renderGroupedChecks = (rows: FieldRow[]) => {
    const byGroup: Record<string, FieldRow[]> = {};
    rows.forEach((f) => (byGroup[f.group ?? f.name] ??= []).push(f));
    return Object.entries(byGroup).map(([g, members]) => {
      const ticked = members.filter((m) => current(m).checked).length;
      const status = ticked === 1 ? "ok" : ticked === 0 ? "missing" : "invalid";
      const anyChanged = members.some((m) => edits[m.name]);
      const anyCorrected = members.some((m) => m.corrected);
      return (
        <div key={g} ref={(el) => members.forEach((m) => (refs.current[m.name] = el))}
             className={`frow${members.some((m) => m.name === focus) ? " focus" : ""}`}>
          <div className="frow-head">
            <span className="frow-label">{members[0].group_label ?? g}{anyCorrected && <span className="tag"><PenLine size={12} /> corrected</span>}</span>
            <FieldPill status={status} />
          </div>
          <div className="choice-row" role="group" aria-label={`${members[0].group_label ?? g}: tick exactly one`}>
            {members.map((m) => {
              const on = !!current(m).checked;
              return (
                <button key={m.name} type="button" className={`choice${on ? " on" : ""}`} aria-pressed={on}
                        onClick={() => setEdit(m, { checked: !on })} onFocus={() => onFocus(m.name)}>
                  <span className="box">{on && <Check size={12} strokeWidth={3} />}</span>{m.label}
                </button>
              );
            })}
          </div>
          <div className="frow-foot">{anyChanged ? <span style={{ color: "var(--pen)" }}>Changed, not re-checked yet</span> : <span>Exactly one box must be ticked.</span>}</div>
        </div>
      );
    });
  };

  return (
    <div>
      <Notice kind="info" icon={<Lightbulb size={17} />}>
        Fix anything the reader got wrong, then re-check. The rules run again on your answers.
      </Notice>
      {groups.map((g) => (
        <div className="fsection" key={g.title} style={{ marginTop: 18 }}>
          <h3>{g.title}</h3>
          {renderGroupedChecks(g.rows.filter((f) => f.type === "checkbox"))}
          {g.rows.filter((f) => f.type !== "checkbox").map((f) => {
            const cur = current(f);
            const isChanged = !!edits[f.name];
            const low = f.confidence != null && f.confidence < 0.8;
            return (
              <div key={f.name} ref={(el) => (refs.current[f.name] = el)} className={`frow${focus === f.name ? " focus" : ""}`}>
                <div className="frow-head">
                  <label className="frow-label" htmlFor={`f-${f.name}`}>
                    {f.label}{f.required && <span className="faint">*</span>}
                    {f.corrected && <span className="tag"><PenLine size={12} /> corrected</span>}
                  </label>
                  <FieldPill status={f.status} />
                </div>
                {(f.type !== "signature" || f.status !== "missing") && (
                  <img className="frow-crop" src={urls.crop(result.id, f.name)} alt={`Close-up of ${f.label} on the form`} loading="lazy" />
                )}
                {f.type === "signature" ? (
                  <div className="choice-row">
                    {[true, false].map((v) => (
                      <button key={String(v)} id={v ? `f-${f.name}` : undefined} type="button"
                              className={`choice${cur.signed === v ? " on" : ""}`} aria-pressed={cur.signed === v}
                              onClick={() => setEdit(f, { signed: v })} onFocus={() => onFocus(f.name)}>
                        <span className="box">{cur.signed === v && <Check size={12} strokeWidth={3} />}</span>
                        {v ? "Signed" : "Not signed"}
                      </button>
                    ))}
                  </div>
                ) : f.kind === "multiline" ? (
                  <textarea id={`f-${f.name}`} className={`input${isChanged ? " changed" : ""}`} rows={2} value={cur.value}
                            placeholder="Empty" onFocus={() => onFocus(f.name)}
                            onChange={(e) => setEdit(f, { value: e.target.value })} />
                ) : (
                  <input id={`f-${f.name}`} className={`input${isChanged ? " changed" : ""}`} value={cur.value}
                         placeholder={f.hint ?? "Empty"} onFocus={() => onFocus(f.name)}
                         onChange={(e) => setEdit(f, { value: e.target.value })} />
                )}
                <div className="frow-foot">
                  {f.confidence != null && !f.corrected && (
                    <span className={`conf${low ? " low" : ""}`} title="How sure the reader was">
                      <span className="bar"><span style={{ width: `${Math.round(f.confidence * 100)}%` }} /></span>
                      {Math.round(f.confidence * 100)}% sure
                    </span>
                  )}
                  {f.value && f.value !== f.raw_text && f.type === "date" && <span>Understood as {f.value}</span>}
                  {f.notes.map((n) => <span key={n}>{n}</span>)}
                  {isChanged && <span style={{ color: "var(--pen)" }}>Changed, not re-checked yet</span>}
                </div>
              </div>
            );
          })}
        </div>
      ))}

      {(changed > 0 || saving) && (
        <div className="savebar glass">
          <span><b>{plural(changed, "change")}</b> <span className="muted">to check</span></span>
          <div className="actions">
            <button className="btn btn-ghost" onClick={() => setEdits({})} disabled={saving}><Undo2 size={16} /> Discard</button>
            <button className="btn btn-primary" onClick={() => save({ ...base, ...edits })} disabled={saving}>
              {saving ? <Loader2 size={16} className="spin" /> : <RotateCcw size={16} />} Re-check form
            </button>
          </div>
        </div>
      )}
      {changed === 0 && Object.keys(base).length > 0 && (
        <div style={{ marginTop: 16, display: "flex", justifyContent: "flex-end" }}>
          <button className="btn btn-ghost btn-sm" onClick={() => setConfirmReset(true)} disabled={saving}>
            <Undo2 size={15} /> Undo all corrections
          </button>
        </div>
      )}
      <Confirm open={confirmReset} title="Undo all corrections?" confirm="Undo corrections"
               body="The form goes back to what FormSense read from the photo, and the rules run again."
               onClose={() => setConfirmReset(false)}
               onConfirm={() => { setConfirmReset(false); save({}, "Corrections removed. Showing what was read from the photo."); }} />
    </div>
  );
}

/* ================================================================ Checklist */

export function ChecklistTab({ result }: { result: FormResult }) {
  const sections: { title: string; rows: FormResult["checklist"] }[] = [];
  for (const r of result.checklist) {
    let s = sections.find((x) => x.title === r.section);
    if (!s) sections.push((s = { title: r.section, rows: [] }));
    s.rows.push(r);
  }
  const ok = result.checklist.filter((r) => r.status === "ok").length;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
        <span className="muted">{ok} of {result.checklist.length} items look right</span>
        <a className="btn btn-secondary btn-sm" href={urls.csv(result.id)} download><Download size={15} /> Download checklist</a>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table className="check-table">
          <thead><tr><th>Item</th><th>Status</th><th>What was found</th></tr></thead>
          <tbody>
            {sections.map((s) => (
              <FragmentRows key={s.title} title={s.title} rows={s.rows} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FragmentRows({ title, rows }: { title: string; rows: FormResult["checklist"] }) {
  return (
    <>
      <tr className="sec"><td colSpan={3}>{title}</td></tr>
      {rows.map((r) => (
        <tr key={r.item}>
          <td>{r.item}</td>
          <td><FieldPill status={r.status} /></td>
          <td className={r.status === "ok" ? "" : "muted"}>{r.detail || "—"}</td>
        </tr>
      ))}
    </>
  );
}

/* ================================================================ Explanation */

export function ExplanationTab({ result, onUpdated }: { result: FormResult; onUpdated: (r: FormResult) => void }) {
  const { health } = useHealth();
  const toast = useToast();
  const [useRag, setUseRag] = useState(result.explanation?.use_rag ?? result.options.use_rag ?? true);
  const [busy, setBusy] = useState(false);
  const exp = result.explanation;

  const run = async () => {
    setBusy(true);
    try {
      const r = await api.explain(result.id, useRag);
      onUpdated(r);
      toast("success", r.explanation?.source === "llm" ? "Explanation written." : "Explanation written from the rules.");
    } catch (e) {
      toast("error", (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const button = (
    <button className="btn btn-primary" onClick={run} disabled={busy}>
      {busy ? <Loader2 size={16} className="spin" /> : <Wand2 size={16} />} {exp ? "Write it again" : "Write an explanation"}
    </button>
  );

  return (
    <div className="explain">
      {!exp ? (
        <div className="empty" style={{ padding: "28px 8px" }}>
          <h3>No explanation yet</h3>
          <p>{health?.llm.available ? "An AI model will explain the problems in plain English for the applicant." :
            "No AI key is set, so the explanation will be written from the rules. Add GROQ_API_KEY to backend/.env for AI explanations."}</p>
          {button}
        </div>
      ) : (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <span className="source">
              {exp.source === "llm" ? <><Sparkles size={14} /> Written by {health?.llm.model ?? "the AI model"}</> : <><BookOpen size={14} /> Written from the rules</>}
              {exp.use_rag && " · with guidelines"} · {fullDate(exp.created_at)}
            </span>
          </div>
          {result.explanation_stale && (
            <Notice kind="warn" icon={<AlertTriangle size={17} />}>The answers were corrected after this was written. Write it again to match.</Notice>
          )}
          {exp.error && <Notice kind="warn" icon={<AlertTriangle size={17} />}>{exp.error}</Notice>}
          <p className="explain-summary">{exp.summary}</p>
          {result.issues.some((i) => i.explanation) && (
            <div>
              {result.issues.filter((i) => i.explanation).map((i) => (
                <div key={i.key} style={{ display: "grid", gridTemplateColumns: "30px 1fr", gap: 8, padding: "8px 0", borderTop: "1px solid var(--line)" }}>
                  <span className="issue-n" style={{ fontSize: "1.5rem", color: i.severity === "warning" ? "var(--amber)" : "var(--red)" }}>{i.n}</span>
                  <span>{i.explanation}</span>
                </div>
              ))}
            </div>
          )}
          {exp.overall_recommendation && (
            <div className="explain-rec"><CheckCircle2 size={18} /><span>{exp.overall_recommendation}</span></div>
          )}
        </>
      )}
      {exp && (
        <div style={{ borderTop: "1px solid var(--line)", paddingTop: 12, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <Switch checked={useRag} onChange={setUseRag} title="Use the guidelines" hint="Give the model the guideline passages." />
          {button}
        </div>
      )}
      {!exp && (
        <div style={{ display: "flex", justifyContent: "center" }}>
          <Switch checked={useRag} onChange={setUseRag} title="Use the guidelines" hint="Give the model the guideline passages." />
        </div>
      )}
      <p className="faint" style={{ fontSize: ".8rem", display: "flex", gap: 6, alignItems: "center" }}>
        <ChevronRight size={13} /> CNIC, phone and email are masked before anything is sent to the AI model.
      </p>
    </div>
  );
}
