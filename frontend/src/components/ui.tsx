import { useEffect, useRef, type ReactNode } from "react";
import type { FieldStatus, JobStatus, Severity, Verdict } from "../lib/api";
import { FIELD_STATUS, VERDICT } from "../lib/format";

export function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 34 34" aria-hidden="true">
      <rect width="34" height="34" rx="9" fill="#2f5bea" />
      <path d="M11 7.5h9.2l5.3 5.3v13.4a1.3 1.3 0 0 1-1.3 1.3H11a1.3 1.3 0 0 1-1.3-1.3V8.8A1.3 1.3 0 0 1 11 7.5z" fill="#f3f6ff" />
      <path d="M20.2 7.5v5.3h5.3" fill="#c9d6ff" />
      <path d="M13.4 19.6l3 3 5.8-6.8" fill="none" stroke="#d7263d" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function VerdictPill({ verdict }: { verdict: Verdict }) {
  return <span className={`pill ${verdict}`}>{VERDICT[verdict].label}</span>;
}

export function JobPill({ status }: { status: JobStatus }) {
  const text = { queued: "Waiting", processing: "Checking", done: "Done", failed: "Failed" }[status];
  return <span className={`pill ${status}`}>{text}</span>;
}

export function FieldPill({ status }: { status: FieldStatus }) {
  return <span className={`pill ${status}`}>{FIELD_STATUS[status]}</span>;
}

export function SeverityPill({ severity }: { severity: Severity }) {
  const text = { error: "Must fix", warning: "Check", info: "Note" }[severity];
  return <span className={`pill ${severity}`}>{text}</span>;
}

export function Skeleton({ w = "100%", h = 14, r }: { w?: number | string; h?: number | string; r?: number }) {
  return <div className="skel" style={{ width: w, height: h, borderRadius: r }} />;
}

/** A sheet of paper with a red-pen circle: the empty-state drawing. */
export function EmptyArt() {
  return (
    <svg className="empty-art" viewBox="0 0 76 92" aria-hidden="true">
      <rect x="6" y="4" width="60" height="80" rx="6" fill="#fff" stroke="#cdd3e0" strokeWidth="1.5" />
      {[18, 30, 42, 54].map((y) => (
        <rect key={y} x="15" y={y} width={y === 42 ? 26 : 42} height="5" rx="2.5" fill="#e3e8f2" />
      ))}
      <path d="M14 63c8-6 30-7 40-1 6 4-3 10-18 10-13 0-25-3-22-9z" fill="none" stroke="#d7263d" strokeWidth="2.4" strokeLinecap="round" />
    </svg>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <EmptyArt />
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Notice({ kind, icon, children }: { kind: "info" | "warn" | "error" | "ok"; icon?: ReactNode; children: ReactNode }) {
  return <div className={`notice ${kind}`} role={kind === "error" ? "alert" : undefined}>{icon}<div>{children}</div></div>;
}

export function Switch({ checked, onChange, title, hint, disabled }: {
  checked: boolean; onChange: (v: boolean) => void; title: string; hint?: ReactNode; disabled?: boolean;
}) {
  return (
    <label className="switch" style={disabled ? { cursor: "not-allowed" } : undefined}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch-track" />
      <span className="switch-text"><strong>{title}</strong>{hint && <span>{hint}</span>}</span>
    </label>
  );
}

export function Confirm({ open, title, body, confirm, danger, onConfirm, onClose }: {
  open: boolean; title: string; body: ReactNode; confirm: string; danger?: boolean;
  onConfirm: () => void; onClose: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    ref.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        <p>{body}</p>
        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button ref={ref} className={`btn ${danger ? "btn-danger" : "btn-primary"}`} onClick={onConfirm}>{confirm}</button>
        </div>
      </div>
    </div>
  );
}
