import { useEffect, useMemo, useRef, useState } from "react";
import { Image as ImageIcon, Maximize2, PenLine, SquareDashed, ZoomIn, ZoomOut } from "lucide-react";
import { urls, type BBox, type FormResult } from "../lib/api";
import { Skeleton } from "./ui";

export type ViewMode = "marks" | "fields" | "photo";

/** A hand-drawn loop around a box: an ellipse with a little wobble that overshoots its start, like a red pen. */
function penLoop([x, y, w, h]: BBox, seed: number): string {
  const rnd = (k: number) => {
    const s = Math.sin(seed * 12.9898 + k * 78.233) * 43758.5453;
    return s - Math.floor(s);
  };
  const cx = x + w / 2, cy = y + h / 2;
  const rx = w / 2 + 16 + rnd(1) * 8, ry = h / 2 + 14 + rnd(2) * 8;
  const start = -Math.PI * (0.6 + rnd(3) * 0.3);
  const n = 30, sweep = Math.PI * 2 + 0.45;
  const pts: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const a = start + (sweep * i) / n;
    const wob = 1 + (rnd(i + 5) - 0.5) * 0.05 + (i / n) * 0.06;   // the end drifts outward, as a real pen stroke does
    pts.push([cx + Math.cos(a) * rx * wob, cy + Math.sin(a) * ry * wob]);
  }
  // Catmull-Rom -> cubic Bezier for a smooth stroke
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  }
  return d;
}

export default function DocumentViewer({ result, mode, onMode, activeIssue, focusField, onPickIssue, onPickField }: {
  result: FormResult;
  mode: ViewMode;
  onMode: (m: ViewMode) => void;
  activeIssue: string | null;
  focusField: string | null;
  onPickIssue: (key: string) => void;
  onPickField: (name: string) => void;
}) {
  const { width: W, height: H } = result.template;
  const [zoom, setZoom] = useState(1);
  const [loaded, setLoaded] = useState(false);
  const canvas = useRef<HTMLDivElement>(null);
  const imgSrc = mode === "photo" ? urls.image(result.id, "page") : urls.image(result.id, "aligned", result.created_at);

  useEffect(() => setLoaded(false), [imgSrc]);

  // bring the selected problem (or field) into view: scrolls the viewer when zoomed in,
  // and the page when the form is fully visible but below the fold
  useEffect(() => {
    const el = canvas.current;
    if (!el || mode === "photo" || (!activeIssue && !focusField)) return;
    const sel = activeIssue ? `[data-k="${CSS.escape(activeIssue)}"]` : `[data-f="${CSS.escape(focusField!)}"]`;
    const target = el.querySelector(sel) as SVGGraphicsElement | null;
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
      return;
    }
    const box = activeIssue ? result.issues.find((i) => i.key === activeIssue)?.evidence[0]
      : result.fields.find((f) => f.name === focusField)?.bbox;
    const doc = el.querySelector(".doc") as HTMLElement | null;
    if (!box || !doc) return;
    const scale = doc.clientWidth / W;
    el.scrollTo({ top: Math.max(0, box[1] * scale - el.clientHeight / 3), behavior: "smooth" });
  }, [activeIssue, focusField, mode, loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  const marks = useMemo(() => result.issues.map((iss) => ({
    iss,
    loops: iss.evidence.map((b, k) => penLoop(b, iss.n * 7 + k)),
    label: (() => {
      // answer boxes sit to the right of their printed label, leaving a gap just left of the box;
      // boxes at the left edge (tick boxes) get the number on their right instead
      const [x, y, w, h] = iss.evidence[0] ?? [0, 0, 0, 0];
      const left = x >= 200;
      return { x: left ? x - 18 : Math.min(W - 30, x + w + 22), y: y + h / 2 + 22, anchor: (left ? "end" : "start") as "end" | "start" };
    })(),
  })), [result.issues, W]);

  return (
    <section className="panel viewer" aria-label="The form">
      <div className="viewer-bar">
        <div className="seg" role="tablist" aria-label="What to show on the form">
          <button role="tab" aria-selected={mode === "marks"} className={mode === "marks" ? "on" : ""} onClick={() => onMode("marks")}>
            <PenLine size={15} /> Problems
          </button>
          <button role="tab" aria-selected={mode === "fields"} className={mode === "fields" ? "on" : ""} onClick={() => onMode("fields")}>
            <SquareDashed size={15} /> All boxes
          </button>
          <button role="tab" aria-selected={mode === "photo"} className={mode === "photo" ? "on" : ""} onClick={() => onMode("photo")}>
            <ImageIcon size={15} /> Original
          </button>
        </div>
        <div className="zoom">
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => setZoom((z) => Math.max(0.6, +(z - 0.2).toFixed(1)))} aria-label="Zoom out"><ZoomOut size={17} /></button>
          <span className="tnum">{Math.round(zoom * 100)}%</span>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => setZoom((z) => Math.min(3, +(z + 0.2).toFixed(1)))} aria-label="Zoom in"><ZoomIn size={17} /></button>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => setZoom(1)} aria-label="Fit to width"><Maximize2 size={16} /></button>
        </div>
      </div>

      <div className="viewer-canvas" ref={canvas}>
        <div className="doc" style={{ width: `${zoom * 100}%`, maxWidth: zoom === 1 ? 760 : undefined, aspectRatio: mode === "photo" ? undefined : `${W} / ${H}` }}>
          {!loaded && <div style={{ position: "absolute", inset: 0 }}><Skeleton h="100%" r={0} /></div>}
          <img src={imgSrc} alt={mode === "photo" ? "The uploaded page" : "The form, straightened"} onLoad={() => setLoaded(true)} />
          {mode !== "photo" && loaded && (
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="group" aria-label={mode === "marks" ? "Problems marked on the form" : "Answer boxes"}>
              {mode === "fields" && result.fields.map((f) => {
                const [x, y, w, h] = f.bbox;
                const cls = f.type === "checkbox" ? (f.checked ? "ticked" : "unticked") : f.status;
                return (
                  <rect key={f.name} data-f={f.name} x={x} y={y} width={w} height={h} rx={4}
                        className={`fbox ${cls}${focusField === f.name ? " focus" : ""}`}
                        onClick={() => onPickField(f.name)}>
                    <title>{f.label}: {f.type === "checkbox" ? (f.checked ? "ticked" : "not ticked") : f.status}</title>
                  </rect>
                );
              })}
              {mode === "marks" && marks.map(({ iss, loops, label }) => {
                const active = activeIssue === iss.key;
                const dim = activeIssue !== null && !active;
                return (
                  <g key={iss.key} data-k={iss.key} className={`mark ${iss.severity}${active ? " active" : ""}${dim ? " dim" : ""}`}
                     onClick={() => onPickIssue(iss.key)} tabIndex={0} role="button"
                     aria-label={`Problem ${iss.n}: ${iss.title}`}
                     onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onPickIssue(iss.key))}>
                    {iss.evidence.map((b, k) => (
                      <rect key={`g${k}`} className="glow" x={b[0] - 10} y={b[1] - 10} width={b[2] + 20} height={b[3] + 20} rx={14} fill="transparent" />
                    ))}
                    {loops.map((d, k) => <path key={k} className="pen" d={d} />)}
                    {iss.evidence.map((b, k) => (
                      <rect key={`h${k}`} className="hit" x={b[0] - 24} y={b[1] - 24} width={b[2] + 48} height={b[3] + 48} />
                    ))}
                    <text x={label.x} y={label.y} textAnchor={label.anchor}>{iss.n}</text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>

      {mode === "fields" && (
        <div className="legend" aria-hidden="true">
          <span><i style={{ borderColor: "#1d8a57" }} />Looks right</span>
          <span><i style={{ borderColor: "#d7263d" }} />Empty</span>
          <span><i style={{ borderColor: "#c98a0b" }} />Hard to read</span>
          <span><i style={{ borderColor: "#7a4fd6" }} />Not valid</span>
          <span><i style={{ borderColor: "#2f5bea", background: "rgba(47,91,234,.2)" }} />Ticked</span>
        </div>
      )}
      {mode === "marks" && result.issues.length === 0 && (
        <div className="legend"><span>No problems to mark. Switch to “All boxes” to see every answer FormSense read.</span></div>
      )}
      {mode === "photo" && <div className="legend"><span>The page as it was uploaded, before it was straightened.</span></div>}
    </section>
  );
}
