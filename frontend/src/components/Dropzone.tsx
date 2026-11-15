import { useRef, useState, type DragEvent } from "react";
import { UploadCloud } from "lucide-react";
import type { Health } from "../lib/api";
import { useHealth } from "../lib/hooks";

const FALLBACK_EXT = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"];

export function validateFile(file: File, health: Health | null): string | null {
  const allowed = health?.allowed ?? FALLBACK_EXT;
  const maxMb = health?.max_upload_mb ?? 20;
  const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
  if (!allowed.includes(ext)) return `“${file.name}” is not a form scan. Upload a JPG, PNG or PDF.`;
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxMb * 1024 * 1024) return `That file is larger than ${maxMb} MB. Upload a smaller scan or photo.`;
  return null;
}

export default function Dropzone({ onFile, onError, variant = "light", compact }: {
  onFile: (f: File) => void; onError: (msg: string) => void; variant?: "hero" | "light"; compact?: boolean;
}) {
  const { health } = useHealth();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const depth = useRef(0);

  const take = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    if (files.length > 1) onError("Drop one form at a time. FormSense checks forms one by one.");
    const f = files[0];
    const err = validateFile(f, health);
    if (err) onError(err); else onFile(f);
  };

  const drag = (e: DragEvent, kind: "enter" | "leave" | "over" | "drop") => {
    e.preventDefault();
    if (kind === "enter") { depth.current += 1; setOver(true); }
    if (kind === "leave") { depth.current -= 1; if (depth.current <= 0) setOver(false); }
    if (kind === "drop") { depth.current = 0; setOver(false); take(e.dataTransfer.files); }
  };

  return (
    <div
      className={`dropzone${variant === "light" ? " light" : ""}${over ? " over" : ""}`}
      style={compact ? { minHeight: 200 } : undefined}
      role="button" tabIndex={0} aria-label="Upload a form: drop a file here or press Enter to browse"
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), input.current?.click())}
      onDragEnter={(e) => drag(e, "enter")} onDragLeave={(e) => drag(e, "leave")}
      onDragOver={(e) => drag(e, "over")} onDrop={(e) => drag(e, "drop")}
    >
      <div className="drop-icon"><UploadCloud size={28} /></div>
      <h3>{over ? "Drop to check this form" : "Drop a filled form here"}</h3>
      <p>or <u>browse your files</u> · a scan or a phone photo works</p>
      <div className="formats"><span>JPG</span><span>PNG</span><span>PDF</span><span>up to {health?.max_upload_mb ?? 20} MB</span></div>
      <input ref={input} type="file" hidden accept={(health?.allowed ?? FALLBACK_EXT).join(",")}
             onChange={(e) => { take(e.target.files); e.target.value = ""; }} />
    </div>
  );
}
