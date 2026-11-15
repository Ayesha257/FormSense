import type { FieldStatus, Stage, Verdict } from "./api";

export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  const s = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.round(h / 24);
  if (d < 7) return `${d} day${d === 1 ? "" : "s"} ago`;
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function fullDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export const VERDICT: Record<Verdict, { label: string; stamp: string; title: string }> = {
  pass: { label: "Ready", stamp: "ACCEPTED", title: "Ready to accept" },
  review: { label: "Review", stamp: "REVIEW", title: "Check the warnings" },
  fail: { label: "Needs fixing", stamp: "RETURN", title: "Needs fixing before it can be accepted" },
  unaligned: { label: "Unreadable photo", stamp: "RETAKE", title: "The photo could not be matched to the form" },
};

export const FIELD_STATUS: Record<FieldStatus, string> = {
  ok: "Looks right",
  missing: "Empty",
  unreadable: "Hard to read",
  invalid: "Not valid",
};

export const STAGE_TEXT: Record<Stage, string> = {
  queued: "Waiting for its turn",
  reading: "Opening the file",
  analyzing: "Reading every box",
  explaining: "Writing the explanation",
  done: "Done",
  failed: "Failed",
};

export function plural(n: number, one: string, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}

export function todayIso(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** "**Signature**: message" -> plain text (the backend summary uses light markdown). */
export function stripMd(s: string) {
  return s.replace(/\*\*(.+?)\*\*/g, "$1");
}
