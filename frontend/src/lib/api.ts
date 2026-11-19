// Typed client for the FormSense API (backend/api/main.py).

export type Verdict = "pass" | "review" | "fail" | "unaligned";
export type FieldStatus = "ok" | "missing" | "unreadable" | "invalid";
export type Severity = "error" | "warning" | "info";
export type JobStatus = "queued" | "processing" | "done" | "failed";
export type Stage = "queued" | "reading" | "analyzing" | "explaining" | "done" | "failed";
export type BBox = [number, number, number, number];

export interface Health {
  ok: boolean;
  version: string;
  models: "idle" | "loading" | "ready" | "error";
  models_error: string | null;
  queue: number;
  llm: { available: boolean; model: string | null };
  handwriting_reader?: { available: boolean; error: string | null };
  samples: number;
  max_upload_mb: number;
  allowed: string[];
}

export interface JobOptions {
  use_trocr: boolean;
  explain: boolean;
  use_rag: boolean;
  today: string;
  page: number;
}

export interface FormSummary {
  verdict: Verdict;
  issues: Record<Severity, number>;
  fields: Record<FieldStatus, number>;
  seconds: number | null;
  top_issues: string[];
  issue_ids: string[];
}

export interface FormMeta {
  id: string;
  filename: string;
  size?: number;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  stage: Stage;
  progress?: number;
  error: string | null;
  options: JobOptions;
  sample: string | null;
  summary?: FormSummary | null;
  pages?: number;
}

export interface FieldRow {
  name: string;
  label: string;
  type: "text" | "date" | "checkbox" | "signature";
  kind: string;
  section: string | null;
  section_title: string;
  group: string | null;
  group_label: string | null;
  required: boolean;
  hint: string | null;
  status: FieldStatus;
  raw_text: string;
  value: string | null;
  confidence: number | null;
  ink: number;
  checked: boolean | null;
  bbox: BBox;
  notes: string[];
  corrected: boolean;
}

export interface IssueRow {
  n: number;
  key: string;
  id: string;
  title: string;
  type: string;
  severity: Severity;
  fields: string[];
  field_labels: string[];
  message: string;
  evidence: BBox[];
  explanation: string | null;
  citation: { source: string; text: string } | null;
}

export interface Explanation {
  summary: string;
  issues: { issue_id: string; explanation: string }[];
  overall_recommendation: string | null;
  source: "llm" | "template";
  use_rag: boolean;
  error: string | null;
  created_at: string;
}

export interface ChecklistRow {
  section: string;
  item: string;
  status: FieldStatus;
  detail: string;
}

export interface SampleTruth {
  name: string;
  profile: string | null;
  true_issues: number;
  found: number;
  false_alarms: number;
  missed: string[];
  extra: string[];
}

export interface FormResult {
  id: string;
  filename: string;
  created_at: string;
  updated_at: string;
  status: "done";
  stage: Stage;
  options: JobOptions;
  pages: number;
  page: number;
  seconds: number | null;
  template: { width: number; height: number };
  alignment: { ok: boolean; inliers?: number; matches?: number };
  verdict: Verdict;
  counts: { issues: Record<Severity, number>; fields: Record<FieldStatus, number> };
  fields: FieldRow[];
  issues: IssueRow[];
  checklist: ChecklistRow[];
  summary: string;
  explanation: Explanation | null;
  explanation_stale: boolean;
  corrections: Record<string, Correction>;
  sample: SampleTruth | null;
}

export type FormState = FormResult | (FormMeta & { status: Exclude<JobStatus, "done"> });

export interface Correction {
  value?: string;
  checked?: boolean;
  signed?: boolean;
}

export interface Sample {
  name: string;
  form_id: string;
  profile: string | null;
  reference_date: string | null;
  issues: number;
  issue_titles: string[];
}

export interface Stats {
  total: number;
  done: number;
  failed: number;
  processing: number;
  verdicts: Record<Verdict, number>;
  issues_found: number;
  avg_seconds: number | null;
  top_rules: { id: string; title: string; count: number }[];
  days: { date: string; count: number }[];
}

export interface GuidelineHit {
  section: string;
  source: string;
  text: string;
  score: number;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(0, "Can't reach the FormSense server. Make sure it is running (python -m api in the backend folder).");
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  stats: () => request<Stats>("/stats"),
  samples: () => request<{ samples: Sample[] }>("/samples").then((r) => r.samples),
  checkSample: (name: string, opts: Partial<JobOptions>) =>
    request<FormMeta>(`/samples/${encodeURIComponent(name)}/check`, { method: "POST", ...json(opts) }),
  forms: () => request<{ forms: FormMeta[] }>("/forms").then((r) => r.forms),
  form: (id: string) => request<FormState>(`/forms/${id}`),
  deleteForm: (id: string) => request<void>(`/forms/${id}`, { method: "DELETE" }),
  correct: (id: string, corrections: Record<string, Correction>) =>
    request<FormResult>(`/forms/${id}/corrections`, { method: "PUT", ...json({ corrections }) }),
  explain: (id: string, use_rag: boolean) =>
    request<FormResult>(`/forms/${id}/explain`, { method: "POST", ...json({ use_rag }) }),
  guidelines: () => request<{ sections: { section: string; source: string; text: string }[] }>("/guidelines")
    .then((r) => r.sections),
  searchGuidelines: (q: string, k = 4) =>
    request<{ results: GuidelineHit[] }>(`/guidelines/search?q=${encodeURIComponent(q)}&k=${k}`).then((r) => r.results),
};

export const urls = {
  image: (id: string, kind: "page" | "aligned" | "thumb" | "original", v?: string) =>
    `/api/forms/${id}/image/${kind}${v ? `?v=${encodeURIComponent(v)}` : ""}`,
  crop: (id: string, field: string) => `/api/forms/${id}/fields/${field}/crop.png`,
  sampleThumb: (name: string) => `/api/samples/${encodeURIComponent(name)}/thumb.jpg`,
  pdf: (id: string) => `/api/forms/${id}/report.pdf`,
  json: (id: string) => `/api/forms/${id}/report.json`,
  csv: (id: string) => `/api/forms/${id}/checklist.csv`,
};

/** Upload with real progress events (fetch has no upload progress). */
export function uploadForm(
  file: File,
  opts: Omit<JobOptions, "page"> & { page?: number },
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<FormMeta> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/forms");
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as FormMeta);
      else reject(new ApiError(xhr.status, (xhr.response && xhr.response.detail) || `Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Can't reach the FormSense server. Is it running?"));
    xhr.onabort = () => reject(new ApiError(-1, "Upload cancelled."));
    signal?.addEventListener("abort", () => xhr.abort());
    const fd = new FormData();
    fd.append("file", file);
    fd.append("use_trocr", String(opts.use_trocr));
    fd.append("explain", String(opts.explain));
    fd.append("use_rag", String(opts.use_rag));
    fd.append("today", opts.today);
    fd.append("page", String(opts.page ?? 1));
    xhr.send(fd);
  });
}
