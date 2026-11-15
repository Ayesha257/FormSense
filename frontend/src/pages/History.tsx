import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Plus, RefreshCcw, Search, Trash2 } from "lucide-react";
import { api, urls, type FormMeta } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { fullDate, timeAgo } from "../lib/format";
import { useToast } from "../components/Toast";
import { Confirm, Empty, JobPill, Skeleton, VerdictPill } from "../components/ui";

type Filter = "all" | "fail" | "review" | "pass" | "failed";
const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "fail", label: "Needs fixing" },
  { key: "review", label: "Review" },
  { key: "pass", label: "Ready" },
  { key: "failed", label: "Couldn't check" },
];

function matches(f: FormMeta, filter: Filter) {
  if (filter === "all") return true;
  if (filter === "failed") return f.status === "failed";
  return f.status === "done" && f.summary?.verdict === filter;
}

export default function History() {
  const { data, loading, error, reload, setData } = useAsync(() => api.forms(), []);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [toDelete, setToDelete] = useState<FormMeta | null>(null);
  const nav = useNavigate();
  const toast = useToast();

  const rows = useMemo(() => (data ?? []).filter((f) => matches(f, filter) &&
    f.filename.toLowerCase().includes(q.trim().toLowerCase())), [data, q, filter]);
  const counts = useMemo(() => Object.fromEntries(FILTERS.map(({ key }) => [key, (data ?? []).filter((f) => matches(f, key)).length])), [data]);

  const remove = async (f: FormMeta) => {
    setToDelete(null);
    try {
      await api.deleteForm(f.id);
      setData((d) => (d ?? []).filter((x) => x.id !== f.id));
      toast("success", `Deleted ${f.filename}.`);
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p>Every form checked on this computer, newest first. Uploads stay on this machine.</p>
        </div>
        <div className="actions">
          <button className="btn btn-secondary" onClick={reload}><RefreshCcw size={16} /> Refresh</button>
          <Link className="btn btn-primary" to="/check"><Plus size={16} /> Check a form</Link>
        </div>
      </div>

      <div className="toolbar">
        <div className="search">
          <Search size={18} />
          <input className="input" placeholder="Search by file name" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search by file name" />
        </div>
        <div className="filters" role="group" aria-label="Filter by result">
          {FILTERS.map((f) => (
            <button key={f.key} className={filter === f.key ? "on" : ""} onClick={() => setFilter(f.key)} aria-pressed={filter === f.key}>
              {f.label} <span className="tnum" style={{ opacity: .7 }}>{counts[f.key] ?? 0}</span>
            </button>
          ))}
        </div>
      </div>

      <section className="panel" style={{ overflow: "hidden" }}>
        {loading && !data ? (
          <div style={{ padding: 16, display: "grid", gap: 12 }}>{[0, 1, 2, 3].map((i) => <Skeleton key={i} h={50} />)}</div>
        ) : error ? (
          <Empty title="The history couldn't be loaded" action={<button className="btn btn-primary" onClick={reload}>Try again</button>}>{error}</Empty>
        ) : (data ?? []).length === 0 ? (
          <Empty title="Nothing checked yet" action={<Link className="btn btn-primary" to="/check"><Plus size={16} /> Check your first form</Link>}>
            Checked forms appear here with their results, so you can reopen, correct or download them later.
          </Empty>
        ) : rows.length === 0 ? (
          <Empty title="No forms match" action={<button className="btn btn-secondary" onClick={() => { setQ(""); setFilter("all"); }}>Clear the search</button>}>
            Try another name or filter.
          </Empty>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="history-table">
              <thead>
                <tr><th>Form</th><th>Result</th><th className="hide-sm">Problems</th><th className="hide-sm">Checked</th><th aria-label="Actions" /></tr>
              </thead>
              <tbody>
                {rows.map((f) => (
                  <tr key={f.id} onClick={() => nav(`/forms/${f.id}`)} tabIndex={0}
                      onKeyDown={(e) => e.key === "Enter" && nav(`/forms/${f.id}`)}>
                    <td>
                      <div className="history-file">
                        <img className="thumb" src={urls.image(f.id, "thumb", f.updated_at)} alt="" loading="lazy"
                             onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} />
                        <div style={{ minWidth: 0 }}>
                          <div className="recent-name">{f.filename}</div>
                          <div className="recent-meta">{f.sample ? "Sample form" : "Uploaded"}</div>
                        </div>
                      </div>
                    </td>
                    <td>{f.status === "done" && f.summary ? <VerdictPill verdict={f.summary.verdict} /> : <JobPill status={f.status} />}</td>
                    <td className="hide-sm">
                      {f.summary ? (
                        <span className="issue-dots tnum">
                          <span><span className="dot" style={{ background: "var(--red)" }} />{f.summary.issues.error}</span>
                          <span><span className="dot" style={{ background: "var(--amber)" }} />{f.summary.issues.warning}</span>
                        </span>
                      ) : <span className="faint">—</span>}
                    </td>
                    <td className="hide-sm muted" title={fullDate(f.created_at)} style={{ whiteSpace: "nowrap" }}>{timeAgo(f.created_at)}</td>
                    <td style={{ textAlign: "right" }}>
                      <button className="btn btn-ghost btn-sm btn-icon" aria-label={`Delete ${f.filename}`}
                              disabled={f.status === "queued" || f.status === "processing"}
                              onClick={(e) => { e.stopPropagation(); setToDelete(f); }}>
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <Confirm open={!!toDelete} danger title="Delete this check?" confirm="Delete"
               body={<>“{toDelete?.filename}” and its results will be removed from the history.</>}
               onClose={() => setToDelete(null)} onConfirm={() => toDelete && remove(toDelete)} />
    </main>
  );
}
