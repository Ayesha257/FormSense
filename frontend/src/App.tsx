import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { BookOpenText, FileUp, History as HistoryIcon, LayoutDashboard, Menu, X } from "lucide-react";
import { HealthProvider, useHealth } from "./lib/hooks";
import { ToastProvider } from "./components/Toast";
import { Logo } from "./components/ui";
import Dashboard from "./pages/Dashboard";
import Check from "./pages/Check";
import Result from "./pages/Result";
import History from "./pages/History";
import Guidelines from "./pages/Guidelines";
import NotFound from "./pages/NotFound";

function SystemStatus() {
  const { health, error } = useHealth();
  const models = error ? "bad" : !health ? "wait" : health.models === "ready" ? "ok" : health.models === "error" ? "bad" : "wait";
  const modelsText = error ? "Server not reachable" : !health ? "Connecting…" :
    health.models === "ready" ? "Reader ready" : health.models === "error" ? "Reader will load on first check" : "Loading the reader…";
  return (
    <div className="status-card" aria-live="polite">
      <div className="status-row"><span className={`dot ${models}`} /><strong>{modelsText}</strong></div>
      {health && (
        <>
          <div className="status-row">
            <span className={`dot ${health.llm.available ? "ok" : "off"}`} />
            <span>{health.llm.available ? "AI explanations on" : "Explanations from the rules (no AI key)"}</span>
          </div>
          {health.handwriting_reader?.available === false && (
            <div className="status-row" title={health.handwriting_reader.error ?? ""}>
              <span className="dot off" /><span>Handwriting model not downloaded (printed-text reader only)</span>
            </div>
          )}
          {health.queue > 0 && (
            <div className="status-row"><span className="dot wait" /><span>{health.queue} form{health.queue === 1 ? "" : "s"} in the queue</span></div>
          )}
        </>
      )}
    </div>
  );
}

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const links = [
    { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
    { to: "/check", label: "Check a form", icon: FileUp },
    { to: "/history", label: "History", icon: HistoryIcon },
    { to: "/guidelines", label: "Guidelines", icon: BookOpenText },
  ];
  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label="Main navigation">
      <Link to="/" className="brand" onClick={onClose}>
        <Logo />
        <span><span className="brand-name">FormSense</span><span className="brand-tag">Admission form checker</span></span>
      </Link>
      <nav className="nav">
        {links.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} onClick={onClose}>
            <Icon size={19} /> {label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-foot"><SystemStatus /></div>
    </aside>
  );
}

function Shell() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo(0, 0); }, [loc.pathname]);
  return (
    <div className="shell">
      <Sidebar open={open} onClose={() => setOpen(false)} />
      {open && <div className="scrim" onClick={() => setOpen(false)} />}
      <div className="main">
        <header className="topbar">
          <Link to="/" className="brand"><Logo size={28} /><span className="brand-name">FormSense</span></Link>
          <button onClick={() => setOpen((o) => !o)} aria-label={open ? "Close menu" : "Open menu"}>
            {open ? <X size={24} /> : <Menu size={24} />}
          </button>
        </header>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/check" element={<Check />} />
          <Route path="/forms/:id" element={<Result />} />
          <Route path="/history" element={<History />} />
          <Route path="/guidelines" element={<Guidelines />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <HealthProvider>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </HealthProvider>
  );
}
