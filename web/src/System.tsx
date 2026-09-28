import { useEffect, useState } from "react";

// PS 2.3: the report must state indexed area, scenes/tiles, build time, storage, query latency, hardware. Live from /stats.
type Stats = {
  aois: Record<string, { area_km2: number; tiles: number; scenes: Record<string, number> }>;
  area_km2: number;
  scenes: number;
  tiles_per_date: number;
  storage_gb: Record<string, number>;
  index: Record<string, number> | null;
  latency_ms: Record<string, number> | null;
  auth: string;
  weights_error: string | null;
  build: { file: string; wall_s: number; stages_s: Record<string, number> } | null;
  hardware: { cpu: string; ram_gb: number; gpu: string | null };
  evaluation: {
    file: string;
    overall: Record<string, number | null>;
    per_type: Record<string, { n: number; precision: number | null; recall: number | null }>;
    earliest_date_median_err_days: number | null;
  } | null;
};
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
const TYPE_NAME: Record<string, string> = { construction: "Construction", clearance: "Land clearing / mining", water: "Water change", road: "Road", other: "Unclassified" };

const n = (v: number | undefined) => (v === undefined ? "—" : v.toLocaleString("en-IN"));
const hms = (s: number) => `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
const STAGE_TEXT: Record<string, string> = {
  scenes: "Read + quality-check scenes", composite: "Seasonal composites", embed_clip: "Text-search vectors (CLIP)",
  ard: "Spectral indices per scene", cascade: "Change detection (per pixel)", events: "Group changes into events", events_commit: "Save events", postgis_commit: "Save to database", qdrant_upsert: "Save vectors to index", embed_eo: "Look-alike vectors (Prithvi)", eo_vec: "Look-alike vectors (Prithvi)",
  sar_evidence: "Radar cross-check", landsat_load: "Landsat load", landsat_evidence: "Landsat cross-check",
  calibrate: "Calibration", context: "Water/road distances", landcover: "Land-cover shares", lulc_evidence: "Land-cover cross-check", eo_vec_store: "Save look-alike vectors",
};

export default function System() {
  const [s, setS] = useState<Stats | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetch("/stats").then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then(setS).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <div className="alert">Could not load system figures: {err}</div>;
  if (!s) return <p className="muted"><i className="spinner" /> Loading…</p>;
  const ix = s.index ?? {};
  const ev = s.evaluation;
  return (
    <div className="system">
      <div className="panel-head">
        <h2>System</h2>
        <p className="muted">What is indexed, how well and how fast it works: the figures the evaluation report must state (PS 2.3), read live.</p>
      </div>
      <div className="kpis">
        <div><b>{n(s.area_km2)} km²</b><span>area indexed</span></div>
        <div><b>{n(s.scenes)}</b><span>satellite scenes</span></div>
        <div><b>{n(ix.text_vec)}</b><span>searchable tile-seasons</span></div>
        <div><b>{n(ix.change_events)}</b><span>change events stored</span></div>
      </div>
      {ev && (
        <div className="card">
          <h3>Measured quality: blind held-out change cases</h3>
          <div className="kpis three">
            <div><b>{pct(ev.overall.precision)}</b><span>precision</span></div>
            <div><b>{pct(ev.overall.recall)}</b><span>recall</span></div>
            <div><b>{ev.overall.f1 === null || ev.overall.f1 === undefined ? "—" : ev.overall.f1.toFixed(2)}</b><span>F1</span></div>
          </div>
          <table className="why"><thead><tr><td>Type</td><td>precision</td><td>recall</td></tr></thead><tbody>
            {Object.entries(ev.per_type).filter(([, v]) => v.n > 0).map(([t, v]) => <tr key={t}><td>{TYPE_NAME[t] ?? t}</td><td>{pct(v.precision)}</td><td>{pct(v.recall)}</td></tr>)}
          </tbody></table>
          <p className="fine">{ev.overall.n} human-labelled cases the system was never tuned on · false alarms on no-change places {pct(ev.overall.false_alarm_rate)} · earliest-date error median {ev.earliest_date_median_err_days ?? "—"} days · {ev.file}</p>
        </div>
      )}
      <div className="card">
        <h3>Areas</h3>
        <table className="why"><tbody>
          {Object.entries(s.aois).map(([k, a]) => (
            <tr key={k}><td>{k}</td><td>{a.area_km2} km² · {n(a.tiles)} tiles · S2 {a.scenes.sentinel2 ?? 0} · S1 {a.scenes.sentinel1 ?? 0} · Landsat {a.scenes.landsat ?? 0}</td></tr>
          ))}
        </tbody></table>
      </div>
      <div className="card">
        <h3>Query speed (last request)</h3>
        <table className="why"><tbody>
          {Object.entries(s.latency_ms ?? {}).map(([k, v]) => <tr key={k}><td>{k.replace("_", " ")}</td><td>{v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`}</td></tr>)}
          {!s.latency_ms && <tr><td colSpan={2} className="muted">No query since the service started. Run a search first.</td></tr>}
        </tbody></table>
      </div>
      <div className="card">
        <h3>Build time: full index from an empty database</h3>
        {s.build ? <>
          <p><b>{hms(s.build.wall_s)}</b> <span className="muted">wall clock ({s.build.file}). Timed steps below; not every step has its own timer.</span></p>
          <table className="why"><tbody>
            {Object.entries(s.build.stages_s).sort((a, b) => b[1] - a[1]).map(([k, v]) => <tr key={k}><td>{STAGE_TEXT[k] ?? k}</td><td>{v >= 60 ? `${Math.round(v / 60)} min` : `${v} s`}</td></tr>)}
          </tbody></table>
        </> : <p className="muted">No timed full build recorded.</p>}
      </div>
      <div className="card">
        <h3>Storage and hardware</h3>
        <table className="why"><tbody>
          <tr><td>Raw imagery extracts</td><td>{s.storage_gb.raw} GB</td></tr>
          <tr><td>Database (PostGIS)</td><td>{ix.postgis_gb ?? "—"} GB</td></tr>
          <tr><td>Model weights</td><td>{s.storage_gb.models} GB</td></tr>
          <tr><td>Free disk</td><td>{s.storage_gb.disk_free} GB</td></tr>
          <tr><td>CPU / RAM</td><td>{s.hardware.cpu} · {s.hardware.ram_gb} GB</td></tr>
          <tr><td>GPU</td><td>{s.hardware.gpu ?? "used by the worker service"}</td></tr>
        </tbody></table>
      </div>
      <div className="card">
        <h3>Security and integrity</h3>
        <table className="why"><tbody>
          <tr><td>Network</td><td className="ok">offline: no internet at runtime</td></tr>
          <tr><td>Model weights</td><td>{s.weights_error ? <span className="bad">{s.weights_error}</span> : <span className="ok">all SHA-256 checksums verified</span>}</td></tr>
          <tr><td>Login</td><td>{s.auth}</td></tr>
          <tr><td>Audit log</td><td>hash-chained, verifiable at /audit/verify</td></tr>
        </tbody></table>
      </div>
    </div>
  );
}
