import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

// M6 review queue: stored change events (confidence fixed at ingest, DESIGN.md §2.2), confirm/reject + reason.
export type EventRow = {
  event_id: string;
  tile_id: string;
  aoi: string;
  change_type: string;
  direction: string | null;
  confidence: number;
  earliest_supported_date: string;
  last_normal_date: string | null;
  analyst_status: string;
  geometry: GeoJSON.MultiPolygon;
  provenance: { scene_ids: string[]; model_versions: Record<string, string>; cascade_version: string };
};
type Timeline = { dates: string[]; [index: string]: (number | null)[] | string[] };
type Check = { agrees: boolean | null } | undefined;
type Obs = { acq_date: string; scene_id: string; source: string; q: number } | null;
type Img = { season: string; start_date: string; end_date: string; n_obs: number; dates: string[]; scene_ids: string[]; chip_url: string } | null;
type Lulc = { before_year: number; after_year: number; before: Record<string, number> | null; after: Record<string, number> | null; type: string | null };
type Hist = { ts: string; actor: string; action: string; payload: Record<string, unknown> };
export type FullEvent = EventRow & {
  evidence?: { timeline?: Timeline; landsat?: Check; s1?: Check; lulc?: Lulc };
  location?: { lat: number; lon: number };
  area_ha?: number;
  sensors?: string[];
  observations?: { last_normal: Obs; first_changed: Obs };
  images?: { before: Img; after: Img };
  tile_geometry?: GeoJSON.Polygon;
  history?: Hist[];
};

const COLOURS: Record<string, [string, string]> = {
  ndvi: ["#2e9e44", "vegetation"], ndbi: ["#d9534f", "built-up"], mndwi: ["#3a7bd5", "water"], bsi: ["#b07d2b", "bare soil"],
};
export const TYPE_COLOUR: Record<string, string> = {
  construction: "#ff7043", clearance: "#c0a060", water: "#3a7bd5", road: "#bbbbbb", other: "#9575cd",
};
const TYPE_TEXT: Record<string, string> = {
  construction: "Construction", clearance: "Land clearing / mining", water: "Water change", road: "Road", other: "Unclassified change",
};
export const SENSOR_TEXT: Record<string, string> = {
  sentinel2: "Sentinel-2 (optical, 10 m)", sentinel1: "Sentinel-1 (radar)", landsat: "Landsat 8/9 (optical, 30 m)",
};
const ACTION_TEXT: Record<string, string> = {
  "change.run": "Change detection run over the area",
  "change.sar_evidence": "Sentinel-1 radar cross-check",
  "change.landsat_evidence": "Landsat cross-check",
  "change.calibrate": "Confidence calibration",
  "change.lulc_evidence": "Land-cover cross-check (independent annual map)",
  "feedback.confirm": "Analyst confirmed",
  "feedback.reject": "Analyst rejected",
  export: "Exported in a signed package",
};
export const typeLabel = (t: string) => TYPE_TEXT[t] ?? t;
const TITLE: Record<string, string> = {
  "construction/appearance": "New construction", "construction/expansion": "Built-up area expanded",
  "clearance/disappearance": "Land cleared (vegetation removed)", "clearance/appearance": "Bare ground or mining appeared",
  "clearance/expansion": "Cleared area expanded", "water/appearance": "Water appeared", "water/expansion": "Water extent grew",
  "water/disappearance": "Water disappeared", "water/contraction": "Water extent shrank", "road/appearance": "New road",
  "road/expansion": "Road widened or extended",
};
/** One plain phrase for type + direction ("Land clearing disappeared" read as nonsense). */
export const eventTitle = (t: string, d: string | null) => TITLE[`${t}/${d}`] ?? (t === "other" ? "Unclassified change (low reliability)" : typeLabel(t));

/** Confidence in words. The calibrated number is not shown as a probability: on our blind held-out set, events
 * shown by default were real 71% of the time (2026-09-28), so "95% likely real" would overstate it. */
export function confBand(c: number): { text: string; cls: string } {
  if (c >= 0.85) return { text: "High confidence", cls: "hi" };
  if (c >= 0.55) return { text: "Medium confidence", cls: "mid" }; // 0.55 = events.OPERATING_CONF (shown by default)
  return { text: "Low confidence (normally hidden)", cls: "lo" };
}
export function ConfTag({ c }: { c: number }) {
  const b = confBand(c);
  return <span className={`conf ${b.cls}`} title={`calibrated score ${c.toFixed(2)}`}>{b.text}</span>;
}

const SEASON: Record<number, string> = { 12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON" };
const SEASON_TEXT: Record<string, string> = { DJF: "winter", MAM: "summer", JJAS: "monsoon", ON: "post-monsoon" };

/** IMD season key of a date, shifted by whole years ("2024-MAM"). DJF belongs to the year of its Jan/Feb. */
export function seasonOf(iso: string, addYears = 0): string {
  const d = new Date(iso);
  const m = d.getUTCMonth() + 1;
  return `${d.getUTCFullYear() + addYears + (m === 12 ? 1 : 0)}-${SEASON[m]}`;
}
export function seasonLabel(key: string | null): string {
  if (!key) return "—";
  const [y, s] = key.split("-");
  return `${SEASON_TEXT[s] ?? s} ${y}`;
}

function TimelineSvg({ tl, ev }: { tl: Timeline; ev: EventRow }) {
  const W = 380, H = 150, P = 20;
  const t = tl.dates.map((d) => new Date(d).getTime());
  const t0 = Math.min(...t), t1 = Math.max(...t);
  const x = (ms: number) => P + ((ms - t0) / Math.max(t1 - t0, 1)) * (W - 2 * P);
  const y = (v: number) => H - P - ((v + 1) / 2) * (H - 2 * P);
  const mark = (iso: string | null, colour: string) =>
    iso && <line x1={x(new Date(iso).getTime())} x2={x(new Date(iso).getTime())} y1={P} y2={H - P} stroke={colour} strokeWidth={1.5} strokeDasharray="4 3" />;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Index timeline with last normal and earliest supported dates">
      <rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} fill="none" stroke="#333" />
      <line x1={P} x2={W - P} y1={y(0)} y2={y(0)} stroke="#2a2f36" />
      {Object.entries(COLOURS).map(([k, [c]]) => {
        const vals = tl[k] as (number | null)[] | undefined;
        if (!vals) return null;
        const pts = vals.map((v, i) => (v === null ? null : `${x(t[i])},${y(v)}`)).filter(Boolean).join(" ");
        return <polyline key={k} points={pts} fill="none" stroke={c} strokeWidth={1.2} />;
      })}
      {mark(ev.last_normal_date, "#aaa")}
      {mark(ev.earliest_supported_date, "#ff3b30")}
      <text x={P} y={H - 4} fill="#7d8691" fontSize="10">{tl.dates[0]?.slice(0, 4)}</text>
      <text x={W - P} y={H - 4} fill="#7d8691" fontSize="10" textAnchor="end">{tl.dates[tl.dates.length - 1]?.slice(0, 4)}</text>
    </svg>
  );
}

function checkText(c: Check, name: string) {
  if (!c || c.agrees === null || c.agrees === undefined) return <span className="muted">{name}: no clear view, no verdict</span>;
  return c.agrees ? <span className="ok">{name}: agrees ✓</span> : <span className="bad">{name}: does not agree</span>;
}

function lulcText(l: Lulc | undefined) {
  if (!l || !l.before || !l.after) return <span className="muted">Land cover (independent annual map): no map for {l ? `${l.before_year}/${l.after_year}` : "these years"}</span>;
  const top = (f: Record<string, number>) => Object.entries(f).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([g, v]) => `${g} ${Math.round(v * 100)}%`).join(", ");
  const verdict = l.type === "stable" ? "no land-cover change" : l.type === "other" ? "some change" : `looks like ${typeLabel(l.type ?? "")}`;
  return <span>Land cover (independent annual map): {l.before_year} {top(l.before)} → {l.after_year} {top(l.after)} · <b>{verdict}</b></span>;
}

function ImageFig({ img, label }: { img: Img; label: string }) {
  if (!img) return <figure><div className="nochip big" /><figcaption>{label}: no clear image</figcaption></figure>;
  const span = img.dates.length ? `${img.dates[0]} … ${img.dates[img.dates.length - 1]}` : `${img.start_date} … ${img.end_date}`;
  return (
    <figure>
      <img src={`${img.chip_url}?size=160`} alt={label} onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} />
      <figcaption><b>{label}</b>: {seasonLabel(img.season)}<br />{img.n_obs} Sentinel-2 image{img.n_obs === 1 ? "" : "s"}, {span}</figcaption>
    </figure>
  );
}

function ObsLine({ o, label, date }: { o: Obs; label: string; date: string | null }) {
  if (!date) return <p><b>{label}:</b> —</p>;
  return (
    <p><b>{label}:</b> {date} · {o ? SENSOR_TEXT[o.source] ?? o.source : "Sentinel-2"}{o && <> · quality {o.q}<br /><span className="mono">{o.scene_id}</span></>}</p>
  );
}

/** Everything PS 2.2.5 asks for next to one candidate: before/after, location, acquisition time, sensor/source,
 * confidence and processing history. Shared by the review queue and the search-result details. */
export function EvidencePanel({ ev }: { ev: FullEvent }) {
  return (
    <>
      <div className="ev-title">
        <h2><i className="dot" style={{ background: TYPE_COLOUR[ev.change_type] }} />{eventTitle(ev.change_type, ev.direction)}</h2>
        <div><ConfTag c={ev.confidence} /> <span className="muted">first seen</span> <b>{ev.earliest_supported_date}</b></div>
      </div>
      <div className="card">
        <div className="pair">
          <ImageFig img={ev.images?.before ?? null} label="Before" />
          <span className="pair-arrow" aria-hidden="true">→</span>
          <ImageFig img={ev.images?.after ?? null} label="After" />
        </div>
      </div>
      <div className="card facts">
        <div><span className="k">Location</span><span>{ev.location ? `${ev.location.lat.toFixed(5)}° N, ${ev.location.lon.toFixed(5)}° E` : "—"}</span></div>
        <div><span className="k">Changed area</span><span>{ev.area_ha === undefined ? "—" : `${ev.area_ha < 1 ? ev.area_ha.toFixed(2) : ev.area_ha.toFixed(1)} ha`} · {ev.aoi}</span></div>
        <div><span className="k">Supported by</span><span>{(ev.sensors ?? []).map((x) => SENSOR_TEXT[x] ?? x).join(", ") || "—"}</span></div>
      </div>
      <div className="card">
        <h3>When: the actual satellite observations</h3>
        <ObsLine o={ev.observations?.last_normal ?? null} label="Last normal" date={ev.last_normal_date} />
        <ObsLine o={ev.observations?.first_changed ?? null} label="First showing the change" date={ev.earliest_supported_date} />
      </div>
      <div className="card">
        <h3>Independent cross-checks</h3>
        <p className="checks">{checkText(ev.evidence?.landsat, "Landsat (2nd optical satellite)")}<br />{checkText(ev.evidence?.s1, "Sentinel-1 radar (sees through cloud)")}<br />{lulcText(ev.evidence?.lulc)}</p>
      </div>
      {ev.evidence?.timeline && (
        <div className="card">
          <h3>What the satellite measured over time</h3>
          <TimelineSvg tl={ev.evidence.timeline} ev={ev} />
          <p className="legend-inline">
            {Object.values(COLOURS).map(([c, n]) => <span key={n}><i className="sw" style={{ background: c }} />{n}</span>)}
            <span><i className="sw dash" style={{ borderColor: "#aaa" }} />last normal</span>
            <span><i className="sw dash" style={{ borderColor: "#ff3b30" }} />change first seen</span>
          </p>
        </div>
      )}
      <details className="card">
        <summary>Processing history ({ev.history?.length ?? 0} steps) and source scenes</summary>
        <ol className="hist">
          {(ev.history ?? []).map((h, i) => (
            <li key={i}><span className="muted">{h.ts.replace("T", " ").slice(0, 16)} UTC</span> · {ACTION_TEXT[h.action] ?? h.action}
              {typeof h.payload.version === "string" && <span className="muted"> ({h.payload.version})</span>}
              {typeof h.payload.reason === "string" && h.payload.reason && <span className="muted"> “{h.payload.reason}”</span>}
            </li>
          ))}
        </ol>
        <p>Detection version {ev.provenance.cascade_version} · models {Object.entries(ev.provenance.model_versions ?? {}).map(([k, v]) => `${k}=${v}`).join(", ") || "—"}</p>
        <p>{ev.provenance.scene_ids.length} source scenes:</p>
        <p className="mono">{ev.provenance.scene_ids.join(", ")}</p>
        <p className="muted">Every step above is in the tamper-evident audit log · {ev.event_id}</p>
      </details>
    </>
  );
}

export default function Review({ onShow, drawer, mapBounds }: { onShow: (ev: EventRow | null) => void; drawer: HTMLElement | null; mapBounds: () => number[] | null }) {
  const [bbox, setBbox] = useState<number[] | null>(null);
  const [aoi, setAoi] = useState("");
  const [type, setType] = useState("");
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [showHidden, setShowHidden] = useState(false);
  const [rows, setRows] = useState<EventRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<FullEvent | null>(null);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState("");
  const [decided, setDecided] = useState(0);

  async function load() {
    const q = new URLSearchParams({ unreviewed: "true", limit: "200" });
    if (showHidden) { q.set("min_conf", "0"); q.set("all_types", "true"); } // default: operating threshold, reliable types only
    if (aoi) q.set("aoi", aoi);
    if (type) q.set("type", type);
    if (t1) q.set("t1", t1);
    if (t2) q.set("t2", t2);
    if (bbox) q.set("bbox", bbox.map((v) => v.toFixed(5)).join(","));
    setLoading(true);
    try {
      const d = await (await fetch(`/change?${q}`)).json();
      setTotal(d.total ?? d.events.length);
      setRows(d.events as EventRow[]);
    } finally { setLoading(false); }
  }
  useEffect(() => { load(); }, [aoi, type, t1, t2, showHidden, bbox]); // eslint-disable-line react-hooks/exhaustive-deps

  async function open(ev: EventRow) {
    const full = await (await fetch(`/events/${ev.event_id}`)).json();
    setSel(full);
    setReason("");
    setMsg("");
    onShow(full);
    document.getElementById(`e-${ev.event_id}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  async function decide(decision: "confirm" | "reject") {
    if (!sel) return;
    const r = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target: { type: "event", id: sel.event_id }, decision, reason }),
    });
    if (!r.ok) return setMsg(`Error ${r.status}`);
    const i = rows.findIndex((e) => e.event_id === sel.event_id);
    const rest = rows.filter((e) => e.event_id !== sel.event_id);
    setRows(rest);
    setTotal((t) => t - 1);
    setDecided((n) => n + 1);
    setMsg(`${sel.event_id} ${decision === "confirm" ? "confirmed as a real change" : "rejected as a false alarm"} · audited`);
    if (rest[i]) open(rest[i]);
    else { setSel(null); onShow(null); }
  }

  useEffect(() => { // analyst shortcuts: J/K move, C confirm, R reject (never while typing)
    const k = (e: KeyboardEvent) => {
      const el = e.target;
      if ((el instanceof Element && el.closest("input, select, textarea")) || e.ctrlKey || e.metaKey || e.altKey) return;
      const i = sel ? rows.findIndex((x) => x.event_id === sel.event_id) : -1;
      if (e.key === "j" && rows[i + 1]) open(rows[i + 1]);
      else if (e.key === "k" && i > 0) open(rows[i - 1]);
      else if (e.key === "c" && sel) decide("confirm");
      else if (e.key === "r" && sel) decide("reject");
      else if (e.key === "Escape" && sel) { setSel(null); onShow(null); }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  });

  const counts = rows.reduce<Record<string, number>>((a, e) => ({ ...a, [e.change_type]: (a[e.change_type] ?? 0) + 1 }), {});
  const pos = sel ? rows.findIndex((x) => x.event_id === sel.event_id) + 1 : 0;
  return (
    <div className="review">
      <div className="panel-head">
        <h2>Change review</h2>
        <p className="muted">Detected automatically, most confident first. Open a change, check the evidence, then confirm or reject it.</p>
      </div>
      <div className="filters">
        <label>Area<select value={aoi} onChange={(e) => setAoi(e.target.value)}>
          {["", "dholera", "hasdeo", "navimumbai"].map((a) => <option key={a} value={a}>{a || "All areas"}</option>)}
        </select></label>
        <label>Type<select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="">All types</option>
          {Object.keys(TYPE_TEXT).map((t) => <option key={t} value={t}>{typeLabel(t)}</option>)}
        </select></label>
        <label>Changed from<input type="date" value={t1} onChange={(e) => setT1(e.target.value)} /></label>
        <label>To<input type="date" value={t2} onChange={(e) => setT2(e.target.value)} /></label>
      </div>
      <div className="area-row">
        {bbox ? <>
          <span className="pill">Area: current map view ({bbox[1].toFixed(3)}°, {bbox[0].toFixed(3)}° → {bbox[3].toFixed(3)}°, {bbox[2].toFixed(3)}°)</span>
          <button className="link" onClick={() => setBbox(null)}>clear</button>
        </> : <button className="link" onClick={() => setBbox(mapBounds())} title="Frame any area on the map, then limit the queue to it">Limit to the area shown on the map</button>}
      </div>
      <label className="check"><input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} /> Include hidden changes (low confidence or unclassified)</label>
      <div className="queue-stats">
        <span><b>{total.toLocaleString("en-IN")}</b> to review{total > rows.length ? ` · top ${rows.length} shown` : ""}</span>
        {decided > 0 && <span className="ok">{decided} decided this session</span>}
      </div>
      <div className="type-counts">{Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([t, n]) => <span key={t}><i className="dot" style={{ background: TYPE_COLOUR[t] }} />{typeLabel(t)} {n}</span>)}</div>

      {sel && drawer && createPortal(
        <section className="drawer">
          <div className="drawer-head">
            <div><b>Change {pos || ""}</b><span className="muted"> of {rows.length} · {sel.event_id}</span></div>
            <button className="icon-btn close" title="Close (Esc)" onClick={() => { setSel(null); onShow(null); }}>×</button>
          </div>
          <div className="drawer-body"><EvidencePanel ev={sel} /></div>
          <div className="drawer-actions">
            <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (optional), e.g. seasonal flooding, crop harvest" aria-label="Reason" />
            <div className="row">
              <button className="primary grow" onClick={() => decide("confirm")} title="Shortcut: C">✓ Real change <kbd>C</kbd></button>
              <button className="danger grow" onClick={() => decide("reject")} title="Shortcut: R">✗ False alarm <kbd>R</kbd></button>
            </div>
            <p className="fine">{msg || "J / K: next / previous change · Esc: close"}</p>
          </div>
        </section>,
        drawer,
      )}

      {loading && !rows.length ? <p className="muted"><i className="spinner" /> Loading changes…</p> : rows.length === 0 ? (
        <div className="empty"><h3>Nothing to review</h3><p className="muted">No unreviewed changes match these filters.</p></div>
      ) : (
        <ol className="results compact">
          {rows.map((e) => (
            <li key={e.event_id} id={`e-${e.event_id}`} className={sel?.event_id === e.event_id ? "sel" : ""} onClick={() => open(e)}>
              <div className="body">
                <div className="line title"><i className="dot" style={{ background: TYPE_COLOUR[e.change_type] }} />{eventTitle(e.change_type, e.direction)}</div>
                <div className="line meta"><ConfTag c={e.confidence} /><span className="fine">{e.aoi} · since {e.earliest_supported_date}</span></div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
