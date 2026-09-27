import { useEffect, useState } from "react";

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
type Full = EventRow & { evidence?: { timeline?: Timeline; landsat?: Check; s1?: Check } };

const COLOURS: Record<string, [string, string]> = {
  ndvi: ["#2e9e44", "vegetation"], ndbi: ["#d9534f", "built-up"], mndwi: ["#3a7bd5", "water"], bsi: ["#b07d2b", "bare soil"],
};
export const TYPE_COLOUR: Record<string, string> = {
  construction: "#ff7043", clearance: "#c0a060", water: "#3a7bd5", road: "#bbbbbb", other: "#9575cd",
};
const TYPE_TEXT: Record<string, string> = {
  construction: "Construction", clearance: "Land clearing / mining", water: "Water change", road: "Road", other: "Unclassified change",
};
const DIRECTION_TEXT: Record<string, string> = {
  appearance: "appeared", disappearance: "disappeared", expansion: "expanded", contraction: "shrank",
};
export const typeLabel = (t: string) => TYPE_TEXT[t] ?? t;
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
    <svg width={W} height={H} role="img" aria-label="Index timeline with last normal and earliest supported dates">
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
  if (!c || c.agrees === null || c.agrees === undefined) return <span className="muted">{name}: no clear view</span>;
  return c.agrees ? <span className="ok">{name}: agrees ✓</span> : <span className="bad">{name}: does not agree</span>;
}

export default function Review({ onShow }: { onShow: (ev: EventRow | null) => void }) {
  const [aoi, setAoi] = useState("");
  const [type, setType] = useState("");
  const [showHidden, setShowHidden] = useState(false);
  const [rows, setRows] = useState<EventRow[]>([]);
  const [total, setTotal] = useState(0);
  const [sel, setSel] = useState<Full | null>(null);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState("");

  async function load() {
    const q = new URLSearchParams();
    if (showHidden) q.set("min_conf", "0"); // default: the API's operating threshold (low-confidence events stay hidden)
    if (aoi) q.set("aoi", aoi);
    if (type) q.set("type", type);
    const r = await fetch(`/change?${q}`);
    const d = await r.json();
    const open = (d.events as EventRow[]).filter((e) => e.analyst_status === "unreviewed");
    setTotal(open.length);
    setRows(open.slice(0, 200));
  }
  useEffect(() => { load(); }, [aoi, type, showHidden]); // eslint-disable-line react-hooks/exhaustive-deps

  async function open(ev: EventRow) {
    const r = await fetch(`/events/${ev.event_id}`);
    const full = await r.json();
    setSel(full);
    setReason("");
    setMsg("");
    onShow(full);
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
    setMsg(`${sel.event_id} ${decision === "confirm" ? "confirmed" : "rejected"} and saved in the audit log.`);
    if (rest[i]) open(rest[i]);
    else { setSel(null); onShow(null); }
  }

  return (
    <div className="review">
      <p className="sub">Changes found automatically in the archive, most likely first. Open one, check the evidence, then confirm or reject it.</p>
      <div className="row">
        <select value={aoi} onChange={(e) => setAoi(e.target.value)} aria-label="Area of interest">
          {["", "dholera", "hasdeo", "navimumbai"].map((a) => <option key={a} value={a}>{a ? `Area: ${a}` : "All areas"}</option>)}
        </select>
        <select value={type} onChange={(e) => setType(e.target.value)} aria-label="Change type">
          <option value="">All change types</option>
          {Object.keys(TYPE_TEXT).map((t) => <option key={t} value={t}>{typeLabel(t)}</option>)}
        </select>
      </div>
      <label className="check"><input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} /> Also show low-confidence changes (normally hidden)</label>
      <p className="status">{total} changes to review{total > rows.length ? ` (showing the top ${rows.length})` : ""}</p>

      {sel && (
        <section className="detail">
          <h2><i className="dot" style={{ background: TYPE_COLOUR[sel.change_type] }} />{typeLabel(sel.change_type)}{sel.direction ? ` ${DIRECTION_TEXT[sel.direction] ?? sel.direction}` : ""}</h2>
          <p className="big-line"><b>{Math.round(sel.confidence * 100)}%</b> likely real · first seen <b>{sel.earliest_supported_date}</b> · {sel.aoi}</p>
          <div className="pair">
            {sel.last_normal_date && <figure><img src={`/chips/${sel.tile_id}/${seasonOf(sel.last_normal_date)}.png?size=160`} alt="before" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} /><figcaption>Before: {seasonLabel(seasonOf(sel.last_normal_date))}</figcaption></figure>}
            {sel.last_normal_date && (() => {
              const after = seasonOf(sel.last_normal_date, 1 + new Date(sel.earliest_supported_date).getUTCFullYear() - new Date(sel.last_normal_date).getUTCFullYear());
              return <figure><img src={`/chips/${sel.tile_id}/${after}.png?size=160`} alt="after" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} /><figcaption>After: {seasonLabel(after)}</figcaption></figure>;
            })()}
          </div>
          <h3>Independent cross-checks</h3>
          <p className="checks">{checkText(sel.evidence?.landsat, "Landsat (2nd optical satellite)")}<br />{checkText(sel.evidence?.s1, "Sentinel-1 radar (sees through cloud)")}</p>
          {sel.evidence?.timeline && <>
            <h3>What the satellite measured over time</h3>
            <TimelineSvg tl={sel.evidence.timeline} ev={sel} />
            <p className="legend-inline">
              {Object.values(COLOURS).map(([c, n]) => <span key={n}><i className="sw" style={{ background: c }} />{n}</span>)}
              <span><i className="sw dash" style={{ borderColor: "#aaa" }} />last normal ({sel.last_normal_date ?? "—"})</span>
              <span><i className="sw dash" style={{ borderColor: "#ff3b30" }} />change first seen</span>
            </p>
          </>}
          <p className="muted">Based on {sel.provenance.scene_ids.length} satellite scenes · {sel.event_id} · detection version {sel.provenance.cascade_version}</p>
          <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (optional), e.g. seasonal flooding, crop harvest" aria-label="Reason" />
          <div className="row">
            <button className="primary" onClick={() => decide("confirm")}>✓ Real change</button>
            <button className="ghost" onClick={() => decide("reject")}>✗ False alarm</button>
          </div>
          <p className="status">{msg}</p>
        </section>
      )}

      <ol className="results compact">
        {rows.map((e) => (
          <li key={e.event_id} className={sel?.event_id === e.event_id ? "sel" : ""} onClick={() => open(e)}>
            <div className="body">
              <div className="line"><i className="dot" style={{ background: TYPE_COLOUR[e.change_type] }} /><b>{typeLabel(e.change_type)}</b> · {e.aoi}</div>
              <div className="line meta">
                <span className="bar"><span style={{ width: `${e.confidence * 100}%` }} /></span> {Math.round(e.confidence * 100)}% likely real · first seen {e.earliest_supported_date}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
