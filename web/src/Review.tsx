import { useEffect, useState } from "react";

// M6 review queue: stored change events (confidence fixed at ingest, DESIGN.md §2.2), confirm/reject + reason.
export type EventRow = {
  event_id: string;
  tile_id: string;
  aoi: string;
  change_type: string;
  confidence: number;
  earliest_supported_date: string;
  last_normal_date: string | null;
  analyst_status: string;
  geometry: GeoJSON.MultiPolygon;
  provenance: { scene_ids: string[]; model_versions: Record<string, string>; cascade_version: string };
};
type Timeline = { dates: string[]; [index: string]: (number | null)[] | string[] };
const COLOURS: Record<string, string> = { ndvi: "#2e9e44", ndbi: "#d9534f", mndwi: "#3a7bd5", bsi: "#b07d2b" };
const SEASON: Record<number, string> = { 12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON" };

function season(iso: string, addYears = 0): string {
  const d = new Date(iso);
  const m = d.getUTCMonth() + 1;
  return `${d.getUTCFullYear() + addYears + (m === 12 ? 1 : 0)}-${SEASON[m]}`;
}

function TimelineSvg({ tl, ev }: { tl: Timeline; ev: EventRow }) {
  const W = 360, H = 150, P = 20;
  const t = tl.dates.map((d) => new Date(d).getTime());
  const t0 = Math.min(...t), t1 = Math.max(...t);
  const x = (ms: number) => P + ((ms - t0) / Math.max(t1 - t0, 1)) * (W - 2 * P);
  const y = (v: number) => H - P - ((v + 1) / 2) * (H - 2 * P);
  const mark = (iso: string | null, colour: string) =>
    iso && <line x1={x(new Date(iso).getTime())} x2={x(new Date(iso).getTime())} y1={P} y2={H - P} stroke={colour} strokeWidth={1.5} />;
  return (
    <svg width={W} height={H} role="img" aria-label="Index timeline with last normal and earliest supported dates">
      <rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} fill="none" stroke="#333" />
      <line x1={P} x2={W - P} y1={y(0)} y2={y(0)} stroke="#2a2f36" />
      {Object.entries(COLOURS).map(([k, c]) => {
        const vals = tl[k] as (number | null)[] | undefined;
        if (!vals) return null;
        const pts = vals.map((v, i) => (v === null ? null : `${x(t[i])},${y(v)}`)).filter(Boolean).join(" ");
        return <polyline key={k} points={pts} fill="none" stroke={c} strokeWidth={1.2} />;
      })}
      {mark(ev.last_normal_date, "#888")}
      {mark(ev.earliest_supported_date, "#ff3b30")}
    </svg>
  );
}

export default function Review({ onShow }: { onShow: (ev: EventRow | null) => void }) {
  const [aoi, setAoi] = useState("");
  const [rows, setRows] = useState<EventRow[]>([]);
  const [sel, setSel] = useState<(EventRow & { evidence?: { timeline?: Timeline } }) | null>(null);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState("");

  async function load() {
    const q = new URLSearchParams({ min_conf: "0" });
    if (aoi) q.set("aoi", aoi);
    const r = await fetch(`/change?${q}`);
    const d = await r.json();
    setRows((d.events as EventRow[]).filter((e) => e.analyst_status === "unreviewed").slice(0, 200));
  }
  useEffect(() => { load(); }, [aoi]); // eslint-disable-line react-hooks/exhaustive-deps

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
    if (!r.ok) return setMsg(`error ${r.status}`);
    const i = rows.findIndex((e) => e.event_id === sel.event_id);
    const rest = rows.filter((e) => e.event_id !== sel.event_id);
    setRows(rest);
    setMsg(`${sel.event_id} ${decision}ed (audited)`);
    if (rest[i]) open(rest[i]);
    else { setSel(null); onShow(null); }
  }

  return (
    <div className="review">
      <div className="row">
        <select value={aoi} onChange={(e) => setAoi(e.target.value)} aria-label="Area of interest">
          {["", "dholera", "hasdeo", "navimumbai"].map((a) => <option key={a} value={a}>{a || "all AOIs"}</option>)}
        </select>
        <span className="note">{rows.length} unreviewed (highest confidence first; confidence not yet calibrated)</span>
      </div>
      <ol className="results">
        {rows.map((e) => (
          <li key={e.event_id} className={sel?.event_id === e.event_id ? "sel" : ""} onClick={() => open(e)}>
            <div>
              <b>{e.event_id}</b> {e.change_type} · {e.aoi}
              <br />
              conf {e.confidence.toFixed(2)} · from {e.earliest_supported_date}
            </div>
          </li>
        ))}
      </ol>
      {sel && (
        <section className="prov">
          <h2>{sel.event_id} · {sel.change_type}</h2>
          <div className="pair">
            {sel.last_normal_date && <figure><img src={`/chips/${sel.tile_id}/${season(sel.last_normal_date)}.png?size=160`} alt="before" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} /><figcaption>before {season(sel.last_normal_date)}</figcaption></figure>}
            {sel.last_normal_date && <figure><img src={`/chips/${sel.tile_id}/${season(sel.last_normal_date, 1 + new Date(sel.earliest_supported_date).getUTCFullYear() - new Date(sel.last_normal_date).getUTCFullYear())}.png?size=160`} alt="after" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} /><figcaption>after (same season, +1 y)</figcaption></figure>}
          </div>
          {sel.evidence?.timeline && <TimelineSvg tl={sel.evidence.timeline} ev={sel} />}
          <p>last normal {sel.last_normal_date ?? "—"} · earliest supported {sel.earliest_supported_date}</p>
          <p>Scenes: {sel.provenance.scene_ids.length} · cascade {sel.provenance.cascade_version}</p>
          <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="reason (e.g. seasonal flooding, fallow field)" aria-label="Reason" />
          <div className="row">
            <button onClick={() => decide("confirm")}>Confirm</button>
            <button onClick={() => decide("reject")}>Reject</button>
          </div>
          <p className="note">{msg}</p>
        </section>
      )}
    </div>
  );
}
