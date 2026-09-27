import { useEffect, useRef, useState } from "react";
import Review, { ConfTag, EvidencePanel, type EventRow, type FullEvent, seasonLabel, SENSOR_TEXT, TYPE_COLOUR, typeLabel } from "./Review";
import System from "./System";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, LngLatBoundsLike, MapLayerMouseEvent } from "maplibre-gl";
// v6 looks for its worker next to its own module, which does not exist once bundled: bundle it and point there.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

maplibregl.setWorkerUrl(workerUrl);

// DESIGN.md §12: every site carries provenance. No basemap: the demo runs offline (§2.8).
type SiteEvent = { event_id: string; change_type: string; confidence: number; earliest_supported_date: string };
type Site = {
  rank: number;
  tile_id: string;
  aoi: string;
  season: string | null;
  matched_season: string | null;
  score: number;
  q_mean: number | null;
  parts: Record<string, number>;
  member_tiles: string[];
  dist_m: Record<string, number>;
  events: SiteEvent[];
  geometry: GeoJSON.Polygon;
  chip_url: string | null;
  provenance: { scene_ids: string[]; model_versions: Record<string, string>; cascade_version: string[] | null };
};
type Parsed = {
  semantic: string;
  change: boolean;
  change_types: string[];
  near: Record<string, number>;
  date_range: [string | null, string | null] | null;
  aoi: string | null;
};
type Mode = { kind: "text"; query: string } | { kind: "similar"; seed: Site; season: string } | null;
type Group = { cluster: number; size: number; contains_seed: boolean };

const WARNING_TEXT: Record<string, string> = {
  below_resolution: "Individual vehicles and small objects are below Sentinel-2's 10 m pixel: treat these results as a weak signal.",
};
const EXAMPLES = ["newly built structures near a river", "land clearing since 2023", "salt pans", "open-cast mine", "water body expanded"];
const AOIS = ["", "dholera", "hasdeo", "navimumbai"];
const GROUP_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"];
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
const monthYear = (iso: string) => new Date(iso).toLocaleDateString("en-GB", { month: "short", year: "numeric" });
const fmtDist = (m: number) => (m < 50 ? "touches" : m < 1000 ? `${Math.round(m)} m from` : `${(m / 1000).toFixed(1)} km from`);

function Bar({ v }: { v: number }) {
  return <span className="bar"><span style={{ width: `${Math.max(4, Math.min(100, v * 100))}%` }} /></span>;
}

/** Plain-language summary of why a site is in the list. */
function headline(s: Site) {
  if (!s.events.length) return <span className="muted">No stored change here: matched on how the place looks</span>;
  const best = [...s.events].sort((a, b) => b.confidence - a.confidence)[0];
  const since = s.events.map((e) => e.earliest_supported_date).sort()[0];
  return (
    <span>
      <i className="dot" style={{ background: TYPE_COLOUR[best.change_type] }} />
      <b>{typeLabel(best.change_type)}</b> detected, first seen <b>{monthYear(since)}</b> · <ConfTag c={best.confidence} />
      {s.events.length > 1 && <span className="muted"> (+{s.events.length - 1} more)</span>}
    </span>
  );
}

export default function App() {
  const mapDiv = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const chipIds = useRef<string[]>([]);
  const [query, setQuery] = useState("");
  const [aoi, setAoi] = useState("");
  const [sites, setSites] = useState<Site[]>([]);
  const [textResults, setTextResults] = useState<{ sites: Site[]; mode: Mode } | null>(null);
  const [mode, setMode] = useState<Mode>(null);
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [ready, setReady] = useState(false);
  const [tab, setTab] = useState<"search" | "review" | "system">("search");
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [sensor, setSensor] = useState("");
  const [selEvent, setSelEvent] = useState<FullEvent | null>(null);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [fb, setFb] = useState<Record<string, string>>({});
  const [groups, setGroups] = useState<{ seed: string; list: Group[]; noise: number } | null>(null);
  const [showEventLayer, setShowEventLayer] = useState(false);
  const [apiKey, setApiKey] = useState(() => { try { return localStorage.getItem("apikey") ?? ""; } catch { return ""; } });

  useEffect(() => {
    const m = new maplibregl.Map({
      container: mapDiv.current!,
      style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#1b1f24" } }] },
      center: [72.2, 22.3],
      zoom: 9,
    });
    m.on("load", () => {
      m.addSource("sites", { type: "geojson", data: EMPTY });
      m.addSource("event", { type: "geojson", data: EMPTY });
      m.addSource("clusters", { type: "geojson", data: EMPTY });
      m.addLayer({
        id: "clusters-fill", type: "fill", source: "clusters",
        paint: { "fill-color": ["match", ["%", ["get", "cluster"], 6], 0, GROUP_COLOURS[0], 1, GROUP_COLOURS[1], 2, GROUP_COLOURS[2], 3, GROUP_COLOURS[3], 4, GROUP_COLOURS[4], GROUP_COLOURS[5]], "fill-opacity": 0.45 },
      });
      m.addLayer({ id: "event-fill", type: "fill", source: "event", paint: { "fill-color": "#ff3b30", "fill-opacity": 0.35 } });
      m.addLayer({ id: "event-line", type: "line", source: "event", paint: { "line-color": "#ff3b30", "line-width": 2 } });
      m.addLayer({ id: "sites-line", type: "line", source: "sites", paint: { "line-color": "#f5b400", "line-width": 2 } });
      m.addLayer({ id: "sites-sel", type: "line", source: "sites", filter: ["==", ["get", "tile_id"], ""], paint: { "line-color": "#ffffff", "line-width": 4 } });
      setReady(true);
      m.on("click", "sites-line", (e: MapLayerMouseEvent) => setSelected(String(e.features?.[0]?.properties?.tile_id ?? "")));
    });
    map.current = m;
    return () => m.remove();
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return; // isStyleLoaded() is false while image sources load
    for (const id of chipIds.current) {
      if (m.getLayer(id)) m.removeLayer(id);
      if (m.getSource(id)) m.removeSource(id);
    }
    chipIds.current = [];
    sites.forEach((s) => {
      if (!s.chip_url) return; // event-only site without a composite chip
      const id = `chip-${s.rank}`;
      const ring = s.geometry.coordinates[0]; // lower-left, lower-right, upper-right, upper-left, closed
      m.addSource(id, { type: "image", url: s.chip_url, coordinates: [ring[3], ring[2], ring[1], ring[0]] as [number, number][] as never });
      m.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": 0.95 } }, "sites-line");
      chipIds.current.push(id);
    });
    (m.getSource("sites") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: sites.map((s) => ({ type: "Feature", geometry: s.geometry, properties: { tile_id: s.tile_id, rank: s.rank } })),
    });
    const all = [...sites.map((s) => s.geometry), ...(mode?.kind === "similar" ? [mode.seed.geometry] : [])];
    if (all.length) {
      const cs = all.flatMap((g) => g.coordinates[0]);
      m.fitBounds([Math.min(...cs.map((c) => c[0])), Math.min(...cs.map((c) => c[1])), Math.max(...cs.map((c) => c[0])), Math.max(...cs.map((c) => c[1]))] as LngLatBoundsLike, { padding: 60, maxZoom: 13 });
    }
  }, [sites, ready, mode]);

  useEffect(() => {
    const m = map.current;
    if (!m || !m.getLayer("sites-sel")) return;
    m.setFilter("sites-sel", ["==", ["get", "tile_id"], selected ?? ""]);
    const s = sites.find((x) => x.tile_id === selected);
    if (s) m.flyTo({ center: s.geometry.coordinates[0][0] as [number, number], zoom: 13 });
  }, [selected, sites]);

  function clearClusters() {
    (map.current?.getSource("clusters") as GeoJSONSource | undefined)?.setData(EMPTY);
    setGroups(null);
  }

  async function runSearch(q: string) {
    if (!q.trim()) return;
    setQuery(q);
    setStatus("Searching…");
    clearClusters();
    const t0 = performance.now();
    try {
      const r = await fetch("/search/text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: q, aoi: aoi || null, date_range: t1 || t2 ? [t1 || null, t2 || null] : null, sensor: sensor || null }) });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const body = await r.json();
      const md: Mode = { kind: "text", query: q };
      setSites(body.sites);
      setTextResults({ sites: body.sites, mode: md });
      setMode(md);
      setQueryId(body.query_id ?? null);
      setFb({});
      setParsed(body.parsed);
      setWarnings(body.warnings ?? []);
      setSelected(null);
      setStatus(`${body.sites.length} places found in ${(performance.now() - t0) / 1000 < 1 ? Math.round(performance.now() - t0) + " ms" : ((performance.now() - t0) / 1000).toFixed(1) + " s"}`);
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`);
    }
  }

  function backToResults() {
    if (!textResults) return;
    clearClusters();
    setSites(textResults.sites);
    setMode(textResults.mode);
    setSelected(null);
    setStatus("");
  }

  async function similar(seed: Site) {
    setStatus("Looking for other places that look like this one…");
    clearClusters();
    const r = await fetch("/search/image", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tile_id: seed.tile_id, k: 20 }) });
    if (!r.ok) return setStatus(`Error ${r.status}: ${await r.text()}`);
    const body = await r.json();
    setSites(body.sites);
    setMode({ kind: "similar", seed, season: body.seed.season });
    setSelected(null);
    setStatus(`${body.sites.length} look-alike places (overlapping neighbours of the original are left out)`);
  }

  async function clusters(seed: Site) {
    setStatus("Grouping places that look alike…");
    const r = await fetch("/discover/cluster", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed_tile_id: seed.tile_id }) });
    if (!r.ok) return setStatus(`Error ${r.status}: ${await r.text()}`);
    const body = await r.json();
    const feats = body.clusters.flatMap((c: { cluster: number; members: { tile_id: string; geometry: GeoJSON.Polygon }[] }) =>
      c.members.map((m) => ({ type: "Feature", geometry: m.geometry, properties: { cluster: c.cluster, tile_id: m.tile_id } })));
    (map.current?.getSource("clusters") as GeoJSONSource | undefined)?.setData({ type: "FeatureCollection", features: feats });
    setGroups({ seed: seed.tile_id, list: body.clusters, noise: body.noise.length });
    setStatus(`${body.clusters.length} groups of look-alike places: see the map legend`);
  }

  async function siteFeedback(tile: string, decision: "confirm" | "reject") {
    if (!queryId) return;
    setStatus(decision === "confirm" ? "Marking as relevant…" : "Removing and re-ranking…");
    const r = await fetch("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: { type: "site", id: tile }, decision, query_id: queryId }) });
    if (!r.ok) return setStatus(`Error ${r.status}`);
    const body = await r.json();
    setFb(body.feedback ?? {});
    setSites(body.sites);
    setTextResults((t) => (t ? { ...t, sites: body.sites } : t));
    setStatus(`Re-ranked using your ${Object.keys(body.feedback ?? {}).length} decision(s). Saved in the audit log.`);
  }

  async function exportQuery() {
    if (!queryId) return;
    setStatus("Preparing signed package…");
    const r = await fetch("/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query_id: queryId }) });
    if (!r.ok) return setStatus(`Export error ${r.status}: ${await r.text()}`);
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = (r.headers.get("Content-Disposition")?.match(/filename="(.+)"/) ?? [])[1] ?? "export.zip";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("Signed package downloaded (digitally signed, recorded in the audit log).");
  }

  function showEvent(ev: EventRow | null) {
    const m = map.current;
    if (!m || !ready) return;
    setShowEventLayer(!!ev);
    (m.getSource("event") as GeoJSONSource).setData(ev ? { type: "Feature", geometry: ev.geometry, properties: {} } : EMPTY);
    if (ev) {
      const cs = ev.geometry.coordinates.flat(2) as GeoJSON.Position[];
      m.fitBounds([Math.min(...cs.map((c) => c[0])), Math.min(...cs.map((c) => c[1])), Math.max(...cs.map((c) => c[0])), Math.max(...cs.map((c) => c[1]))] as LngLatBoundsLike, { padding: 120, maxZoom: 15 });
    }
  }

  const sel = sites.find((s) => s.tile_id === selected);
  const selBest = sel && sel.events.length ? [...sel.events].sort((a, b) => b.confidence - a.confidence)[0].event_id : null;
  useEffect(() => {
    setSelEvent(null);
    showEvent(null);
    if (!selBest) return;
    let live = true;
    fetch(`/events/${selBest}`).then((r) => r.json()).then((d) => { if (live) { setSelEvent(d); showEvent(d); } });
    return () => { live = false; };
  }, [selBest]); // eslint-disable-line react-hooks/exhaustive-deps
  const pills = parsed && mode?.kind === "text" ? [
    parsed.change ? `Looking for change: ${parsed.change_types.map(typeLabel).join(" / ") || "any type"}` : "Looking at appearance (no change filter)",
    ...Object.entries(parsed.near).map(([t, m]) => `Within ${m / 1000} km of ${t === "water" ? "water (river, lake, reservoir, creek or salt pan: not told apart)" : t}`),
    parsed.date_range ? `Dates ${parsed.date_range[0] ?? "…"} → ${parsed.date_range[1] ?? "…"}` : null,
    parsed.aoi ? `Area: ${parsed.aoi}` : null,
    `Appearance: “${parsed.semantic}”`,
  ].filter(Boolean) as string[] : [];

  return (
    <div className="app">
      <aside>
        <header>
          <h1>EO Archive</h1>
          <p className="sub">Search satellite imagery by meaning, and review detected changes. Runs fully offline.</p>
        </header>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "search"} className={tab === "search" ? "on" : ""} onClick={() => setTab("search")}>Search places</button>
          <button role="tab" aria-selected={tab === "review"} className={tab === "review" ? "on" : ""} onClick={() => { setTab("review"); clearClusters(); }}>Review changes</button>
          <button role="tab" aria-selected={tab === "system"} className={tab === "system" ? "on" : ""} onClick={() => setTab("system")}>System</button>
        </div>

        {tab === "system" ? <System /> : tab === "review" ? <Review onShow={showEvent} /> : <>
          <form onSubmit={(e) => { e.preventDefault(); runSearch(query); }}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Describe what you are looking for…" autoFocus />
            <select value={aoi} onChange={(e) => setAoi(e.target.value)} aria-label="Area of interest">
              {AOIS.map((a) => <option key={a} value={a}>{a ? `Area: ${a}` : "All areas"}</option>)}
            </select>
            <button type="submit" className="primary">Search</button>
            <div className="row dates">
              <label>Changed/imaged from <input type="date" value={t1} onChange={(e) => setT1(e.target.value)} /></label>
              <label>to <input type="date" value={t2} onChange={(e) => setT2(e.target.value)} /></label>
            </div>
            <select value={sensor} onChange={(e) => setSensor(e.target.value)} aria-label="Sensor">
              <option value="">Any sensor</option>
              {Object.entries(SENSOR_TEXT).map(([k, v]) => <option key={k} value={k}>{k === "sentinel2" ? v : `Changes confirmed by ${v}`}</option>)}
            </select>
          </form>
          {!mode && (
            <div className="examples">
              <span className="muted">Try:</span>
              {EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => runSearch(x)}>{x}</button>)}
            </div>
          )}
          <p className="status">{status}</p>

          {mode?.kind === "text" && (
            <div className="banner">
              <div className="banner-title">Results for “{mode.query}”</div>
              <div className="pills">{pills.map((p) => <span key={p} className="pill">{p}</span>)}</div>
              {warnings.map((w) => <p key={w} className="warn">{WARNING_TEXT[w] ?? w}</p>)}
              {queryId && sites.length > 0 && <button className="ghost" onClick={exportQuery}>Download signed results</button>}
            </div>
          )}
          {mode?.kind === "similar" && (
            <div className="banner">
              <div className="banner-title">Places that look like this one</div>
              <div className="seed">
                {mode.seed.chip_url && <img src={`/chips/${mode.seed.tile_id}/${mode.season}.png`} alt="" width={56} height={56} />}
                <div className="muted">Original place: {mode.seed.aoi} · compared using its {seasonLabel(mode.season)} image. Ranked by visual similarity (Prithvi model).</div>
              </div>
              {textResults && <button className="ghost" onClick={backToResults}>← Back to search results</button>}
            </div>
          )}

          <ol className="results">
            {sites.map((s) => (
              <li key={s.tile_id} className={s.tile_id === selected ? "sel" : ""} onClick={() => setSelected(s.tile_id)}>
                <div className="thumb">
                  {s.chip_url ? <img src={s.chip_url} alt="" width={72} height={72} /> : <div className="nochip" />}
                  <span className="rank">{s.rank}</span>
                </div>
                <div className="body">
                  <div className="line">{headline(s)}</div>
                  <div className="line meta">
                    Match <Bar v={s.parts.sim_norm ?? s.score} />
                    {"water" in s.dist_m && <span>· {fmtDist(s.dist_m.water)} water</span>}
                    {"road" in s.dist_m && s.dist_m.road < 2000 && <span>· {fmtDist(s.dist_m.road)} road</span>}
                  </div>
                  <div className="line muted">{s.aoi} · picture: {seasonLabel(s.season)}{s.events.length ? " (current state)" : ""}{s.q_mean !== null && s.q_mean < 0.8 ? " · partly cloudy" : ""}</div>
                  <div className="actions" onClick={(e) => e.stopPropagation()}>
                    <button className="ghost small" title="Find other places that look like this (image search)" onClick={() => similar(s)}>Find look-alikes</button>
                    <button className="ghost small" title="Colour groups of look-alike places on the map" onClick={() => clusters(s)}>Group look-alikes</button>
                    {queryId && mode?.kind === "text" && <>
                      <button className={`ghost small ${fb[s.tile_id] === "confirm" ? "on" : ""}`} title="Relevant: pull similar results up" onClick={() => siteFeedback(s.tile_id, "confirm")}>✓ Relevant</button>
                      <button className="ghost small" title="Not relevant: remove and re-rank" onClick={() => siteFeedback(s.tile_id, "reject")}>✗ Not relevant</button>
                    </>}
                  </div>
                </div>
              </li>
            ))}
          </ol>

          {sel && (
            <section className="detail">
              <h2>Place #{sel.rank}: details</h2>
              {sel.events.length > 0
                ? (selEvent ? <EvidencePanel ev={selEvent} /> : <p className="muted">Loading evidence…</p>)
                : sel.chip_url && <><img src={`${sel.chip_url}?size=160`} alt="" className="big" /><p className="muted">{seasonLabel(sel.season)} image; no change stored at this place.</p></>}
              <h3>Why it is ranked here</h3>
              <table className="why"><tbody>
                <tr><td>Looks like your description</td><td><Bar v={sel.parts.sim_norm ?? 0} /></td></tr>
                <tr><td>Confidence of a detected change</td><td><Bar v={sel.parts.event_conf ?? 0} /></td></tr>
                <tr><td>Image quality (cloud-free)</td><td><Bar v={sel.parts.q ?? 0} /></td></tr>
                <tr><td>Fits the place you asked for</td><td><Bar v={sel.parts.spatial_fit ?? 0} /></td></tr>
              </tbody></table>
              {sel.events.length > 1 && <h3>All changes at this place</h3>}
              {sel.events.length > 1 && sel.events.map((e) => (
                <p key={e.event_id} className="evline"><i className="dot" style={{ background: TYPE_COLOUR[e.change_type] }} />{typeLabel(e.change_type)} · first seen {e.earliest_supported_date} · <ConfTag c={e.confidence} /> <span className="muted">({e.event_id})</span></p>
              ))}
              <details>
                <summary>Search provenance (images and models behind this result)</summary>
                <p>Satellite scenes used: {sel.provenance.scene_ids.length}</p>
                <p className="mono">{sel.provenance.scene_ids.join(", ")}</p>
                <p>Models: {Object.entries(sel.provenance.model_versions).map(([k, v]) => `${k} = ${v}`).join(", ") || "—"}</p>
                <p>Change-detection version: {sel.provenance.cascade_version?.join(", ") ?? "—"}</p>
                <p>Grid cells: {sel.member_tiles.join(", ")}</p>
              </details>
            </section>
          )}

          <details className="help">
            <summary>How to read this screen</summary>
            <ul>
              <li><b>Search</b> in plain words. The system understands change words (built, cleared, flooded), places (near a river, road) and dates (since 2023).</li>
              <li>Each result is a 640 m square of ground. The picture is a cloud-free satellite image of it for one season.</li>
              <li><b>Match</b> = how well it looks like your description. A coloured dot means a change was detected there, with how likely it is to be real.</li>
              <li><b>✓ / ✗</b> teaches the search what you mean; the list re-ranks immediately.</li>
              <li><b>Find look-alikes</b> shows other places that look like the chosen one. <b>Group look-alikes</b> colours groups of similar places on the map.</li>
              <li>At 10 m per pixel we see construction sites, clearing, mining, water and major roads, not individual vehicles or small buildings.</li>
            </ul>
          </details>
          <input className="apikey" type="password" value={apiKey} placeholder="API key (only if login is enabled)" aria-label="API key"
            onChange={(e) => { setApiKey(e.target.value); try { localStorage.setItem("apikey", e.target.value); } catch { /* ignore */ } }} />
        </>}
      </aside>
      <main>
        <div ref={mapDiv} className="map" />
        <div className="legend">
          <b>Map key</b>
          {tab === "search" && <>
            <div><i className="sw line" style={{ borderColor: "#f5b400" }} /> Result (satellite image inside)</div>
            <div><i className="sw line" style={{ borderColor: "#fff" }} /> Selected result</div>
          </>}
          {showEventLayer && tab !== "system" && <div><i className="sw" style={{ background: "rgba(255,59,48,.5)", borderColor: "#ff3b30" }} /> Area that changed</div>}
          {tab === "review" && !showEventLayer && <div className="muted">Pick a change on the left</div>}
          {groups && <>
            <div className="muted" style={{ marginTop: 6 }}>Look-alike groups</div>
            {groups.list.map((g, i) => (
              <div key={g.cluster}><i className="sw" style={{ background: GROUP_COLOURS[g.cluster % 6] }} /> Group {String.fromCharCode(65 + i)}: {g.size} places{g.contains_seed ? " (incl. your place)" : ""}</div>
            ))}
            <div className="muted">{groups.noise} places fit no group</div>
          </>}
        </div>
      </main>
    </div>
  );
}
