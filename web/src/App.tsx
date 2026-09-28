import { useEffect, useRef, useState } from "react";
import Review, { ConfTag, EvidencePanel, type EventRow, type FullEvent, seasonLabel, SENSOR_TEXT, TYPE_COLOUR, typeLabel } from "./Review";
import System from "./System";
import * as Icon from "./Icons";
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
  landcover?: string[];
};
type Mode = { kind: "text"; query: string } | { kind: "similar"; seed: Site; season: string } | null;
type Group = { cluster: number; size: number; contains_seed: boolean };
type Tab = "search" | "review" | "system";
type Archive = { aois: Record<string, { bbox: number[]; area_km2: number }>; area_km2: number; scenes: number; auth: string };

const WARNING_TEXT: Record<string, string> = {
  below_resolution: "Individual vehicles and small objects are below Sentinel-2's 10 m pixel: treat these results as a weak signal.",
};
const EXAMPLES = ["newly built structures near a river", "land clearing since 2023", "open-cast coal mine", "dense forest", "salt pans", "water body expanded"];
const AOIS = ["", "dholera", "hasdeo", "navimumbai"];
const GROUP_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"];
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
const monthYear = (iso: string) => new Date(iso).toLocaleDateString("en-GB", { month: "short", year: "numeric" });
const fmtDist = (m: number) => (m < 50 ? "touches" : m < 1000 ? `${Math.round(m)} m from` : `${(m / 1000).toFixed(1)} km from`);
const fmtMs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

function Bar({ v }: { v: number }) {
  return <span className="bar" title={`${Math.round(v * 100)} / 100`}><span style={{ width: `${Math.max(4, Math.min(100, v * 100))}%` }} /></span>;
}

/** Plain-language summary of why a site is in the list. */
function headline(s: Site) {
  if (!s.events.length) return <span className="title muted-strong">Appearance match</span>;
  const best = [...s.events].sort((a, b) => b.confidence - a.confidence)[0];
  const since = s.events.map((e) => e.earliest_supported_date).sort()[0];
  return (
    <span className="title">
      <i className="dot" style={{ background: TYPE_COLOUR[best.change_type] }} />
      {typeLabel(best.change_type)} <span className="muted">since {monthYear(since)}</span>
      {s.events.length > 1 && <span className="muted"> · +{s.events.length - 1}</span>}
    </span>
  );
}

function Skeleton() {
  return <ol className="results">{Array.from({ length: 6 }, (_, i) => <li key={i} className="skeleton"><div className="sk-thumb" /><div className="body"><div className="sk-line w60" /><div className="sk-line w80" /><div className="sk-line w40" /></div></li>)}</ol>;
}

function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="How to use" onClick={(e) => e.stopPropagation()}>
        <button className="icon-btn close" title="Close" onClick={onClose}><Icon.X /></button>
        <h2>How to use the workbench</h2>
        <div className="help-grid">
          <section><h3><Icon.Search /> Search places</h3>
            <p>Type what you are looking for in plain words. The system understands <b>change words</b> (built, cleared, flooded), <b>places</b> (near a river or road), <b>dates</b> (since 2023) and <b>land cover</b> (forest, farmland, urban).</p>
            <p>Each result is a 640 m square of ground with a cloud-free satellite picture. Click one for its evidence.</p></section>
          <section><h3><Icon.Similar /> Look-alikes and groups</h3>
            <p><b>Look-alikes</b> finds other places that look like the chosen one, anywhere in the archive. <b>Groups</b> colours clusters of similar places on the map.</p>
            <p><b>✓ / ✗</b> teaches the search what you mean; the list re-ranks immediately and the decision is audited.</p></section>
          <section><h3><Icon.Change /> Review changes</h3>
            <p>Changes detected automatically, most confident first. Choose an area and time window, open a change and check before/after images, dates, sensors and the independent cross-checks.</p>
            <p>Keyboard: <kbd>J</kbd>/<kbd>K</kbd> next/previous, <kbd>C</kbd> real change, <kbd>R</kbd> false alarm.</p></section>
          <section><h3><Icon.Shield /> What to trust</h3>
            <p>Confidence is shown as high / medium / low. Low-confidence and unclassified changes are hidden unless asked for. Every decision and export is recorded in a tamper-evident audit log; exports are digitally signed.</p>
            <p>At 10 m per pixel we see construction sites, clearing and mining, water and major roads, not individual vehicles or small buildings.</p></section>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const mapDiv = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const chipIds = useRef<string[]>([]);
  const markers = useRef<maplibregl.Marker[]>([]);
  const overview = useRef<LngLatBoundsLike | null>(null);
  const [drawerEl, setDrawerEl] = useState<HTMLDivElement | null>(null);
  const [query, setQuery] = useState("");
  const [aoi, setAoi] = useState("");
  const [sites, setSites] = useState<Site[]>([]);
  const [textResults, setTextResults] = useState<{ sites: Site[]; mode: Mode } | null>(null);
  const [mode, setMode] = useState<Mode>(null);
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [tab, setTab] = useState<Tab>("search");
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [sensor, setSensor] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [selEvent, setSelEvent] = useState<FullEvent | null>(null);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [fb, setFb] = useState<Record<string, string>>({});
  const [groups, setGroups] = useState<{ seed: string; list: Group[]; noise: number } | null>(null);
  const [showEventLayer, setShowEventLayer] = useState(false);
  const [focusArea, setFocusArea] = useState<string | null>(null);
  const [archive, setArchive] = useState<Archive | null>(null);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const [help, setHelp] = useState(false);
  const [settings, setSettings] = useState(false);
  const [apiKey, setApiKey] = useState(() => { try { return localStorage.getItem("apikey") ?? ""; } catch { return ""; } });

  useEffect(() => {
    const m = new maplibregl.Map({
      container: mapDiv.current!,
      style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#161b21" } }] },
      center: [72.2, 22.3],
      zoom: 9,
      attributionControl: false,
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    m.on("mousemove", (e) => setCursor([e.lngLat.lat, e.lngLat.lng]));
    m.on("mouseout", () => setCursor(null));
    m.on("load", () => {
      m.addSource("aois", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "aois-fill", type: "fill", source: "aois", paint: { "fill-color": "#58a6ff", "fill-opacity": 0.04 } });
      m.addLayer({ id: "aois-line", type: "line", source: "aois", paint: { "line-color": "#58a6ff", "line-opacity": 0.55, "line-width": 1.2, "line-dasharray": [3, 2] } });
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
      // Archive extent from the start, so an analyst knows where imagery exists (labels as DOM markers: no glyphs offline)
      fetch("/stats").then((r) => r.json()).then((st: Archive) => {
        setArchive(st);
        const entries = Object.entries(st.aois).filter(([, a]) => a.bbox);
        (m.getSource("aois") as GeoJSONSource).setData({
          type: "FeatureCollection",
          features: entries.map(([name, a]) => {
            const [x0, y0, x1, y1] = a.bbox;
            return { type: "Feature", properties: { name }, geometry: { type: "Polygon", coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] } };
          }),
        });
        entries.forEach(([name, a]) => {
          const el = document.createElement("button");
          el.className = "aoi-label";
          el.title = `Zoom to ${name}`;
          el.textContent = `${name} · ${Math.round(a.area_km2)} km²`;
          el.onclick = (e) => { e.stopPropagation(); m.fitBounds(a.bbox as LngLatBoundsLike, { padding: 60 }); };
          new maplibregl.Marker({ element: el, anchor: "bottom-left" }).setLngLat([a.bbox[0], a.bbox[3]]).addTo(m);
        });
        overview.current = entries.length ? (() => {
          const xs = entries.flatMap(([, a]) => [a.bbox[0], a.bbox[2]]), ys = entries.flatMap(([, a]) => [a.bbox[1], a.bbox[3]]);
          return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] as LngLatBoundsLike;
        })() : null;
        if (overview.current) m.fitBounds(overview.current, { padding: 80, duration: 0 });
      }).catch(() => undefined);
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
      m.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": 0.95 } }, "clusters-fill"); // under groups + change area
      chipIds.current.push(id);
    });
    markers.current.forEach((mk) => mk.remove());
    markers.current = sites.map((s) => {
      const el = document.createElement("button");
      el.className = "site-marker";
      el.textContent = String(s.rank);
      el.title = `Result ${s.rank}`;
      el.onclick = (e) => { e.stopPropagation(); setSelected(s.tile_id); };
      const ring = s.geometry.coordinates[0];
      const c: [number, number] = [(ring[0][0] + ring[2][0]) / 2, ring[2][1]];
      return new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat(c).addTo(m);
    });
    (m.getSource("sites") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: sites.map((s) => ({ type: "Feature", geometry: s.geometry, properties: { tile_id: s.tile_id, rank: s.rank } })),
    });
    if (sites.length) zoomToArea(focusArea ?? sites[0].aoi);
  }, [sites, ready, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Areas are hundreds of km apart: frame one area's results at a time (the header offers the others). */
  function zoomToArea(area: string) {
    const m = map.current;
    if (!m) return;
    setFocusArea(area);
    const geoms = [...sites.filter((s) => s.aoi === area).map((s) => s.geometry), ...(mode?.kind === "similar" && mode.seed.aoi === area ? [mode.seed.geometry] : [])];
    if (!geoms.length) return;
    const cs = geoms.flatMap((g) => g.coordinates[0]);
    m.fitBounds([Math.min(...cs.map((c) => c[0])), Math.min(...cs.map((c) => c[1])), Math.max(...cs.map((c) => c[0])), Math.max(...cs.map((c) => c[1]))] as LngLatBoundsLike, { padding: 80, maxZoom: 13 });
  }
  const areaCounts = Object.entries(sites.reduce<Record<string, number>>((a, s) => ({ ...a, [s.aoi]: (a[s.aoi] ?? 0) + 1 }), {}));

  useEffect(() => {
    const m = map.current;
    if (!m || !m.getLayer("sites-sel")) return;
    m.setFilter("sites-sel", ["==", ["get", "tile_id"], selected ?? ""]);
    const s = sites.find((x) => x.tile_id === selected);
    if (s && !s.events.length) { // change sites zoom to the changed area instead (showEvent)
      const ring = s.geometry.coordinates[0];
      m.fitBounds([ring[0] as [number, number], ring[2] as [number, number]], { padding: { top: 120, bottom: 120, left: 120, right: Math.min(560, m.getContainer().clientWidth / 2) }, maxZoom: 14 });
    }
    markers.current.forEach((mk) => mk.getElement().classList.toggle("on", mk.getElement().textContent === String(s?.rank)));
    if (s) document.getElementById(`r-${s.tile_id}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected, sites]);

  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const vis = tab === "search" ? "visible" : "none";
    for (const id of [...chipIds.current, "sites-line", "sites-sel", "clusters-fill"]) if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", vis);
    markers.current.forEach((mk) => { mk.getElement().style.display = tab === "search" ? "" : "none"; });
    if (tab === "system" && overview.current) m.fitBounds(overview.current, { padding: 80 });
    if (tab === "search" && sites.length) zoomToArea(focusArea ?? sites[0].aoi);
  }, [tab, ready]); // eslint-disable-line react-hooks/exhaustive-deps

  function clearClusters() {
    (map.current?.getSource("clusters") as GeoJSONSource | undefined)?.setData(EMPTY);
    setGroups(null);
  }

  async function call<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(true); setError(""); setStatus(label);
    try { return await fn(); } catch (e) { setError((e as Error).message); setStatus(""); return undefined; } finally { setBusy(false); }
  }
  async function post(path: string, body: unknown) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${r.status}: ${(await r.text()).slice(0, 200)}`);
    return r;
  }

  async function runSearch(q: string) {
    if (!q.trim()) return;
    setQuery(q);
    clearClusters();
    setSelected(null);
    const t0 = performance.now();
    const body = await call("Searching the archive…", async () => (await post("/search/text", { query: q, aoi: aoi || null, date_range: t1 || t2 ? [t1 || null, t2 || null] : null, sensor: sensor || null })).json());
    if (!body) return;
    const md: Mode = { kind: "text", query: q };
    setFocusArea(null);
    setSites(body.sites);
    setTextResults({ sites: body.sites, mode: md });
    setMode(md);
    setQueryId(body.query_id ?? null);
    setFb({});
    setParsed(body.parsed);
    setWarnings(body.warnings ?? []);
    setStatus(`${body.sites.length} places · ${fmtMs(performance.now() - t0)}`);
  }

  function backToResults() {
    if (!textResults) return;
    clearClusters();
    setSites(textResults.sites);
    setMode(textResults.mode);
    setSelected(null);
    setStatus(`${textResults.sites.length} places`);
  }

  async function similar(seed: Site) {
    clearClusters();
    const body = await call("Finding places that look like this one…", async () => (await post("/search/image", { tile_id: seed.tile_id, k: 20 })).json());
    if (!body) return;
    setFocusArea(null);
    setSites(body.sites);
    setMode({ kind: "similar", seed, season: body.seed.season });
    setSelected(null);
    setStatus(`${body.sites.length} look-alike places`);
  }

  async function clusters(seed: Site) {
    const body = await call("Grouping look-alike places…", async () => (await post("/discover/cluster", { seed_tile_id: seed.tile_id })).json());
    if (!body) return;
    const feats = body.clusters.flatMap((c: { cluster: number; members: { tile_id: string; geometry: GeoJSON.Polygon }[] }) =>
      c.members.map((m) => ({ type: "Feature", geometry: m.geometry, properties: { cluster: c.cluster, tile_id: m.tile_id } })));
    (map.current?.getSource("clusters") as GeoJSONSource | undefined)?.setData({ type: "FeatureCollection", features: feats });
    setGroups({ seed: seed.tile_id, list: body.clusters, noise: body.noise.length });
    setStatus(`${body.clusters.length} look-alike groups on the map`);
  }

  async function siteFeedback(tile: string, decision: "confirm" | "reject") {
    if (!queryId) return;
    const body = await call(decision === "confirm" ? "Learning from your choice…" : "Removing and re-ranking…", async () => (await post("/feedback", { target: { type: "site", id: tile }, decision, query_id: queryId })).json());
    if (!body) return;
    setFb(body.feedback ?? {});
    setSites(body.sites);
    setTextResults((t) => (t ? { ...t, sites: body.sites } : t));
    setStatus(`Re-ranked with ${Object.keys(body.feedback ?? {}).length} decision(s) · audited`);
  }

  async function exportQuery() {
    if (!queryId) return;
    const r = await call("Preparing the signed package…", () => post("/export", { query_id: queryId, format: "gpkg" }));
    if (!r) return;
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = (r.headers.get("Content-Disposition")?.match(/filename="(.+)"/) ?? [])[1] ?? "export.zip";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("Signed package downloaded (Ed25519 signature, audited)");
  }

  function showEvent(ev: EventRow | null) {
    const m = map.current;
    if (!m || !ready) return;
    setShowEventLayer(!!ev);
    (m.getSource("event") as GeoJSONSource).setData(ev ? { type: "Feature", geometry: ev.geometry, properties: {} } : EMPTY);
    // Review tab: the tile's "after" image under the changed area, so the outline has context
    if (m.getLayer("event-chip")) m.removeLayer("event-chip");
    if (m.getSource("event-chip")) m.removeSource("event-chip");
    const full = ev as FullEvent | null;
    const img = full?.images?.after ?? full?.images?.before;
    if (tab === "review" && img && full?.tile_geometry) {
      const ring = full.tile_geometry.coordinates[0];
      m.addSource("event-chip", { type: "image", url: `${img.chip_url}?size=256`, coordinates: [ring[3], ring[2], ring[1], ring[0]] as [number, number][] as never });
      m.addLayer({ id: "event-chip", type: "raster", source: "event-chip", paint: { "raster-opacity": 0.95 } }, "clusters-fill");
    }
    if (ev) {
      const cs = ev.geometry.coordinates.flat(2) as GeoJSON.Position[];
      const right = Math.min(500, m.getContainer().clientWidth / 2); // the details panel covers the right side
      m.fitBounds([Math.min(...cs.map((c) => c[0])), Math.min(...cs.map((c) => c[1])), Math.max(...cs.map((c) => c[0])), Math.max(...cs.map((c) => c[1]))] as LngLatBoundsLike, { padding: { top: 100, bottom: 100, left: 100, right: right + 60 }, maxZoom: 15 });
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

  useEffect(() => { // Esc closes the details panel / dialogs
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") { setHelp(false); setSettings(false); if (tab === "search") setSelected(null); } };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [tab]);

  const pills = parsed && mode?.kind === "text" ? [
    parsed.change ? `Change: ${parsed.change_types.map(typeLabel).join(" / ") || "any type"}` : "Appearance search",
    ...Object.entries(parsed.near).map(([t, m]) => `Within ${m / 1000} km of ${t === "water" ? "water*" : t}`),
    parsed.date_range ? `${parsed.date_range[0] ?? "…"} → ${parsed.date_range[1] ?? "…"}` : null,
    parsed.aoi ? `Area: ${parsed.aoi}` : null,
    parsed.landcover?.length ? `Land cover: ${parsed.landcover.join(" / ")}` : null,
    `“${parsed.semantic}”`,
  ].filter(Boolean) as string[] : [];
  const activeFilters = [aoi, t1, t2, sensor].filter(Boolean).length;
  const switchTab = (t: Tab) => { setTab(t); if (t !== "search") clearClusters(); if (t !== "review") showEvent(null); };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><Icon.Logo /><div><b>EO Archive</b><span>Semantic search &amp; change analysis</span></div></div>
        <nav className="nav" role="tablist">
          <button role="tab" aria-selected={tab === "search"} className={tab === "search" ? "on" : ""} onClick={() => switchTab("search")}><Icon.Search /> Search</button>
          <button role="tab" aria-selected={tab === "review"} className={tab === "review" ? "on" : ""} onClick={() => switchTab("review")}><Icon.Change /> Review changes</button>
          <button role="tab" aria-selected={tab === "system"} className={tab === "system" ? "on" : ""} onClick={() => switchTab("system")}><Icon.Chart /> System</button>
        </nav>
        <div className="top-right">
          {archive && <span className="chip-stat" title="Indexed archive">{Math.round(archive.area_km2).toLocaleString("en-IN")} km² · {archive.scenes} scenes</span>}
          <span className="chip-stat ok" title="All models and data are local; no internet is used at runtime"><i className="live" /> Offline</span>
          <button className="icon-btn" title="How to use (?)" onClick={() => setHelp(true)}><Icon.Help size={18} /></button>
          <div className="pop-wrap">
            <button className="icon-btn" title="Settings" onClick={() => setSettings((v) => !v)}><Icon.Gear size={18} /></button>
            {settings && (
              <div className="popover">
                <label>API key <span className="muted">(only when login is enabled: {archive?.auth ?? "…"})</span>
                  <input type="password" value={apiKey} placeholder="paste key" onChange={(e) => { setApiKey(e.target.value); try { localStorage.setItem("apikey", e.target.value); } catch { /* ignore */ } }} />
                </label>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="app">
        <aside>
          {tab === "system" ? <System /> : tab === "review" ? <Review onShow={showEvent} drawer={drawerEl} /> : <>
            <form className="searchbox" onSubmit={(e) => { e.preventDefault(); runSearch(query); }}>
              <div className="search-input">
                <Icon.Search />
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Describe what you are looking for…" aria-label="Search query" autoFocus />
                <button type="button" className={`icon-btn ${showFilters || activeFilters ? "on" : ""}`} title="Filters" onClick={() => setShowFilters((v) => !v)}>
                  <Icon.Filter />{activeFilters > 0 && <span className="badge">{activeFilters}</span>}
                </button>
              </div>
              {showFilters && (
                <div className="filters">
                  <label>Area<select value={aoi} onChange={(e) => setAoi(e.target.value)}>{AOIS.map((a) => <option key={a} value={a}>{a || "All areas"}</option>)}</select></label>
                  <label>Sensor<select value={sensor} onChange={(e) => setSensor(e.target.value)}>
                    <option value="">Any</option>
                    {Object.entries(SENSOR_TEXT).map(([k, v]) => <option key={k} value={k}>{k === "sentinel2" ? v : `Confirmed by ${v}`}</option>)}
                  </select></label>
                  <label>From<input type="date" value={t1} onChange={(e) => setT1(e.target.value)} /></label>
                  <label>To<input type="date" value={t2} onChange={(e) => setT2(e.target.value)} /></label>
                  {activeFilters > 0 && <button type="button" className="link" onClick={() => { setAoi(""); setSensor(""); setT1(""); setT2(""); }}>Clear filters</button>}
                </div>
              )}
              <button type="submit" className="primary block" disabled={busy}>{busy ? "Searching…" : "Search"}</button>
            </form>

            {error && <div className="alert">Something went wrong: {error}</div>}

            {!mode && !busy && (
              <div className="empty">
                <h3>Search 7 years of satellite imagery by meaning</h3>
                <p className="muted">Sentinel-2, Sentinel-1 and Landsat over {archive ? Object.keys(archive.aois).length : 3} areas. Try one of these:</p>
                <div className="examples">{EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => runSearch(x)}>{x}</button>)}</div>
              </div>
            )}

            {mode && (
              <div className="results-head">
                {mode.kind === "text" ? (
                  <>
                    <div className="rh-row"><b className="rh-title">“{mode.query}”</b>
                      {queryId && sites.length > 0 && <button className="icon-btn" title="Download signed results (GeoJSON + GeoPackage)" onClick={exportQuery}><Icon.Download /></button>}
                    </div>
                    <div className="pills">{pills.map((p) => <span key={p} className="pill">{p}</span>)}</div>
                    {Object.keys(parsed?.near ?? {}).includes("water") && <p className="fine">* water = river, lake, reservoir, creek or salt pan (the water map does not tell them apart)</p>}
                  </>
                ) : (
                  <>
                    <div className="rh-row">
                      <button className="link" onClick={backToResults} disabled={!textResults}><Icon.Back /> Back</button>
                      <b className="rh-title">Places like this one</b>
                    </div>
                    <div className="seed">
                      {mode.seed.chip_url && <img src={`/chips/${mode.seed.tile_id}/${mode.season}.png`} alt="" width={48} height={48} />}
                      <span className="muted">{mode.seed.aoi} · {seasonLabel(mode.season)} image · ranked by visual similarity (Prithvi-EO-2.0)</span>
                    </div>
                  </>
                )}
                {warnings.map((w) => <p key={w} className="warn">{WARNING_TEXT[w] ?? w}</p>)}
                {areaCounts.length > 1 && <div className="areas">{areaCounts.map(([a, k]) => <button key={a} className={`chip ${a === focusArea ? "on" : ""}`} onClick={() => zoomToArea(a)}>{a} <b>{k}</b></button>)}</div>}
              </div>
            )}
            <div className={`status ${busy ? "busy" : ""}`}>{busy && <i className="spinner" />}{status}</div>

            {busy && !sites.length ? <Skeleton /> : (
              <ol className="results">
                {sites.map((s) => (
                  <li key={s.tile_id} id={`r-${s.tile_id}`} className={s.tile_id === selected ? "sel" : ""} onClick={() => setSelected(s.tile_id)}>
                    <div className="thumb">
                      {s.chip_url ? <img src={s.chip_url} alt="" width={68} height={68} loading="lazy" /> : <div className="nochip" />}
                      <span className="rank">{s.rank}</span>
                    </div>
                    <div className="body">
                      <div className="line">{headline(s)}{s.events.length > 0 && <ConfTag c={Math.max(...s.events.map((e) => e.confidence))} />}</div>
                      <div className="line meta">
                        <span className="kv">match <Bar v={s.parts.sim_norm ?? s.score} /></span>
                        {"water" in s.dist_m && <span className="kv">{fmtDist(s.dist_m.water)} water</span>}
                        {"road" in s.dist_m && s.dist_m.road < 2000 && <span className="kv">{fmtDist(s.dist_m.road)} road</span>}
                      </div>
                      <div className="line fine">{s.aoi} · {seasonLabel(s.season)}{s.events.length ? " (current)" : ""}{s.q_mean !== null && s.q_mean < 0.8 ? " · partly cloudy" : ""}</div>
                    </div>
                    <div className="actions" onClick={(e) => e.stopPropagation()}>
                      <button className="icon-btn" title="Look-alikes: other places that look like this" onClick={() => similar(s)}><Icon.Similar /></button>
                      <button className="icon-btn" title="Groups: colour clusters of similar places" onClick={() => clusters(s)}><Icon.Group /></button>
                      {queryId && mode?.kind === "text" && <>
                        <button className={`icon-btn good ${fb[s.tile_id] === "confirm" ? "on" : ""}`} title="Relevant: pull similar results up" onClick={() => siteFeedback(s.tile_id, "confirm")}><Icon.Check /></button>
                        <button className="icon-btn bad" title="Not relevant: remove and re-rank" onClick={() => siteFeedback(s.tile_id, "reject")}><Icon.X /></button>
                      </>}
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </>}
        </aside>

        <main>
          <div ref={mapDiv} className="map" />
          <div ref={setDrawerEl} />
          {tab === "search" && sel && (
            <section className="drawer">
              <div className="drawer-head">
                <span className="rank big">{sel.rank}</span>
                <div><b>Result {sel.rank}</b><span className="muted"> · {sel.aoi}</span></div>
                <button className="icon-btn close" title="Close (Esc)" onClick={() => setSelected(null)}><Icon.X /></button>
              </div>
              <div className="drawer-body">
                {sel.events.length > 0
                  ? (selEvent ? <EvidencePanel ev={selEvent} /> : <p className="muted"><i className="spinner" /> Loading evidence…</p>)
                  : sel.chip_url && <div className="card"><img src={`${sel.chip_url}?size=200`} alt="" className="big" /><p className="muted">{seasonLabel(sel.season)} · no change stored at this place</p></div>}
                <div className="card">
                  <h3>Why it is ranked here</h3>
                  <table className="why"><tbody>
                    <tr><td>Looks like your description</td><td><Bar v={sel.parts.sim_norm ?? 0} /></td></tr>
                    <tr><td>Confidence of a detected change</td><td><Bar v={sel.parts.event_conf ?? 0} /></td></tr>
                    <tr><td>Image quality (cloud-free)</td><td><Bar v={sel.parts.q ?? 0} /></td></tr>
                    <tr><td>Fits the place you asked for</td><td><Bar v={sel.parts.spatial_fit ?? 0} /></td></tr>
                    {(sel.parts.landcover ?? 0) > 0 && <tr><td>Share of that land cover (independent map)</td><td><Bar v={sel.parts.landcover} /></td></tr>}
                  </tbody></table>
                </div>
                {sel.events.length > 1 && (
                  <div className="card">
                    <h3>All changes at this place</h3>
                    {sel.events.map((e) => (
                      <p key={e.event_id} className="evline"><i className="dot" style={{ background: TYPE_COLOUR[e.change_type] }} />{typeLabel(e.change_type)} · {e.earliest_supported_date} <ConfTag c={e.confidence} /></p>
                    ))}
                  </div>
                )}
                <details className="card">
                  <summary>Search provenance</summary>
                  <p>Satellite scenes behind this picture: {sel.provenance.scene_ids.length}</p>
                  <p className="mono">{sel.provenance.scene_ids.join(", ")}</p>
                  <p>Models: {Object.entries(sel.provenance.model_versions).map(([k, v]) => `${k} = ${v}`).join(", ") || "—"}</p>
                  <p>Change-detection version: {sel.provenance.cascade_version?.join(", ") ?? "—"}</p>
                  <p>Grid cells: {sel.member_tiles.join(", ")}</p>
                </details>
              </div>
            </section>
          )}

          <div className="legend">
            <div className="legend-title">Map key</div>
            <div><i className="sw line dash-blue" /> Archive coverage</div>
            {tab === "search" && sites.length > 0 && <>
              <div><i className="sw line" style={{ borderColor: "#f5b400" }} /> Result (satellite picture)</div>
              <div><i className="sw line" style={{ borderColor: "#fff" }} /> Selected</div>
            </>}
            {showEventLayer && tab !== "system" && <div><i className="sw" style={{ background: "rgba(255,59,48,.5)", borderColor: "#ff3b30" }} /> Area that changed</div>}
            {groups && <>
              <div className="legend-sub">Look-alike groups</div>
              {groups.list.map((g, i) => (
                <div key={g.cluster}><i className="sw" style={{ background: GROUP_COLOURS[g.cluster % 6] }} /> Group {String.fromCharCode(65 + i)} · {g.size}{g.contains_seed ? " (yours)" : ""}</div>
              ))}
              <div className="fine">{groups.noise} places fit no group</div>
            </>}
          </div>
          {cursor && <div className="coords">{cursor[0].toFixed(5)}° N · {cursor[1].toFixed(5)}° E</div>}
        </main>
      </div>
      {help && <HelpModal onClose={() => setHelp(false)} />}
    </div>
  );
}
