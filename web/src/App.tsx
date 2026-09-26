import { useEffect, useRef, useState } from "react";
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
const WARNING_TEXT: Record<string, string> = {
  below_resolution: "Individual vehicles/small objects are below Sentinel-2's 10 m resolution: results are a weak, low-confidence signal.",
};

const AOIS = ["", "dholera", "hasdeo", "navimumbai"];
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export default function App() {
  const mapDiv = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const chipIds = useRef<string[]>([]);
  const [query, setQuery] = useState("");
  const [aoi, setAoi] = useState("");
  const [sites, setSites] = useState<Site[]>([]);
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const m = new maplibregl.Map({
      container: mapDiv.current!,
      style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#1b1f24" } }] },
      center: [72.2, 22.3],
      zoom: 9,
    });
    m.on("load", () => {
      m.addSource("sites", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "sites-line", type: "line", source: "sites", paint: { "line-color": "#f5b400", "line-width": 2 } });
      m.addLayer({
        id: "sites-sel",
        type: "line",
        source: "sites",
        filter: ["==", ["get", "tile_id"], ""],
        paint: { "line-color": "#ff3b30", "line-width": 4 },
      });
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
      m.addSource(id, {
        type: "image",
        url: s.chip_url,
        coordinates: [ring[3], ring[2], ring[1], ring[0]] as [number, number][] as never,
      });
      m.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": 0.95 } }, "sites-line");
      chipIds.current.push(id);
    });
    (m.getSource("sites") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: sites.map((s) => ({ type: "Feature", geometry: s.geometry, properties: { tile_id: s.tile_id, rank: s.rank } })),
    });
    if (sites.length) {
      const xs = sites.flatMap((s) => s.geometry.coordinates[0].map((c: GeoJSON.Position) => c[0]));
      const ys = sites.flatMap((s) => s.geometry.coordinates[0].map((c: GeoJSON.Position) => c[1]));
      m.fitBounds([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] as LngLatBoundsLike, { padding: 60 });
    }
  }, [sites, ready]);

  useEffect(() => {
    const m = map.current;
    if (!m || !m.getLayer("sites-sel")) return;
    m.setFilter("sites-sel", ["==", ["get", "tile_id"], selected ?? ""]);
    const s = sites.find((x) => x.tile_id === selected);
    if (s) m.flyTo({ center: s.geometry.coordinates[0][0] as [number, number], zoom: 13 });
  }, [selected, sites]);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setStatus("searching…");
    const t0 = performance.now();
    try {
      const r = await fetch("/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, aoi: aoi || null }),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const body = await r.json();
      setSites(body.sites);
      setParsed(body.parsed);
      setWarnings(body.warnings ?? []);
      setSelected(null);
      setStatus(`${body.sites.length} sites · ${Math.round(performance.now() - t0)} ms`);
    } catch (err) {
      setStatus(`error: ${(err as Error).message}`);
    }
  }

  const sel = sites.find((s) => s.tile_id === selected);
  const understood = parsed && [
    parsed.change ? `change: ${parsed.change_types.join("/") || "any type"}` : null,
    Object.keys(parsed.near).length ? `near ${Object.entries(parsed.near).map(([t, m]) => `${t} ≤${m / 1000} km`).join(", ")}` : null,
    parsed.date_range ? `dates ${parsed.date_range[0] ?? "…"} → ${parsed.date_range[1] ?? "…"}` : null,
    parsed.aoi ? `in ${parsed.aoi}` : null,
    `meaning: “${parsed.semantic}”`,
  ].filter(Boolean).join(" · ");
  return (
    <div className="app">
      <aside>
        <h1>EO Archive Search</h1>
        <form onSubmit={search}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. salt pans, industrial area, river" autoFocus />
          <select value={aoi} onChange={(e) => setAoi(e.target.value)} aria-label="Area of interest">
            {AOIS.map((a) => (
              <option key={a} value={a}>{a || "all AOIs"}</option>
            ))}
          </select>
          <button type="submit">Search</button>
        </form>
        <p className="status">{status}</p>
        {understood && <p className="note">Understood: {understood}</p>}
        {warnings.map((w) => (
          <p key={w} className="note warn">{WARNING_TEXT[w] ?? w}</p>
        ))}
        <p className="note">Sentinel-2, 10 m: finds land cover and large structures, not individual vehicles or small buildings.</p>
        <ol className="results">
          {sites.map((s) => (
            <li key={s.tile_id} className={s.tile_id === selected ? "sel" : ""} onClick={() => setSelected(s.tile_id)}>
              {s.chip_url ? <img src={s.chip_url} alt="" width={64} height={64} /> : <div className="nochip" />}
              <div>
                <b>#{s.rank}</b> {s.tile_id}
                <br />
                {s.season ?? "—"} · score {s.score.toFixed(3)}
                {s.q_mean !== null && <> · q {s.q_mean.toFixed(2)}</>}
                {s.events.length > 0 && (
                  <>
                    <br />
                    {s.events.length} event(s): {[...new Set(s.events.map((e) => e.change_type))].join(", ")} · from{" "}
                    {s.events.map((e) => e.earliest_supported_date).sort()[0]}
                  </>
                )}
                {"water" in s.dist_m && <> · water {Math.round(s.dist_m.water)} m</>}
              </div>
            </li>
          ))}
        </ol>
        {sel && (
          <section className="prov">
            <h2>Provenance</h2>
            <p>Scenes: {sel.provenance.scene_ids.join(", ")}</p>
            <p>Models: {Object.entries(sel.provenance.model_versions).map(([k, v]) => `${k}=${v}`).join(", ") || "—"}</p>
            <p>Cascade: {sel.provenance.cascade_version?.join(", ") ?? "—"}</p>
            <p>Score parts: {Object.entries(sel.parts).map(([k, v]) => `${k} ${v}`).join(" · ")}</p>
            <p>Tiles in site: {sel.member_tiles.join(", ")}</p>
            {sel.events.length > 0 && <p>Events: {sel.events.map((e) => e.event_id).join(", ")}</p>}
          </section>
        )}
      </aside>
      <main ref={mapDiv} />
    </div>
  );
}
