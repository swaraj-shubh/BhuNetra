import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, LngLatBoundsLike, MapLayerMouseEvent } from "maplibre-gl";

// DESIGN.md §12: every site carries provenance. No basemap: the demo runs offline (§2.8).
type Site = {
  rank: number;
  tile_id: string;
  aoi: string;
  season: string;
  score: number;
  q_mean: number;
  geometry: GeoJSON.Polygon;
  chip_url: string;
  provenance: { scene_ids: string[]; model_versions: Record<string, string>; cascade_version: string | null };
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
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState("");

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
      m.on("click", "sites-line", (e: MapLayerMouseEvent) => setSelected(String(e.features?.[0]?.properties?.tile_id ?? "")));
    });
    map.current = m;
    return () => m.remove();
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!m || !m.isStyleLoaded()) return;
    for (const id of chipIds.current) {
      if (m.getLayer(id)) m.removeLayer(id);
      if (m.getSource(id)) m.removeSource(id);
    }
    chipIds.current = [];
    sites.forEach((s) => {
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
  }, [sites]);

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
      setSelected(null);
      setStatus(`${body.sites.length} sites · ${Math.round(performance.now() - t0)} ms`);
    } catch (err) {
      setStatus(`error: ${(err as Error).message}`);
    }
  }

  const sel = sites.find((s) => s.tile_id === selected);
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
        <p className="note">Sentinel-2, 10 m: finds land cover and large structures, not individual vehicles or small buildings.</p>
        <ol className="results">
          {sites.map((s) => (
            <li key={s.tile_id} className={s.tile_id === selected ? "sel" : ""} onClick={() => setSelected(s.tile_id)}>
              <img src={s.chip_url} alt="" width={64} height={64} />
              <div>
                <b>#{s.rank}</b> {s.tile_id}
                <br />
                {s.season} · score {s.score.toFixed(3)} · q {s.q_mean.toFixed(2)}
              </div>
            </li>
          ))}
        </ol>
        {sel && (
          <section className="prov">
            <h2>Provenance</h2>
            <p>Scenes: {sel.provenance.scene_ids.join(", ")}</p>
            <p>Models: {Object.entries(sel.provenance.model_versions).map(([k, v]) => `${k}=${v}`).join(", ")}</p>
          </section>
        )}
      </aside>
      <main ref={mapDiv} />
    </div>
  );
}
