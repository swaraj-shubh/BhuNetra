"""ONLINE staging: the only code allowed to touch the network (DESIGN.md §2.8). Everything is resumable.

python scripts/stage_data.py [--workers N] all                # everything below, every AOI
python scripts/stage_data.py series --source sentinel2 --aoi navimumbai [--start ... --end ...]
python scripts/stage_data.py series --source all --aoi all
python scripts/stage_data.py osm --aoi all
python scripts/stage_data.py benchmarks | models | web
"""

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from configs.settings import aois, models, paths, sources, stac_url
from governance import lineage
from ingest import stac
from ingest.validate import sha256

log = logging.getLogger("stage")
STAGING = sources["staging"]


def _manifest_upsert(file: str, key: str, entry: dict) -> None:
    """manifests/<file>: mapping key -> entry (DESIGN.md §9). Re-staging overwrites the entry."""
    path = paths.manifests / file
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or {}
    data[key] = entry
    path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")


def _sha256_files(dest: Path, min_bytes: float = 0) -> dict[str, str]:
    return {
        p.relative_to(dest).as_posix(): sha256(p)
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.stat().st_size > min_bytes and ".cache" not in p.parts and p.name != "SHA256SUMS"
    }


def stage_models() -> None:
    api = HfApi()
    for name, m in models["downloads"].items():
        dest = paths.models / name
        snapshot_download(m["repo"], allow_patterns=m["allow"], local_dir=dest)
        sums = _sha256_files(dest, min_bytes=1e6)
        (dest / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()), encoding="utf-8")
        info = api.model_info(m["repo"])
        lic = (getattr(info.card_data, "license", None) if info.card_data else None) or "VERIFY"
        _manifest_upsert(
            "models.yaml",
            name,
            {
                "source": f"https://huggingface.co/{m['repo']}",
                "version": info.sha,
                "licence": lic,
                "role": m["role"],
                "path": f"models/{name}",
                "sha256": sums,
            },
        )
        log.info("model %s: %d files, licence %s", name, len(sums), lic)


def _overpass(query: str) -> dict:
    """POST an Overpass query, rotating mirrors with backoff (busy servers answer 429/5xx)."""
    osm = sources["osm"]
    last = None
    for attempt in range(osm["attempts"]):
        url = osm["urls"][attempt % len(osm["urls"])]
        req = urllib.request.Request(
            url,
            data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": osm["user_agent"]},
        )
        try:
            with urllib.request.urlopen(req, timeout=osm["timeout_s"]) as r:
                return json.loads(r.read().decode("utf-8"))
        except (OSError, ValueError) as ex:
            last = ex
            log.warning("overpass %s attempt %d: %s", url, attempt + 1, ex)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"overpass failed: {last}")


def _ways_to_geojson(data: dict, date: str) -> dict:
    return {
        "type": "FeatureCollection",
        "properties": {"osm_date": date, "osm_base": data.get("osm3s", {}).get("timestamp_osm_base")},
        "features": [
            {
                "type": "Feature",
                "id": f"way/{w['id']}",
                "properties": w.get("tags", {}),
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p["lon"], p["lat"]] for p in w["geometry"]],
                },
            }
            for w in data["elements"]
            if w.get("type") == "way" and len(w.get("geometry", [])) >= 2
        ],
    }


def stage_osm(aoi: str) -> None:
    w, s, e, n = aois[aoi].bbox
    osm = sources["osm"]
    dest = paths.raw / "osm" / aoi
    dest.mkdir(parents=True, exist_ok=True)
    for date in osm["snapshots"]:
        path = dest / f"roads_{date}.geojson"
        if path.exists():
            continue
        try:
            fc = _ways_to_geojson(_overpass(osm["query"].format(date=date, s=s, w=w, n=n, e=e)), date)
        except RuntimeError as ex:
            log.warning("osm %s %s FAILED: %s", aoi, date, ex)
            continue
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(fc), encoding="utf-8")
        tmp.rename(path)
        log.info("osm %s %s: %d ways", aoi, date, len(fc["features"]))
        time.sleep(osm["pause_s"])
    lineage.append(
        "staging",
        lineage.record("stage_osm", {"source": osm["urls"][0]}, {"aoi": aoi, "snapshots": osm["snapshots"]}),
    )
    _manifest_upsert(
        "datasets.yaml",
        f"osm/{aoi}",
        {
            "source": f"{osm['urls'][0]} (attic [date:] queries)",
            "licence": "ODbL (c) OpenStreetMap contributors",
            "role": "road-change labels",
            "path": f"data/raw/osm/{aoi}",
            "snapshots": osm["snapshots"],
            "files": sorted(p.name for p in dest.glob("*.geojson")),
            "sha256": _sha256_files(dest),
        },
    )


def stage_benchmarks() -> None:
    for name, b in sources["benchmarks"].items():
        dest = paths.benchmarks / name
        snapshot_download(b["hf_dataset"], repo_type="dataset", local_dir=dest, allow_patterns=b["allow"])
        sums = _sha256_files(dest)
        (dest / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()), encoding="utf-8")
        _manifest_upsert(
            "datasets.yaml",
            f"benchmark/{name}",
            {
                "source": f"https://huggingface.co/datasets/{b['hf_dataset']}",
                "licence": b["licence"],
                "role": b["role"],
                "path": f"data/benchmarks/{name}",
                "sha256": sums,
            },
        )
        log.info("benchmark %s -> %s (licence %s)", name, dest, b["licence"])


def stage_web() -> None:
    v = sources["web_vendor"]
    dest = paths.web / "vendor"
    dest.mkdir(parents=True, exist_ok=True)
    for f in v["files"]:
        urllib.request.urlretrieve(v["base_url"] + f, dest / f)


def stage_series(names: list[str], aoi_names: list[str], start: str, end: str, workers: int) -> list[dict]:
    reports = []
    for source in names:
        for aoi in aoi_names:
            try:
                r = stac.stage_series(source, aoi, start, end, workers=workers)
            except Exception as e:  # noqa: BLE001 - one bad source must not stop the rest; reported below
                r = {"source": source, "aoi": aoi, "error": str(e)}
            log.info("REPORT %s", json.dumps(r))
            reports.append(r)
            if "error" not in r:
                src = sources["sources"][source]
                _manifest_upsert(
                    "datasets.yaml",
                    f"{source}/{aoi}",
                    {
                        "source": f"{stac_url(source)}/collections/{src['collection']}",
                        "licence": src["licence"],
                        "path": f"data/raw/{source}/{aoi}",
                        "acquisitions": len(stac.staged(source, aoi)),
                        "range": [start, end],
                        "failed": r["failed"],
                        "months_without_acquisition": r["months_without_acquisition"],
                        "sha256": "per file in each <date>/scene.json",
                    },
                )
    paths.logs.mkdir(parents=True, exist_ok=True)
    (paths.logs / "staging_report.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    return reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("rasterio", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workers", type=int, default=STAGING["workers"], help="parallel acquisitions (lower = less RAM)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    se = sub.add_parser("series")
    se.add_argument("--source", required=True, help=f"all | {' | '.join(sources['sources'])}")
    se.add_argument("--aoi", required=True, help="all | AOI name")
    se.add_argument("--start", default=STAGING["series_start"])
    se.add_argument("--end", default=STAGING["series_end"])
    sub.add_parser("osm").add_argument("--aoi", default="all")
    for c in ("benchmarks", "models", "web", "all"):
        sub.add_parser(c)
    a = ap.parse_args()

    aoi_names = list(aois) if getattr(a, "aoi", "all") == "all" else [a.aoi]
    if a.cmd == "series":
        stage_series(
            STAGING["order"] if a.source == "all" else [a.source], aoi_names, a.start, a.end, a.workers
        )
    elif a.cmd == "osm":
        for aoi in aoi_names:
            stage_osm(aoi)
    elif a.cmd == "benchmarks":
        stage_benchmarks()
    elif a.cmd == "models":
        stage_models()
    elif a.cmd == "web":
        stage_web()
    else:
        stage_web()
        stage_models()
        stage_benchmarks()
        # imagery first: OSM depends on a shared public Overpass server that is often overloaded
        stage_series(STAGING["order"], aoi_names, STAGING["series_start"], STAGING["series_end"], a.workers)
        for aoi in aoi_names:
            stage_osm(aoi)


if __name__ == "__main__":
    main()
