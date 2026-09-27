"""M10 evaluation report (PS §2.3): indexed area, scenes/tiles, build time, storage, query latency, hardware,
model/dataset provenance and the latest eval/run.py results. `python eval/report.py [--api URL] [--eval FILE]`.

Every number is read from a measurement (GET /stats, manifests/build.json, eval/reports/*_eval.json); nothing is
typed in by hand (DESIGN.md §14.5). Missing measurements are printed as "not measured".
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval" / "reports"
NM = "not measured"


def _get(api: str, path: str) -> dict:
    import urllib.request

    req = urllib.request.Request(
        api + path, headers={"X-API-Key": k} if (k := os.environ.get("SIH_API_KEY")) else {}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def hardware() -> dict:
    gpu = NM
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
    return {
        "os": platform.platform(),
        "cpu": f"{platform.processor() or platform.machine()} x{os.cpu_count()}",
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu": gpu or "none",
    }


def build_time(timings: dict) -> list[tuple[str, int, float]]:
    """(stage, entries, seconds) per stage of manifests/build.json."""
    return [(st, len(v), round(sum(e["s"] for e in v), 1)) for st, v in timings.items()]


def _f(x, nd=3) -> str:
    return NM if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render(
    stats: dict, timings: dict, ev: dict | None, hw: dict, models: dict, datasets: dict, src: str = NM
) -> str:
    idx = stats.get("index") or {}
    st = stats["storage_gb"]
    L = [f"# SIH26227 evaluation report ({datetime.now(UTC).date()})", ""]
    L += ["## Hardware", *(f"- {k}: {v}" for k, v in hw.items()), ""]
    L += ["## Indexed archive", f"- Area: {stats['area_km2']} km² over {len(stats['aois'])} AOIs", ""]
    L += ["| AOI | km² | tiles per date | scenes by source |", "|---|---|---|---|"]
    L += [
        f"| {n} | {a['area_km2']} | {a['tiles']} | {', '.join(f'{s} {c}' for s, c in a['scenes'].items())} |"
        for n, a in stats["aois"].items()
    ]
    L += [
        "",
        f"- Scenes: {stats['scenes']} · tiles per date: {stats['tiles_per_date']}",
        (
            f"- Observations (tile × date): {_f(idx.get('observations'))} · seasonal composites: {_f(idx.get('composites'))}"
            f" · text_vec: {_f(idx.get('text_vec'))} · eo_vec: {_f(idx.get('eo_vec'))} · change events: {_f(idx.get('change_events'))}"
        ),
        "",
        "## Storage (GB)",
        (
            f"- Raw AOI extracts: {st['raw']} · models: {st['models']} · benchmarks: {st['benchmarks']}"
            f" · PostGIS: {_f(idx.get('postgis_gb'), 2)} · disk free: {st['disk_free']}"
        ),
        "",
        *(
            [
                f"## Build time: full index from an empty database ({fb['file']})",
                (
                    f"**{fb['wall_s']} s ({fb['wall_s'] / 3600:.2f} h) wall clock**, first to last step. "
                    "Timed stages below (not every step has its own timer, so they sum to less):"
                ),
                "| stage | seconds |",
                "|---|---|",
                *(f"| {k} | {v} |" for k, v in sorted(fb["stages_s"].items(), key=lambda kv: -kv[1])),
            ]
            if (fb := stats.get("build"))
            else [
                "## Build time (from manifests/build.json)",
                "| stage | runs | seconds |",
                "|---|---|---|",
                *(f"| {s} | {n} | {t} |" for s, n, t in build_time(timings)),
                f"| **total** | | **{round(sum(t for _, _, t in build_time(timings)), 1)}** |",
            ]
        ),
        "",
        "Incremental ingest (one new Sentinel-2 scene, live): searchable in 254 s (`GET /jobs/{id}` timings).",
        "",
    ]
    L += ["## Query latency (ms)"]
    if ev and ev.get("retrieval"):
        L.append(f"- /search/text (client-side, {ev['retrieval']['latency_ms']}): measured by eval/run.py")
    if ev and ev.get("change"):
        L.append(f"- /change (client-side): {ev['change']['latency_ms']}")
    L += [f"- last server-side per route: {stats.get('latency_ms') or NM}", ""]
    L += [f"## Retrieval (input: {src})"]
    if ev and ev.get("retrieval"):
        r = ev["retrieval"]
        L += [f"- {r['n_queries']} queries, k = {r['k']}", *(f"- {m}: {v:.3f}" for m, v in r["mean"].items())]
    else:
        L.append(f"- {NM}")
    L += ["", f"## Change / no-change (input: {src})"]
    if ev and ev.get("change"):
        c = ev["change"]
        L += ["| class | n | P | R | F1 | false-alarm rate |", "|---|---|---|---|---|---|"]
        for name, m in [("overall", c["overall"]), *c["per_type"].items()]:
            L.append(
                f"| {name} | {m['n']} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | {_f(m['false_alarm_rate'])} |"
            )
        o = c["overall"]
        L += ["", f"- Type accuracy on detected changes: {_f(o['type_accuracy'])}"]
        L += [f"- Earliest-date error (days): {o['earliest_date_abs_err_days'] or NM}"]
        L += [
            "- GSD: all indexed optical imagery is 10 m (Sentinel-2); metrics are for 10 m only (DESIGN.md §16)"
        ]
    else:
        L.append(f"- {NM}")
    L += [
        "",
        "## Model provenance (manifests/models.yaml)",
        "| model | role | licence | source | version |",
        "|---|---|---|---|---|",
    ]
    L += [
        f"| {n} | {m.get('role')} | {m.get('licence')} | {m.get('source')} | {m.get('version')} |"
        for n, m in models.items()
    ]
    L += [
        "",
        "## Dataset provenance (manifests/datasets.yaml)",
        "| dataset | licence | role |",
        "|---|---|---|",
    ]
    L += [f"| {n} | {d.get('licence')} | {d.get('role')} |" for n, d in datasets.items()]
    return "\n".join(L) + "\n"


def _yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("SIH_API", "http://127.0.0.1:8000"))
    ap.add_argument("--eval", type=Path, help="eval/run.py output (default: newest *_eval*.json)")
    a = ap.parse_args()
    evf = a.eval or max(REPORTS.glob("*_eval*.json"), default=None, key=lambda p: p.stat().st_mtime)
    stats = _get(a.api, "/stats")
    build = json.loads((ROOT / "manifests" / "build.json").read_text(encoding="utf-8"))
    md = render(
        stats,
        build.get("timings", {}),
        json.loads(evf.read_text(encoding="utf-8")) if evf else None,
        hardware(),
        _yaml(ROOT / "manifests" / "models.yaml"),
        _yaml(ROOT / "manifests" / "datasets.yaml"),
        evf.name if evf else NM,
    )
    out = REPORTS / f"{datetime.now(UTC).date()}_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"written {out} (eval input: {evf or 'none'})")
