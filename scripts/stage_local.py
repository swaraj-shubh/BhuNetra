"""Stage organiser / local GeoTIFF or COG files (M3b). Offline; the original files are only read.

    python scripts/stage_local.py <file-or-folder> [...] --aoi <name> [--source organiser] [--manifest m.yaml]
    make stage-local FILES="/media/usb/*.tif" AOI=finals

Then: `make ingest AOI=<name>` and `make change AOI=<name>` as for any AOI. See ingest/local.py for what is read
from the files and what a manifest can supply (date, sensor, bands, scale/offset, nodata).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from configs.settings import paths
from ingest import local

sys.path.insert(0, str(Path(__file__).parent))
from stage_data import _manifest_upsert  # sibling script: shares the manifests writer (§9)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--source", default="organiser")
    ap.add_argument("--manifest", type=Path)
    a = ap.parse_args()
    files = sorted(
        {f for p in a.inputs for f in ([p] if p.is_file() else [*p.rglob("*.tif"), *p.rglob("*.tiff")])}
    )
    if not files:
        raise SystemExit("no .tif/.tiff inputs found")
    man = yaml.safe_load(a.manifest.read_text(encoding="utf-8")) if a.manifest else None
    try:
        results = local.stage(files, aoi=a.aoi, source=a.source, manifest=man)
    except local.LocalStageError as e:
        raise SystemExit(f"cannot stage: {e}") from e
    _manifest_upsert(
        "datasets.yaml",
        f"{a.source}/{a.aoi}",
        {
            "source": "local files (organiser-provided)",
            "licence": "as provided by the organisers",
            "inputs": {f.name: local.sha256(f) for f in files},
            "path": f"data/raw/{a.source}/{a.aoi}",
            "dates": [r["date"] for r in results],
        },
    )
    print(
        json.dumps(
            {"aoi": a.aoi, "source": a.source, "results": results, "raw": str(paths.raw / a.source / a.aoi)}
        )
    )


if __name__ == "__main__":
    main()
