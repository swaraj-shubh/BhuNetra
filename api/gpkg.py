"""Minimal OGC GeoPackage 1.3 writer (stdlib sqlite3 + shapely WKB): no GDAL vector driver in the image, no new
dependency, offline. Feature tables in EPSG:4326; properties become columns (dict/list values as JSON text).
Opens in QGIS / GDAL (`ogrinfo export.gpkg`).
"""

import json
import sqlite3
import struct
import tempfile
from pathlib import Path
from typing import Any

import shapely
from shapely.geometry import shape

WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
)


def _blob(geom: dict[str, Any]) -> tuple[bytes, tuple[float, float, float, float]]:
    """GeoPackage geometry blob: 'GP', version 0, flags (little endian + xy envelope), srs_id, envelope, WKB."""
    g = shape(geom)
    x0, y0, x1, y1 = g.bounds
    head = b"GP" + bytes([0, 0b011]) + struct.pack("<i4d", 4326, x0, x1, y0, y1)
    return head + shapely.to_wkb(g, byte_order=1), (x0, y0, x1, y1)


def _sql_type(values: list[Any]) -> str:
    vs = [v for v in values if v is not None]
    if vs and all(isinstance(v, bool | int) for v in vs):
        return "INTEGER"
    if vs and all(isinstance(v, int | float) and not isinstance(v, bool) for v in vs):
        return "REAL"
    return "TEXT"


def _cell(v: Any) -> Any:
    return json.dumps(v) if isinstance(v, dict | list) else v


def write(layers: dict[str, list[dict[str, Any]]]) -> bytes:
    """layers: name -> GeoJSON features. Returns the .gpkg file bytes."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.gpkg"
        con = sqlite3.connect(path)
        con.execute("PRAGMA application_id = 1196444487")  # 'GPKG'
        con.execute("PRAGMA user_version = 10300")
        con.executescript(
            """
            CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
              organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL,
              description TEXT);
            CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
              identifier TEXT UNIQUE, description TEXT DEFAULT '',
              last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
              min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
            CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
              geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
              CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name));
            """
        )
        con.executemany(
            "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
                ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
                ("WGS 84 geodetic", 4326, "EPSG", 4326, WGS84_WKT, None),
            ],
        )
        for name, feats in layers.items():
            cols = sorted({k for f in feats for k in f["properties"]})
            types = {c: _sql_type([f["properties"].get(c) for f in feats]) for c in cols}
            coldefs = "".join(f', "{c}" {types[c]}' for c in cols)
            con.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB{coldefs})')
            ext = [float("inf"), float("inf"), float("-inf"), float("-inf")]
            rows = []
            for f in feats:
                blob, (x0, y0, x1, y1) = _blob(f["geometry"])
                ext = [min(ext[0], x0), min(ext[1], y0), max(ext[2], x1), max(ext[3], y1)]
                rows.append([blob, *(_cell(f["properties"].get(c)) for c in cols)])
            marks = ", ".join("?" * (len(cols) + 1))
            names = ", ".join(["geom", *(f'"{c}"' for c in cols)])
            con.executemany(f'INSERT INTO "{name}" ({names}) VALUES ({marks})', rows)
            con.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id) "
                "VALUES (?, 'features', ?, ?, ?, ?, ?, 4326)",
                (name, name, *(ext if feats else [None] * 4)),
            )
            kinds = {
                f["geometry"]["type"].upper() for f in feats
            }  # one declared type when uniform (QGIS styling)
            gtype = kinds.pop() if len(kinds) == 1 else "GEOMETRY"
            con.execute("INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', ?, 4326, 0, 0)", (name, gtype))
        con.commit()
        con.close()
        return path.read_bytes()
