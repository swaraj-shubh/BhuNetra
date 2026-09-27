"""api/gpkg.py: a valid GeoPackage 1.3 container that round-trips geometry and attributes."""

import sqlite3
import struct

import shapely
from shapely.geometry import box, mapping

from api import gpkg


def test_gpkg_structure_and_round_trip(tmp_path):
    feats = [
        {
            "type": "Feature",
            "geometry": mapping(box(73.0, 19.0, 73.01, 19.01)),
            "properties": {
                "event_id": "CHG-000001",
                "confidence": 0.8,
                "n": 3,
                "provenance": {"scene_ids": ["a"]},
            },
        },
        {
            "type": "Feature",
            "geometry": mapping(box(73.1, 19.1, 73.2, 19.2)),
            "properties": {"event_id": "CHG-000002", "confidence": 0.5, "n": None},
        },
    ]
    f = tmp_path / "x.gpkg"
    f.write_bytes(gpkg.write({"events": feats, "sites": []}))
    con = sqlite3.connect(f)
    assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
    assert con.execute("PRAGMA user_version").fetchone()[0] == 10300
    assert {r[0] for r in con.execute("SELECT table_name FROM gpkg_contents")} == {"events", "sites"}
    ext = con.execute(
        "SELECT min_x, min_y, max_x, max_y FROM gpkg_contents WHERE table_name='events'"
    ).fetchone()
    assert ext == (73.0, 19.0, 73.2, 19.2)
    blob, eid, conf, prov = con.execute(
        "SELECT geom, event_id, confidence, provenance FROM events ORDER BY fid"
    ).fetchone()
    assert blob[:2] == b"GP" and struct.unpack("<i", blob[4:8])[0] == 4326
    assert shapely.from_wkb(blob[40:]).equals(box(73.0, 19.0, 73.01, 19.01))
    assert (eid, conf, prov) == ("CHG-000001", 0.8, '{"scene_ids": ["a"]}')
    types = dict(con.execute("SELECT table_name, geometry_type_name FROM gpkg_geometry_columns"))
    assert types == {"events": "POLYGON", "sites": "GEOMETRY"}
