import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from configs.settings import T, bands
from ingest import quality, scene, tiling

GSD = 10.0
ORIGIN = (320_000.0, 2_000_000.0)  # multiple of STRIDE_M, so tile (0, 0) starts at pixel (0, 0)


def test_q_formula_matches_design_doc():
    sun_low = T.quality.SUN_ELEV_MIN - 1
    q = quality.q_score(
        valid_frac=np.array([1.0, 0.8, 1.0, 1.0, 1.0]),
        shift_px=np.array([0.0, 0.0, 0.4, 2.0, 0.0]),
        cloud_near=np.array([0.0, 0.0, 0.0, 0.0, 0.5]),
        sun_elev=np.array([60.0, 60.0, 60.0, 60.0, sun_low]),
    )
    assert q == pytest.approx([1.0, 0.8, 0.6, 0.0, 0.75 * 0.7])


def test_unmeasured_shift_counts_as_zero():
    assert quality.q_score(1.0, np.nan, 0.0, 60.0) == pytest.approx(1.0)


def test_tile_fractions_valid_and_cloud_near():
    px = int(T.tiling.TILE_M / GSD)
    stride = int(T.tiling.STRIDE_M / GSD)
    h = w = px + 5 * stride  # a few tiles in each direction
    transform = from_origin(*ORIGIN, GSD, GSD)
    bounds = (ORIGIN[0], ORIGIN[1] - h * GSD, ORIGIN[0] + w * GSD, ORIGIN[1])
    tl = list(tiling.tiles("t", bounds, 32643))
    valid = np.ones((h, w), bool)
    valid[: px // 2, :px] = False  # top half of tile (0, 0) invalid
    cloud = np.zeros((h, w), bool)
    cloud[px + stride - 1, px + stride - 1] = True  # outside tile (0, 0), inside its one-stride buffer

    vf, cn = quality.tile_fractions(valid, cloud, transform, tl)
    first = tl.index(next(t for t in tl if (t.row, t.col) == (0, 0)))
    assert vf[first] == pytest.approx(0.5)
    assert cn[first] > 0  # cloud outside the tile but inside the near-buffer counts
    far = next(i for i, t in enumerate(tl) if (t.row, t.col) == (5, 5))
    assert cn[far] == 0 and vf[far] == 1.0


def _write_scene(folder, dn, scl, sun=60.0):
    folder.mkdir(parents=True)
    h, w = scl.shape
    prof = {"driver": "GTiff", "height": h, "width": w, "count": 1, "crs": "EPSG:32643"}
    prof["transform"] = from_origin(*ORIGIN, GSD, GSD)
    meta = {
        "aoi": "t",
        "sensor": "sentinel2",
        "date": "2024-01-05",
        "epsg": 32643,
        "gsd": GSD,
        "source_items": ["S2A_X", "S2A_Y"],
        "properties": {"view:sun_elevation": sun},
        "bands": {},
    }
    from ingest.validate import sha256

    for name in bands["sentinel2"]["bands"].values():
        with rasterio.open(folder / f"{name}.tif", "w", dtype="uint16", nodata=0, **prof) as d:
            d.write(dn, 1)
        meta["bands"][name] = {
            "file": f"{name}.tif",
            "sha256": sha256(folder / f"{name}.tif"),
            "scale": 0.0001,
            "offset": -0.1,
            "nodata": 0.0,
        }
    for name, arr in (("SCL", scl), ("CLD", np.zeros_like(scl))):
        with rasterio.open(folder / f"{name}.tif", "w", dtype="uint8", nodata=0, **prof) as d:
            d.write(arr, 1)
        meta["bands"][name] = {
            "file": f"{name}.tif",
            "sha256": sha256(folder / f"{name}.tif"),
            "scale": 1.0,
            "offset": 0.0,
            "nodata": 0.0,
        }
    (folder / "scene.json").write_text(json.dumps(meta), encoding="utf-8")


def test_load_s2_reflectance_masks_and_ids(tmp_path):
    dn = np.full((64, 64), 2000, np.uint16)
    dn[0, 0] = 0  # nodata
    scl = np.full((64, 64), 4, np.uint8)
    scl[40, 40] = 9  # cloud
    _write_scene(tmp_path / "2024-01-05", dn, scl)

    obs = scene.load_s2(tmp_path / "2024-01-05")
    assert obs.scene_id == "S2A_X+S2A_Y" and obs.sun_elev == 60.0
    assert obs.refl["red"][1, 1] == pytest.approx(0.1)  # (2000 * 1e-4) - 0.1
    assert not obs.valid[0, 0] and not obs.valid[40, 40] and obs.valid[10, 10]
    assert obs.cloud[40, 40] and not obs.cloud[10, 10]
    assert obs.transform == from_origin(*ORIGIN, GSD, GSD)


def test_load_s2_rejects_tampered_band(tmp_path):
    scl = np.full((64, 64), 4, np.uint8)
    _write_scene(tmp_path / "s", np.full((64, 64), 2000, np.uint16), scl)
    with rasterio.open(tmp_path / "s" / "B04.tif", "r+") as d:
        d.write(np.full((64, 64), 9, np.uint16), 1)
    with pytest.raises(scene.SceneRejected, match="sha256"):
        scene.load_s2(tmp_path / "s")
