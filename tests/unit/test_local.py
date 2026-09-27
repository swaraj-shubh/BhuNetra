"""M3b: staging organiser GeoTIFF/COG (ingest/local.py) + the generic scene loader."""

import json

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from ingest import local, scene


def _tif(
    path,
    arrays,
    crs="EPSG:3857",
    res=2.5,
    origin=(8_030_000.0, 2_540_000.0),
    descriptions=None,
    tags=None,
    dtype="uint16",
    nodata=0,
):
    arrays = [np.asarray(a, dtype) for a in arrays]
    h, w = arrays[0].shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=len(arrays),
        dtype=dtype,
        crs=crs,
        transform=from_origin(*origin, res, res),
        nodata=nodata,
    ) as d:
        for i, a in enumerate(arrays, 1):
            d.write(a, i)
            if descriptions:
                d.set_band_description(i, descriptions[i - 1])
        if tags:
            d.update_tags(**tags)
    return path


@pytest.mark.parametrize(
    ("name", "tags", "want"),
    [
        ("S2A_MSIL2A_20240315T053641_N0510_R005_T43QBE.tif", {}, "2024-03-15"),
        ("scene_2023-11-02_rgbn.tif", {}, "2023-11-02"),
        ("liss4_2022_05_01.tif", {}, "2022-05-01"),
        ("mosaic.tif", {"TIFFTAG_DATETIME": "2021:07:14 10:22:00"}, "2021-07-14"),
        ("mosaic.tif", {}, None),
        ("scan_19990101_x.tif", {}, "1999-01-01"),
        ("id_20241399.tif", {}, None),  # not a real date
    ],
)
def test_detect_date(name, tags, want):
    assert local.detect_date(name, tags) == want


def test_sensor_family_from_hints():
    assert local.sensor_family("S2B_MSIL2A_x.tif", {}) == "sentinel2"
    assert local.sensor_family("LC09_L2SP_146045.tif", {}) == "landsat89"
    assert local.sensor_family("R2_LISS4_scene.tif", {}) == "liss"
    assert local.sensor_family("ortho.tif", {}) is None


def test_band_tokens_are_ambiguous_without_a_sensor():
    # "B2" is blue for Sentinel-2 but green for LISS: without a known sensor it must not be mapped
    assert local.canonical_for("B2", "sentinel2") == "blue"
    assert local.canonical_for("B2", "liss") == "green"
    assert local.canonical_for("B2", None) is None
    assert local.canonical_for("Red", None) == "red"  # generic names are always fine


def test_stage_multiband_foreign_crs_and_gsd(tmp_path):
    h = w = 200
    red = np.full((h, w), 1200)
    nir = np.full((h, w), 3000)
    src = _tif(
        tmp_path / "ortho_2024-02-10.tif",
        [red, red, red, nir],
        descriptions=["red", "green", "blue", "nir"],
        tags={"scale_factor": "0.0001"},
    )
    out = local.stage(
        [src],
        aoi="orgx",
        source="organiser",
        manifest=None,
        raw_root=tmp_path / "raw",
        aoi_dir=tmp_path / "aoi",
    )
    assert [r["status"] for r in out] == ["staged"]
    cfg = yaml.safe_load((tmp_path / "aoi" / "orgx.yaml").read_text(encoding="utf-8"))
    assert cfg["optical_source"] == "organiser" and len(cfg["bbox"]) == 4
    folder = tmp_path / "raw" / "organiser" / "orgx" / "2024-02-10"
    meta = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    assert meta["gsd"] == pytest.approx(2.5, abs=0.3)  # native resolution kept (EPSG:3857 metres at ~22°N)
    assert set(meta["canonical"]) == {"red", "green", "blue", "nir"}
    assert meta["cloud_mask"] == "none"
    assert meta["source_sha256"] == {src.name: local.sha256(src)}  # original file provenance (§2.7)
    obs = scene.load(folder)
    assert obs.scene_id == src.name and "swir1" not in obs.refl
    # the AOI grid snaps outward to STRIDE_M, so only the image footprint is valid
    assert 0.1 < obs.valid.mean() < 1 and not obs.cloud.any()
    assert np.allclose(obs.refl["red"][obs.valid], 0.12, atol=1e-6)  # scale from the TIFF tag


def test_integer_dn_without_scale_is_rejected(tmp_path):
    src = _tif(
        tmp_path / "x_2024-02-10.tif", [np.ones((50, 50)) * 900] * 3, descriptions=["red", "green", "blue"]
    )
    with pytest.raises(local.LocalStageError, match="scale"):
        local.stage(
            [src],
            aoi="orgx",
            source="organiser",
            manifest=None,
            raw_root=tmp_path / "raw",
            aoi_dir=tmp_path / "aoi",
        )


def test_missing_date_is_rejected_with_manifest_hint(tmp_path):
    src = _tif(
        tmp_path / "mosaic.tif",
        [np.ones((50, 50))] * 3,
        descriptions=["red", "green", "blue"],
        dtype="float32",
        nodata=None,
    )
    with pytest.raises(local.LocalStageError, match="--manifest"):
        local.stage(
            [src],
            aoi="orgx",
            source="organiser",
            manifest=None,
            raw_root=tmp_path / "raw",
            aoi_dir=tmp_path / "aoi",
        )


def test_manifest_supplies_date_bands_and_scale(tmp_path):
    src = _tif(tmp_path / "mosaic.tif", [np.full((50, 50), 800), np.full((50, 50), 2500)])
    man = {"date": "2023-12-01", "bands": {"red": 1, "nir": 2}, "scale": 0.0001, "offset": 0.0}
    out = local.stage(
        [src],
        aoi="orgx",
        source="organiser",
        manifest=man,
        raw_root=tmp_path / "raw",
        aoi_dir=tmp_path / "aoi",
    )
    assert out[0]["status"] == "staged"
    obs = scene.load(tmp_path / "raw" / "organiser" / "orgx" / "2023-12-01")
    assert obs.valid.any() and np.allclose(obs.refl["nir"][obs.valid], 0.25)


def test_restaging_is_a_noop_and_never_overwrites(tmp_path):
    src = _tif(
        tmp_path / "a_2024-02-10.tif",
        [np.full((50, 50), 0.1)] * 3,
        descriptions=["red", "green", "blue"],
        dtype="float32",
        nodata=None,
    )
    kw = {
        "aoi": "orgx",
        "source": "organiser",
        "manifest": None,
        "raw_root": tmp_path / "raw",
        "aoi_dir": tmp_path / "aoi",
    }
    assert local.stage([src], **kw)[0]["status"] == "staged"
    assert local.stage([src], **kw)[0]["status"] == "exists"  # raw is write-once (§2.12)


def test_existing_aoi_with_other_optical_source_is_rejected(tmp_path):
    # a hand-written AOI file without optical_source would make `make ingest` ignore the staged files
    (tmp_path / "aoi").mkdir()
    (tmp_path / "aoi" / "orgy.yaml").write_text(
        "name: orgy\nbbox: [73.0, 19.0, 73.1, 19.1]\n", encoding="utf-8"
    )
    red = np.full((50, 50), 1200)
    src = _tif(
        tmp_path / "x_2024-02-10.tif",
        [red, red, red, red],
        descriptions=["red", "green", "blue", "nir"],
        tags={"scale_factor": "0.0001"},
    )
    with pytest.raises(local.LocalStageError, match="optical_source"):
        local.stage(
            [src], aoi="orgy", source="organiser", raw_root=tmp_path / "raw", aoi_dir=tmp_path / "aoi"
        )
