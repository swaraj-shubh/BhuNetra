"""DESIGN.md §11: one synthetic case per change type + "other"."""

import pytest

from change.classify import Evidence, classify
from configs.settings import T

C = T.classify


def ev(**kw):
    base = {
        "d_ndvi": 0.0,
        "d_mndwi": 0.0,
        "d_ndbi": 0.0,
        "d_bsi": 0.0,
        "water_pre": 0.0,
        "water_post": 0.0,
        "elongation": 1.5,
        "width_px": 10.0,
    }
    return Evidence(**(base | kw))


@pytest.mark.parametrize(
    ("e", "ctype", "direction"),
    [
        (ev(d_ndvi=C.CONS_DNDVI - 0.05, d_ndbi=C.CONS_DNDBI + 0.05), "construction", "appearance"),
        (ev(d_ndvi=C.CLR_DNDVI - 0.1, d_bsi=C.CLR_DBSI + 0.05), "clearance", "disappearance"),
        (ev(d_mndwi=C.WATER_DMNDWI + 0.1, water_pre=0.0, water_post=0.8), "water", "appearance"),
        (ev(d_mndwi=C.WATER_DMNDWI + 0.1, water_pre=0.4, water_post=0.9), "water", "expansion"),
        (ev(d_mndwi=-C.WATER_DMNDWI - 0.1, water_pre=0.9, water_post=0.3), "water", "contraction"),
        (ev(d_mndwi=-C.WATER_DMNDWI - 0.1, water_pre=0.8, water_post=0.0), "water", "disappearance"),
        (
            ev(d_ndvi=-0.2, d_ndbi=0.1, elongation=C.ROAD_ELONG + 2, width_px=C.ROAD_WIDTH_PX),
            "road",
            "appearance",
        ),
        (ev(d_ndvi=-0.05, d_ndbi=0.02), "other", "appearance"),
        (ev(d_ndvi=0.2), "other", "expansion"),  # greening: real but not a supported type
    ],
)
def test_one_case_per_type(e, ctype, direction):
    assert classify(e) == (ctype, direction)


def test_water_needs_extent_change_not_just_mndwi():
    # turbid water getting clearer: MNDWI moves but the water fraction does not change
    e = ev(d_mndwi=C.WATER_DMNDWI + 0.1, water_pre=0.9, water_post=0.9)
    assert classify(e)[0] != "water"


def test_construction_vs_clearance_takes_the_stronger_rule():
    # both fire; construction's NDBI margin is larger -> construction
    e = ev(d_ndvi=-0.3, d_ndbi=C.CONS_DNDBI + 0.2, d_bsi=C.CLR_DBSI + 0.01)
    assert classify(e)[0] == "construction"
    e2 = ev(d_ndvi=-0.3, d_ndbi=C.CONS_DNDBI + 0.01, d_bsi=C.CLR_DBSI + 0.2)
    assert classify(e2)[0] == "clearance"


def test_dark_surface_vegetation_loss_is_clearance_not_construction():
    # forest -> coal/mine face: NDBI rises past the construction rule, but MNDWI rises without water extent change
    e = ev(
        d_ndvi=-0.45, d_ndbi=0.16, d_bsi=0.10, d_mndwi=C.DARK_DMNDWI + 0.05, water_pre=0.0, water_post=0.02
    )
    assert classify(e) == ("clearance", "disappearance")


def test_road_needs_vegetation_loss():
    # elongated strip with only an NDBI rise (tidal flat, revegetated haul road) is not a road
    e = ev(d_ndvi=0.2, d_ndbi=0.1, elongation=C.ROAD_ELONG + 2, width_px=C.ROAD_WIDTH_PX)
    assert classify(e)[0] != "road"
