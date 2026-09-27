import pytest

from configs.settings import Fusion, T
from query.dedup import dedup
from query.fuse import Candidate, fuse, spatial_fit
from query.parser import parse

NEAR = T.context.NEAR_M


@pytest.mark.parametrize(
    ("text", "change", "types", "near", "dates", "aoi", "sensor", "warn", "semantic_has"),
    [
        # PS examples (DESIGN.md §11)
        (
            "newly built structures near a river",
            True,
            ("construction",),
            {"water": NEAR},
            None,
            None,
            None,
            False,
            "structures",
        ),
        (
            "large vehicle concentrations on open ground",
            False,
            (),
            {},
            None,
            None,
            None,
            True,
            "vehicle concentrations",
        ),
        # change + time + aoi
        ("new roads since 2022", True, ("road",), {}, ("2022-01-01", None), None, None, False, "roads"),
        (
            "forest cleared between 2021 and 2023 in hasdeo",
            True,
            ("clearance",),
            {},
            ("2021-01-01", "2023-12-31"),
            "hasdeo",
            None,
            False,
            "forest",
        ),
        (
            "flooding along the coast in 2024",
            True,
            ("water",),
            {"water": NEAR},
            ("2024-01-01", "2024-12-31"),
            None,
            None,
            False,
            "flooding",
        ),
        (
            "buildings under construction near a lake",
            True,
            ("construction",),
            {"water": NEAR},
            None,
            None,
            None,
            False,
            "buildings",
        ),
        # no change intent: pure semantic, with/without spatial relation
        ("airport runway", False, (), {}, None, None, None, False, "airport runway"),
        (
            "industrial buildings within 2 km of a highway",
            False,
            (),
            {"road": 2000.0},
            None,
            None,
            None,
            False,
            "industrial buildings",
        ),
        # a type word plus a time phrase is a change question even without a cue word
        ("construction since 2023", True, ("construction",), {}, ("2023-01-01", None), None, None, False, ""),
        (
            "mining between 2021 and 2024",
            True,
            ("clearance",),
            {},
            ("2021-01-01", "2024-12-31"),
            None,
            None,
            False,
            "",
        ),
        # change with no specific type = any type
        (
            "what changed near the river since 2020",
            True,
            (),
            {"water": NEAR},
            ("2020-01-01", None),
            None,
            None,
            False,
            "",
        ),
        (
            "new construction on SAR before 2025",
            True,
            ("construction",),
            {},
            (None, "2024-12-31"),
            None,
            "sentinel1",
            False,
            "construction",
        ),
    ],
)
def test_parser_table(text, change, types, near, dates, aoi, sensor, warn, semantic_has):
    p = parse(text)
    assert p.change is change
    assert p.change_types == types
    assert p.near == near
    assert p.date_range == dates
    assert p.aoi == aoi
    assert p.sensor == sensor
    assert ("below_resolution" in p.warnings) is warn
    assert semantic_has in p.semantic
    assert p.semantic.strip()  # CLIP always gets something to encode


def test_parser_spatial_target_is_not_a_change_type():
    p = parse("new buildings near a road")
    assert p.change_types == ("construction",)  # "road" is the spatial target, not a road-change
    assert p.near == {"road": NEAR}


def test_spatial_fit_full_then_linear_decay():
    assert spatial_fit(0, 1000) == 1.0
    assert spatial_fit(1000, 1000) == 1.0
    assert spatial_fit(1500, 1000) == pytest.approx(0.5)
    assert spatial_fit(2500, 1000) == 0.0


W = Fusion(w_sim=0.5, w_conf=0.2, w_q=0.1, w_sp=0.2)


def _c(tid, sim, conf=0.0, q=0.9, water=None, x=0.0):
    return Candidate(
        tile_id=tid,
        aoi="a",
        centroid_m=(x, 0.0),
        sim=sim,
        event_conf=conf,
        q=q,
        dist_m={} if water is None else {"water": water},
    )


def test_fuse_is_deterministic_and_filters_far_candidates():
    # sims span 0.10..0.30 after the far tile is dropped, so t2's 0.28 normalises to 0.9 (a small gap)
    cands = [
        _c("t3", 0.30, water=500),
        _c("t1", 0.30, water=500),
        _c("t2", 0.28, conf=0.9, water=800),
        _c("t4", 0.10, water=500),
        _c("far", 0.40, water=5000),
    ]
    out = fuse(cands, {"water": 1000}, W)
    assert [s.cand.tile_id for s in out] == [
        s.cand.tile_id for s in fuse(list(reversed(cands)), {"water": 1000}, W)
    ]
    assert "far" not in [s.cand.tile_id for s in out]  # precision first: outside 2x NEAR is dropped (§2.11)
    ids = [s.cand.tile_id for s in out]
    assert ids.index("t1") < ids.index("t3")  # equal scores tie-break by tile_id
    assert out[0].cand.tile_id == "t2"  # event confidence outweighs a small similarity gap
    assert all(set(s.parts) == {"sim_norm", "event_conf", "q", "spatial_fit"} for s in out)


def test_fuse_without_spatial_constraint_keeps_everyone():
    out = fuse([_c("a", 0.3), _c("b", 0.2)], {}, W)
    assert [s.cand.tile_id for s in out] == ["a", "b"]


def test_dedup_collapses_overlapping_tiles_into_one_site():
    r = T.retrieval.DEDUP_RADIUS_M
    cands = [
        _c("t0", 0.9, x=0),
        _c("t1", 0.8, x=r),
        _c("t2", 0.7, x=2 * r + 1),
    ]  # t1 overlaps t0; t2 is separate
    sites = dedup(fuse(cands, {}, W), r)
    assert [s.head.cand.tile_id for s in sites] == ["t0", "t2"]
    assert [m.cand.tile_id for m in sites[0].members] == ["t0", "t1"]


def test_spatial_targets_without_context_are_dropped_with_a_warning():
    from query.retrieve import _available_near

    class Conn:  # only 'road' has a finite distance in scope
        def execute(self, *_):
            return [("road",)]

    warnings: list[str] = []
    assert _available_near(Conn(), {"water": 1000.0, "road": 500.0}, "x", warnings) == {"road": 500.0}
    assert warnings == ["spatial_context_unavailable:water"]
    assert _available_near(Conn(), {}, "x", warnings) == {}


def test_image_search_excludes_tiles_sharing_ground_with_the_seed():
    from query.retrieve import overlapping

    ex = overlapping("navimumbai_43N_0038_0020")
    assert len(ex) == 9 and "navimumbai_43N_0039_0021" in ex and "navimumbai_43N_0040_0020" not in ex


def test_before_after_images_bracket_the_change_and_prefer_dry_seasons():
    from api.main import _pick_images

    c = lambda s, a, b: {"season": s, "start_date": a, "end_date": b}
    comps = [
        c("2021-MAM", "2021-03-01", "2021-05-31"),
        c("2021-JJAS", "2021-06-01", "2021-09-30"),
        c("2022-JJAS", "2022-06-01", "2022-09-30"),
        c("2022-ON", "2022-10-01", "2022-11-30"),
        c("2023-MAM", "2023-03-01", "2023-05-31"),
    ]
    got = _pick_images(comps, "2021-10-05", "2022-06-10")
    assert got["before"]["season"] == "2021-MAM"  # dry beats the later monsoon composite
    assert got["after"]["season"] == "2023-MAM"  # same season of year as before, after the change
