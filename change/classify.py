"""Rule-based change type + direction (DESIGN.md §7 enums, thresholds in configs/thresholds.yaml -> classify).

Deltas are deseasonalised: median of (observed - harmonic expectation) after the break, in index units. M5 may
replace or feed these rules into a learned classifier; the output enums stay the same.
"""

from dataclasses import dataclass

from configs.settings import T as CFG


@dataclass(frozen=True)
class Evidence:
    d_ndvi: float
    d_mndwi: float
    d_ndbi: float
    d_bsi: float
    water_pre: float  # fraction of the event's pixels that were water (MNDWI > 0) before / after
    water_post: float
    elongation: float  # major / minor axis of the event footprint
    width_px: float  # mean width across the major axis


def classify(e: Evidence) -> tuple[str, str]:
    c = CFG.classify
    water_shift = abs(e.water_post - e.water_pre)
    if abs(e.d_mndwi) >= c.WATER_DMNDWI and water_shift >= c.WATER_FRAC:
        if e.d_mndwi > 0:
            return "water", "expansion" if e.water_pre >= c.WATER_FRAC else "appearance"
        return "water", "contraction" if e.water_post >= c.WATER_FRAC else "disappearance"

    built = e.d_ndbi >= c.CONS_DNDBI and e.d_ndvi <= c.CONS_DNDVI
    bare = e.d_bsi >= c.CLR_DBSI and e.d_ndvi <= c.CLR_DNDVI
    # M5b: a bare NDBI rise is not enough (tidal flats, revegetating mine benches): some vegetation loss must come
    # with it. Dholera roads over farmland/salt flats have dNDVI -0.09..-0.25, so full clearance is not required.
    linear = e.elongation >= c.ROAD_ELONG and e.width_px <= c.ROAD_WIDTH_PX
    if linear and (built or bare or (e.d_ndbi > 0 and e.d_ndvi <= c.ROAD_DNDVI)):
        return "road", "appearance"
    # M5b: vegetation -> dark surface (MNDWI up, no water-extent change): coal / mine face, not buildings.
    # Calibration pre-labels: hasdeo forest->mine had dMNDWI +0.20..0.28, real construction <= +0.18.
    if e.d_mndwi >= c.DARK_DMNDWI and e.d_ndvi <= c.CLR_DNDVI:
        return "clearance", "disappearance"
    if built and bare:  # both fire: the rule with the larger margin over its own threshold wins
        return (
            ("construction", "appearance")
            if e.d_ndbi - c.CONS_DNDBI >= e.d_bsi - c.CLR_DBSI
            else ("clearance", "disappearance")
        )
    if built:
        return "construction", "appearance"
    if bare:
        return "clearance", "disappearance"
    return "other", "expansion" if e.d_ndvi > 0 else "appearance"
