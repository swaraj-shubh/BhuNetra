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
    if (built or bare or e.d_ndbi > 0) and e.elongation >= c.ROAD_ELONG and e.width_px <= c.ROAD_WIDTH_PX:
        return "road", "appearance"
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
