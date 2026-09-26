"""Unique-site dedup: 640 m tiles on a 320 m stride overlap, so one place shows up as up to 4 tiles."""

import math
from dataclasses import dataclass

from query.fuse import Scored


@dataclass(frozen=True)
class Site:
    head: Scored  # best-scoring tile of the site
    members: list[Scored]  # head first, then absorbed neighbours in score order


def dedup(ranked: list[Scored], radius_m: float) -> list[Site]:
    """Greedy, in score order: a tile within radius_m of a kept site's head (same AOI) joins that site."""
    sites: list[Site] = []
    for s in ranked:
        for site in sites:
            h = site.head.cand
            if h.aoi == s.cand.aoi and math.dist(h.centroid_m, s.cand.centroid_m) <= radius_m:
                site.members.append(s)
                break
        else:
            sites.append(Site(s, [s]))
    return sites
