"""Typed settings: repo paths + configs/*.yaml (DESIGN.md §8). No magic numbers anywhere else.

from configs.settings import T, paths, sources, models, bands, aois
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(os.environ.get("SIH_ROOT", Path(__file__).resolve().parents[1]))
CONFIGS = ROOT / "configs"
PROCESSING_VERSION = "prep-0.2.0"  # bump when a stage's output changes for the same input


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quality(_Strict):
    Q_MIN: float
    SHIFT_ACCEPT_PX: float
    SHIFT_REJECT_PX: float
    VALID_FRAC_IDX: float
    VALID_FRAC_EMB: float
    CLOUD_PROB: float
    CLOUD_DILATE_PX: int
    SUN_ELEV_MIN: float  # q penalty below this sun elevation (degrees; DESIGN.md §7)


class Tiling(_Strict):
    TILE_M: float
    STRIDE_M: float


class Breaks(_Strict):
    R_MAX: float
    K_PERSIST: int
    BASELINE_MONTHS: int


class Classify(_Strict):
    CONS_DNDVI: float
    CONS_DNDBI: float
    CLR_DNDVI: float
    CLR_DBSI: float
    WATER_DMNDWI: float
    WATER_FRAC: float
    ROAD_ELONG: float
    ROAD_WIDTH_PX: int


class Events(_Strict):
    OPERATING_CONF: float
    TARGET_PRECISION: float


class Retrieval(_Strict):
    TOP_N: int
    TOP_K_SITES: int
    DEDUP_RADIUS_M: float


class Cluster(_Strict):
    MIN_CLUSTER_SIZE: int
    UMAP_DIM: int


class Composites(_Strict):
    SEASONS: dict[str, list[int]]  # season name -> calendar months


class Thresholds(_Strict):
    quality: Quality
    tiling: Tiling
    breaks: Breaks
    classify: Classify
    events: Events
    retrieval: Retrieval
    cluster: Cluster
    composites: Composites


class AOI(_Strict):
    name: str
    type: str
    why: str
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max


class Paths(_Strict):
    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    stac: Path = ROOT / "data" / "stac"
    tiles: Path = ROOT / "data" / "tiles"
    composites: Path = ROOT / "data" / "composites"
    benchmarks: Path = ROOT / "data" / "benchmarks"
    qdrant: Path = ROOT / "data" / "qdrant"
    logs: Path = ROOT / "data" / "manifests"  # staging log, jsonl lineage
    manifests: Path = ROOT / "manifests"  # datasets.yaml, models.yaml (in git)
    models: Path = ROOT / "models"
    web: Path = ROOT / "web"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


paths = Paths()
T = Thresholds(**_yaml(CONFIGS / "thresholds.yaml"))
sources: dict = _yaml(CONFIGS / "sources.yaml")
models: dict = _yaml(CONFIGS / "models.yaml")
bands: dict = _yaml(CONFIGS / "bands.yaml")
aois: dict[str, AOI] = {p.stem: AOI(**_yaml(p)) for p in sorted((CONFIGS / "aoi").glob("*.yaml"))}


def stac_url(source: str) -> str:
    return sources["stac"][sources["sources"][source]["stac"]]
