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
    BASELINE_MAX_MONTHS: int  # sparse series: baseline grows to MIN_BASELINE_OBS dates, at most this long
    TREND_MIN_MONTHS: int  # fit the c*t trend only when the baseline spans this long (M3 decision)
    MIN_BASELINE_OBS: int  # fewer valid baseline obs -> pixel not modelled
    SIGMA_MIN: float  # floor on the robust sigma (index units): near-constant pixels would break on noise


class Classify(_Strict):
    CONS_DNDVI: float
    CONS_DNDBI: float
    CLR_DNDVI: float
    CLR_DBSI: float
    WATER_DMNDWI: float
    WATER_FRAC: float
    ROAD_ELONG: float
    ROAD_WIDTH_PX: int
    ROAD_DNDVI: float  # elongated NDBI rise = road only with at least this NDVI drop
    DARK_DMNDWI: float  # MNDWI rise + vegetation loss, no water extent change = dark surface (mine)


class Events(_Strict):
    OPERATING_CONF: float
    TARGET_PRECISION: float
    MIN_EVENT_PX: int  # connected break pixels needed for an event (25 px = 0.25 ha at 10 m)
    MIN_POST_DAYS: (
        int  # break -> end of series span for full confidence (shorter = end-of-series, provisional)
    )
    HIDDEN_TYPES: list[str]  # stored, but not reported unless asked (type never reaches TARGET_PRECISION)


class Retrieval(_Strict):
    TOP_N: int
    TOP_K_SITES: int
    DEDUP_RADIUS_M: float


class Cluster(_Strict):
    MIN_CLUSTER_SIZE: int
    UMAP_DIM: int


class Context(_Strict):
    NEAR_M: float  # "near X" = tile within this distance of X (full spatial_fit), linear decay to 0 at 2x
    WATER_OCCURRENCE_MIN: int  # JRC GSW occurrence % counted as water (0 = never water, not nodata)
    WATER_MIN_PX: int  # water components smaller than this (JRC 30 m pixels) are speckle
    ROAD_CLASSES: list[str]  # OSM highway=* values that count as a road


class Fusion(_Strict):  # DESIGN.md §7 fusion score
    w_sim: float
    w_conf: float
    w_q: float
    w_sp: float
    w_lc: float  # independent land-cover share of the queried area type (0 when the query names none)


class Landsat(_Strict):
    WINDOW_DAYS: (
        int  # year before last_normal_date vs year after earliest_supported_date (seasons average out)
    )
    MIN_OBS: int  # clear Landsat dates needed on each side
    MIN_VALID_FRAC: float  # share of the footprint's pixels valid for a date to count


class Sar(_Strict):
    MIN_DB: float  # |log-ratio| for construction (VV up) / clearance (VH down) agreement
    WATER_MIN_DB: float  # |VV log-ratio| for water appearance/disappearance agreement
    N_BEFORE: int  # S1 acquisitions before last_normal_date used for the "before" median
    N_AFTER: int  # S1 acquisitions from earliest_supported_date on for the "after" median


class Lulc(_Strict):
    MIN_FRAC: float  # footprint share a land-cover group must gain/lose for a transition (else "stable")


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
    context: Context
    sar: Sar
    landsat: Landsat
    lulc: Lulc


class AOI(_Strict):
    name: str
    type: str
    why: str
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    optical_source: str = (
        "sentinel2"  # the one optical source this AOI's time series uses (M3b; fusion is M9)
    )


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
fusion = Fusion(**_yaml(CONFIGS / "fusion.yaml")["weights"])
lexicon: dict = _yaml(CONFIGS / "query.yaml")
eval_cfg: dict = _yaml(CONFIGS / "eval.yaml")
aois: dict[str, AOI] = {p.stem: AOI(**_yaml(p)) for p in sorted((CONFIGS / "aoi").glob("*.yaml"))}


def stac_url(source: str) -> str:
    return sources["stac"][sources["sources"][source]["stac"]]
