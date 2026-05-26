"""common.py – Shared paths, Africa region definitions, and utility helpers.

All PADR-Net scripts import from here.  Set DATA_ROOT to point at the
external drive if it differs from the default below.
"""

from __future__ import annotations

from pathlib import Path
import json
import datetime as dt

# ── Root paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"F:\_DATA\FloodData")

PAPER_DIR = Path(__file__).resolve().parents[1]   # flood-mathematical_geosciences/
SCRIPTS_DIR = PAPER_DIR / "scripts"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR  = PAPER_DIR / "tables"
RESULTS_DIR = PAPER_DIR / "results"
SUPP_DIR    = PAPER_DIR / "supplementary"

# Create output dirs on import so scripts never need to mkdir manually
for _d in (FIGURES_DIR, TABLES_DIR, RESULTS_DIR, SUPP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data locations ────────────────────────────────────────────────────────
RAW_DIR     = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"

EMDAT_FILE  = RAW_DIR / "disaster_inventory" / "public_emdat_2026-05-11.xlsx"
GFD_META    = RAW_DIR / "satellite_flood_extent" / "global_flood_database" / "gfd_event_metadata.json"
GFD_SUMMARY = RAW_DIR / "satellite_flood_extent" / "global_flood_database" / "gfd_region_summary.csv"
ERA5_DIR    = RAW_DIR / "reanalysis" / "era5"
TOPO_DIR    = RAW_DIR / "topography"
SOIL_DIR    = RAW_DIR / "soil_moisture"

HARMONISED_EVENTS = INTERIM_DIR / "harmonized_flood_events.csv"
STUDY_REGIONS_JSON = DATA_ROOT / "study_regions.json"

# ── Africa study regions ──────────────────────────────────────────────────────
# Only these three are used for the Mathematical Geosciences paper.
AFRICA_REGIONS = {
    "west_africa_niger_benue": {
        "label": "West Africa: Niger / Benue basin",
        "bbox": [4.0, -12.0, 15.0, 15.0],   # [lat_min, lon_min, lat_max, lon_max]
        "iso": {"BEN", "BFA", "CMR", "MLI", "NER", "NGA", "TCD"},
        "mechanisms": ["Sahel rainfall", "urban flood exposure", "data-sparse"],
    },
    "east_africa_nile_headwaters": {
        "label": "East Africa: Nile headwaters / Sudan-Ethiopia corridor",
        "bbox": [-4.0, 28.0, 16.0, 40.0],
        "iso": {"BDI", "ETH", "KEN", "RWA", "SDN", "SSD", "TZA", "UGA"},
        "mechanisms": ["seasonal rainfall", "large-basin routing", "high exposure"],
    },
    "southern_africa_limpopo_zambezi": {
        "label": "Southern Africa: Limpopo / Zambezi",
        "bbox": [-27.0, 20.0, -8.0, 37.0],
        "iso": {"AGO", "BWA", "MWI", "MOZ", "NAM", "ZAF", "ZMB", "ZWE"},
        "mechanisms": ["tropical cyclones", "reservoir influence", "drought-flood"],
    },
}

ALL_AFRICA_ISO: set[str] = set().union(
    *(r["iso"] for r in AFRICA_REGIONS.values())
)

# ── Validation setup ──────────────────────────────────────────────────────────
# Events with estimated return period < 50 yr → training split
# Events with return period ≥ 100 yr          → extreme test split
RETURN_PERIOD_TRAIN_MAX  = 50    # years
RETURN_PERIOD_TEST_MIN   = 100   # years

TRAIN_YEARS = list(range(2000, 2018))   # 2000-2017 for training
VAL_YEARS   = [2018, 2019]              # validation / hyperparameter tuning
TEST_YEARS  = list(range(2020, 2025))   # 2020-2024 held-out test (recent extremes)

# ── Evaluation metrics (names used consistently across scripts) ───────────────
METRIC_NAMES = ["NSE", "CSI", "RMSE", "MAE", "delta_mass_pct"]


def load_study_regions() -> dict:
    """Load the full study_regions.json from DATA_ROOT (8 global regions)."""
    with open(STUDY_REGIONS_JSON) as fh:
        return json.load(fh)


def africa_region_names() -> list[str]:
    """Return the three Africa region keys used in this paper."""
    return list(AFRICA_REGIONS.keys())


def timestamp() -> str:
    """ISO-format UTC timestamp for log messages."""
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def print_banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def print_rule(width: int = 60) -> None:
    print("-" * width)


if __name__ == "__main__":
    print_banner("common.py self-test")
    print(f"DATA_ROOT       : {DATA_ROOT}  (exists={DATA_ROOT.exists()})")
    print(f"EMDAT_FILE      : {EMDAT_FILE.name}  (exists={EMDAT_FILE.exists()})")
    print(f"GFD_META        : {GFD_META.name}  (exists={GFD_META.exists()})")
    print(f"HARMONISED_EVENTS: {HARMONISED_EVENTS.name}  (exists={HARMONISED_EVENTS.exists()})")
    print(f"\nAfrica regions  : {africa_region_names()}")
