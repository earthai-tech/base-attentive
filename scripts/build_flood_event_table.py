"""Build the harmonized regional flood-event table for the flood paper.

The table joins Global Flood Database event metadata to our eight study regions
and extends recent years with EM-DAT flood/disaster inventory rows where the
satellite event catalogue is no longer available. Optional ERA5 event-window
summaries are included when a NetCDF backend is installed for xarray.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATA_ROOT = Path(r"F:\_DATA\FloodData")
DEFAULT_TARGET_YEARS = [2010, 2013, 2015, 2018, 2020, 2021, 2022, 2023, 2024]
RECENT_INVENTORY_YEARS = [2020, 2021, 2022, 2023, 2024]


EMDAT_REGION_MATCHERS = {
    "west_africa_niger_benue": {
        "iso": {"BEN", "BFA", "CMR", "MLI", "NER", "NGA", "TCD"},
        "terms": [
            "niger",
            "benue",
            "niamey",
            "maradi",
            "tahoua",
            "tillaberi",
            "dosso",
            "mopti",
            "segou",
            "bamako",
            "benin",
            "nigeria",
            "cameroon",
            "chad",
            "adamawa",
            "taraba",
            "kogi",
            "kebbi",
            "benue state",
            "borno",
        ],
    },
    "southern_africa_limpopo_zambezi": {
        "iso": {"AGO", "BWA", "MWI", "MOZ", "NAM", "ZAF", "ZMB", "ZWE"},
        "terms": [
            "limpopo",
            "zambezi",
            "zambia",
            "zimbabwe",
            "mozambique",
            "malawi",
            "botswana",
            "south africa",
            "kwazulu",
            "gauteng",
            "mpumalanga",
            "sofala",
            "tete",
        ],
    },
    "east_africa_nile_headwaters": {
        "iso": {"BDI", "ETH", "KEN", "RWA", "SDN", "SSD", "TZA", "UGA"},
        "terms": [
            "nile",
            "white nile",
            "blue nile",
            "sudan",
            "south sudan",
            "ethiopia",
            "uganda",
            "kenya",
            "tanzania",
            "gambella",
            "amhara",
            "afar",
            "oromia",
            "lake victoria",
            "bara",
            "baro",
        ],
    },
    "ganges_brahmaputra_meghna": {
        "iso": {"BGD", "BTN", "IND", "NPL"},
        "terms": [
            "ganges",
            "brahmaputra",
            "meghna",
            "bangladesh",
            "nepal",
            "bhutan",
            "assam",
            "bihar",
            "uttar pradesh",
            "west bengal",
            "sylhet",
            "jamalpur",
            "kurigram",
            "gaibandha",
            "sunamganj",
            "sherpur",
            "sirajganj",
        ],
    },
    "indus": {
        "iso": {"AFG", "IND", "PAK"},
        "terms": [
            "indus",
            "pakistan",
            "punjab",
            "sindh",
            "balochistan",
            "khyber",
            "gilgit",
            "jammu",
            "kashmir",
            "himachal",
            "uttarakhand",
            "sutlej",
        ],
    },
    "rhine_meuse": {
        "iso": {"BEL", "CHE", "DEU", "FRA", "LUX", "NLD"},
        "terms": [
            "rhine",
            "rhein",
            "meuse",
            "maas",
            "ahr",
            "vesdre",
            "limburg",
            "liege",
            "namur",
            "valkenburg",
            "north rhine",
            "rheinland",
            "belgium",
            "netherlands",
        ],
    },
    "mississippi_missouri_texas_gulf": {
        "iso": {"USA"},
        "terms": [
            "mississippi",
            "missouri",
            "arkansas",
            "tennessee",
            "louisiana",
            "texas",
            "houston",
            "gulf",
            "iowa",
            "minnesota",
            "nebraska",
            "wisconsin",
            "oklahoma",
            "kansas",
            "alabama",
        ],
    },
    "mekong": {
        "iso": {"KHM", "LAO", "MMR", "THA", "VNM"},
        "terms": [
            "mekong",
            "cambodia",
            "lao",
            "laos",
            "thailand",
            "viet nam",
            "vietnam",
            "myanmar",
            "chiang mai",
            "chiang rai",
            "nan",
            "vientiane",
            "champasak",
            "savannakhet",
            "phnom penh",
            "prey veng",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target-years", nargs="*", type=int, default=DEFAULT_TARGET_YEARS)
    parser.add_argument("--include-era5", action="store_true")
    parser.add_argument("--event-window-days", type=int, default=7)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_gfd_date(value: str) -> dt.date:
    return dt.datetime.strptime(str(value), "%Y%m%d").date()


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text))


def emdat_date(row: pd.Series, prefix: str) -> dt.date | None:
    year = optional_int(row.get(f"{prefix} Year"))
    if year is None:
        return None
    month = optional_int(row.get(f"{prefix} Month")) or (1 if prefix == "Start" else 12)
    if prefix == "Start":
        day = optional_int(row.get(f"{prefix} Day")) or 1
    else:
        day = optional_int(row.get(f"{prefix} Day")) or calendar.monthrange(year, month)[1]
    day = min(day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def month_range(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    year_months: list[tuple[int, int]] = []
    cursor = dt.date(start.year, start.month, 1)
    stop = dt.date(end.year, end.month, 1)
    while cursor <= stop:
        year_months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = dt.date(cursor.year + 1, 1, 1)
        else:
            cursor = dt.date(cursor.year, cursor.month + 1, 1)
    return year_months


def detect_era5_years(data_root: Path) -> list[int]:
    years: set[int] = set()
    for path in (data_root / "raw" / "reanalysis" / "era5").glob("*/*.nc"):
        parts = path.stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            years.add(int(parts[1]))
    return sorted(years)


def era5_file(data_root: Path, region: str, year: int, month: int) -> Path:
    mm = f"{month:02d}"
    return data_root / "raw" / "reanalysis" / "era5" / region / f"era5_hourly_{region}_{year}_{mm}.nc"


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def era5_cache_dir(data_root: Path) -> Path:
    path = data_root / "interim" / "era5_zip_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def point_in_bbox(latitude: object, longitude: object, bbox: list[float]) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat) or math.isnan(lon):
        return False
    north, west, south, east = bbox
    return south <= lat <= north and west <= lon <= east


def emdat_region_match(row: pd.Series, region: str, bbox: list[float]) -> str | None:
    if point_in_bbox(row.get("Latitude"), row.get("Longitude"), bbox):
        return "coordinate_bbox"

    matcher = EMDAT_REGION_MATCHERS[region]
    iso = clean_text(row.get("ISO")).upper()
    haystack = " ".join(
        clean_text(row.get(col)).lower()
        for col in ["Country", "Subregion", "Region", "Location", "River Basin", "Event Name"]
    )
    term_match = any(term in haystack for term in matcher["terms"])
    if iso in matcher["iso"] and term_match:
        return "iso_and_keyword"
    return None


def emdat_value(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_era5_member(data_root: Path, archive_path: Path, variable: str) -> Path | None:
    """Return a readable NetCDF path for one ERA5 variable.

    CDS currently returns a ZIP containing separate NetCDF files for
    accumulated and instantaneous variables. Older or provider-specific
    downloads may be direct NetCDF files, so direct files are also supported.
    """

    with archive_path.open("rb") as fh:
        signature = fh.read(4)
    if signature != b"PK\x03\x04":
        return archive_path

    preferred = {
        "tp": "data_stream-oper_stepType-accum.nc",
        "t2m": "data_stream-oper_stepType-instant.nc",
    }
    cache_root = era5_cache_dir(data_root) / archive_path.parent.name / archive_path.stem
    cache_root.mkdir(parents=True, exist_ok=True)
    target_name = preferred.get(variable)

    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if target_name not in names:
            candidates = [name for name in names if name.endswith(".nc")]
            if not candidates:
                return None
            target_name = candidates[0]
        out_path = cache_root / Path(target_name).name
        if not out_path.exists() or out_path.stat().st_size == 0:
            with zf.open(target_name) as src, out_path.open("wb") as dst:
                dst.write(src.read())
    return out_path


def count_noaa_stations(data_root: Path, region: str) -> int | None:
    station_dir = data_root / "raw" / "precipitation" / "noaa_stations"
    matches = sorted(station_dir.glob(f"noaa_ghcnd_prcp_stations_{region}_*.json"))
    if not matches:
        return None
    try:
        payload = read_json(matches[-1])
    except json.JSONDecodeError:
        return None
    return payload.get("station_count") or len(payload.get("results", []))


def latest_emdat_path(data_root: Path) -> Path | None:
    inventory_dir = data_root / "raw" / "disaster_inventory"
    matches = sorted(inventory_dir.glob("*.xlsx")) + sorted(inventory_dir.glob("*.csv"))
    return matches[-1] if matches else None


def load_recent_emdat_rows(data_root: Path) -> pd.DataFrame:
    path = latest_emdat_path(data_root)
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name="EM-DAT Data")
    else:
        df = pd.read_csv(path)
    return df[
        (df["Disaster Type"].astype(str).str.lower() == "flood")
        & (df["Start Year"].isin(RECENT_INVENTORY_YEARS))
    ].copy()


def build_emdat_rows(
    data_root: Path,
    regions: dict[str, Any],
    target_years: list[int],
    available_era5_years: list[int],
) -> list[dict[str, Any]]:
    emdat = load_recent_emdat_rows(data_root)
    if emdat.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, event in emdat.iterrows():
        start = emdat_date(event, "Start")
        end = emdat_date(event, "End") or start
        if start is None or end is None:
            continue
        if end < start:
            end = start
        event_years = list(range(start.year, end.year + 1))
        for region, region_meta in regions.items():
            bbox = region_meta["bbox"]
            match_method = emdat_region_match(event, region, bbox)
            if match_method is None:
                continue
            north, west, south, east = bbox
            era5_event_months = [
                (year, month)
                for year, month in month_range(start, end)
                if path_exists(era5_file(data_root, region, year, month))
            ]
            noaa_station_count = count_noaa_stations(data_root, region)
            disno = clean_text(event.get("DisNo."))
            event_name = clean_text(event.get("Event Name")) or f"EM-DAT {disno}"
            rows.append(
                {
                    "event_region_id": f"{region}__EMDAT_{disno}",
                    "dfo_id": disno,
                    "event_name": event_name,
                    "region": region,
                    "region_label": region_meta.get("label", region),
                    "bbox_north": north,
                    "bbox_west": west,
                    "bbox_south": south,
                    "bbox_east": east,
                    "mechanisms": "; ".join(region_meta.get("mechanisms", [])),
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "start_year": start.year,
                    "end_year": end.year,
                    "start_month": start.month,
                    "duration_days": (end - start).days + 1,
                    "event_midpoint": (start + dt.timedelta(days=((end - start).days // 2))).isoformat(),
                    "target_year_event": any(year in target_years for year in event_years),
                    "downloaded_era5_year_event": any(year in available_era5_years for year in event_years),
                    "era5_available_months": len(era5_event_months),
                    "era5_month_keys": ";".join(f"{year}-{month:02d}" for year, month in era5_event_months),
                    "noaa_prcp_station_count": noaa_station_count,
                    "event_inventory_source": "EM-DAT Public Table",
                    "event_inventory_match": match_method,
                    "gfd_source": "",
                    "emdat_disno": disno,
                    "emdat_iso": clean_text(event.get("ISO")),
                    "emdat_country": clean_text(event.get("Country")),
                    "emdat_location": clean_text(event.get("Location")),
                    "emdat_river_basin": clean_text(event.get("River Basin")),
                    "emdat_total_deaths": emdat_value(event, "Total Deaths"),
                    "emdat_total_affected": emdat_value(event, "Total Affected"),
                    "has_gfd_metadata": False,
                    "has_noaa_station_inventory": noaa_station_count is not None,
                    "has_hydrolakes": path_exists(
                        data_root / "raw" / "reservoirs" / "HydroLAKES_points_v10_shp.zip"
                    ),
                    "has_emdat_public_file": True,
                }
            )
    return rows


def build_base_table(data_root: Path, target_years: list[int]) -> pd.DataFrame:
    regions = read_json(data_root / "study_regions.json")
    gfd = read_json(
        data_root / "raw" / "satellite_flood_extent" / "global_flood_database" / "gfd_region_summary.json"
    )
    available_era5_years = detect_era5_years(data_root)

    rows: list[dict[str, Any]] = []
    for region_block in gfd["regions"]:
        region = region_block["region"]
        region_meta = regions.get(region, {})
        bbox = region_block.get("bbox") or region_meta.get("bbox")
        north, west, south, east = bbox
        noaa_station_count = count_noaa_stations(data_root, region)
        for event in region_block["events"]:
            start = parse_gfd_date(event["start_date"])
            end = parse_gfd_date(event["end_date"])
            event_years = list(range(start.year, end.year + 1))
            era5_event_months = [
                (year, month)
                for year, month in month_range(start, end)
                if path_exists(era5_file(data_root, region, year, month))
            ]
            rows.append(
                {
                    "event_region_id": f"{region}__DFO_{event['dfo_id']}",
                    "dfo_id": str(event["dfo_id"]),
                    "event_name": event["event_name"],
                    "region": region,
                    "region_label": region_block.get("region_label", region_meta.get("label", region)),
                    "bbox_north": north,
                    "bbox_west": west,
                    "bbox_south": south,
                    "bbox_east": east,
                    "mechanisms": "; ".join(region_meta.get("mechanisms", [])),
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "start_year": start.year,
                    "end_year": end.year,
                    "start_month": start.month,
                    "duration_days": (end - start).days + 1,
                    "event_midpoint": (start + dt.timedelta(days=((end - start).days // 2))).isoformat(),
                    "target_year_event": any(year in target_years for year in event_years),
                    "downloaded_era5_year_event": any(year in available_era5_years for year in event_years),
                    "era5_available_months": len(era5_event_months),
                    "era5_month_keys": ";".join(f"{year}-{month:02d}" for year, month in era5_event_months),
                    "noaa_prcp_station_count": noaa_station_count,
                    "event_inventory_source": "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1",
                    "event_inventory_match": "earth_engine_filter_bounds",
                    "gfd_source": "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1",
                    "emdat_disno": "",
                    "emdat_iso": "",
                    "emdat_country": "",
                    "emdat_location": "",
                    "emdat_river_basin": "",
                    "emdat_total_deaths": None,
                    "emdat_total_affected": None,
                    "has_gfd_metadata": True,
                    "has_noaa_station_inventory": noaa_station_count is not None,
                    "has_hydrolakes": path_exists(
                        data_root / "raw" / "reservoirs" / "HydroLAKES_points_v10_shp.zip"
                    ),
                    "has_emdat_public_file": any((data_root / "raw" / "disaster_inventory").glob("*.xlsx")),
                }
            )
    rows.extend(build_emdat_rows(data_root, regions, target_years, available_era5_years))
    return pd.DataFrame(rows).sort_values(["region", "start_date", "dfo_id"]).reset_index(drop=True)


def summarize_era5_window(data_root: Path, row: pd.Series, event_window_days: int) -> dict[str, float | int | str]:
    try:
        import xarray as xr
    except ImportError as exc:
        return {"era5_summary_status": f"xarray unavailable: {exc}"}

    start = dt.date.fromisoformat(row["start_date"]) - dt.timedelta(days=event_window_days)
    end = dt.date.fromisoformat(row["end_date"]) + dt.timedelta(days=event_window_days)
    paths = [era5_file(data_root, row["region"], year, month) for year, month in month_range(start, end)]
    paths = [path for path in paths if path_exists(path)]
    if not paths:
        return {"era5_summary_status": "no local ERA5 files for event window"}

    frames: list[Any] = []
    try:
        out: dict[str, float | int | str] = {"era5_summary_status": "ok", "era5_summary_file_count": len(paths)}
        for variable in ["tp", "t2m"]:
            variable_paths = [extract_era5_member(data_root, path, variable) for path in paths]
            variable_paths = [path for path in variable_paths if path is not None]
            if not variable_paths:
                continue
            variable_frames = [xr.open_dataset(path) for path in variable_paths]
            frames.extend(variable_frames)
            ds = xr.concat(variable_frames, dim="valid_time" if "valid_time" in variable_frames[0].coords else "time")
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            event_ds = ds.sel({time_name: slice(str(start), str(end))})
            if variable not in event_ds:
                continue
            spatial_dims = [dim for dim in event_ds[variable].dims if dim != time_name]
            regional_series = event_ds[variable].mean(dim=spatial_dims)
            if variable == "tp":
                precip_mm = regional_series * 1000.0
                out["era5_window_total_precip_mm"] = float(precip_mm.sum().values)
                out["era5_window_mean_hourly_precip_mm"] = float(precip_mm.mean().values)
                out["era5_window_max_hourly_precip_mm"] = float(precip_mm.max().values)
            elif variable == "t2m":
                temp_c = regional_series - 273.15
                out["era5_window_mean_t2m_c"] = float(temp_c.mean().values)
                out["era5_window_min_t2m_c"] = float(temp_c.min().values)
                out["era5_window_max_t2m_c"] = float(temp_c.max().values)
            ds.close()
        return out
    except Exception as exc:  # backend availability varies by machine
        return {"era5_summary_status": f"failed: {type(exc).__name__}: {exc}"}
    finally:
        for frame in frames:
            frame.close()


def add_optional_era5_summaries(df: pd.DataFrame, data_root: Path, event_window_days: int) -> pd.DataFrame:
    summaries = [summarize_era5_window(data_root, row, event_window_days) for _, row in df.iterrows()]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(summaries)], axis=1)


def write_outputs(df: pd.DataFrame, data_root: Path, target_years: list[int], include_era5: bool) -> None:
    interim = data_root / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    event_path = interim / "harmonized_flood_events.csv"
    df.to_csv(event_path, index=False)

    summary = (
        df.groupby(["region", "region_label"], as_index=False)
        .agg(
            gfd_event_count=("dfo_id", "count"),
            target_year_event_count=("target_year_event", "sum"),
            downloaded_era5_event_count=("downloaded_era5_year_event", "sum"),
            median_duration_days=("duration_days", "median"),
            noaa_prcp_station_count=("noaa_prcp_station_count", "max"),
        )
        .sort_values("gfd_event_count", ascending=False)
    )
    summary.to_csv(interim / "region_evidence_summary.csv", index=False)

    yearly = (
        df.assign(year=df["start_year"])
        .groupby(["region", "region_label", "year"], as_index=False)
        .agg(event_count=("dfo_id", "count"))
        .sort_values(["region", "year"])
    )
    yearly.to_csv(interim / "region_year_event_counts.csv", index=False)

    manifest = {
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "data_root": str(data_root),
        "target_years": target_years,
        "include_era5": include_era5,
        "rows": int(len(df)),
        "regions": int(df["region"].nunique()),
        "downloaded_era5_years": detect_era5_years(data_root),
        "outputs": {
            "events": str(event_path),
            "region_summary": str(interim / "region_evidence_summary.csv"),
            "yearly_counts": str(interim / "region_year_event_counts.csv"),
        },
    }
    (interim / "harmonized_flood_events.metadata.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    target_years = sorted(set(args.target_years))
    df = build_base_table(data_root, target_years)
    if args.include_era5:
        df = add_optional_era5_summaries(df, data_root, args.event_window_days)
    write_outputs(df, data_root, target_years, args.include_era5)
    print(f"Wrote {len(df)} event-region rows to {data_root / 'interim' / 'harmonized_flood_events.csv'}")


if __name__ == "__main__":
    main()
