"""Download or stage data sources for the flood paper.

Some sources are direct HTTP downloads. Others require accounts, accepted
licences, or provider CLIs, so this script writes reproducible templates for
those instead of pretending they are one-click files.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FLOOD_DATA_ROOT", ROOT / "data" / "flood")).resolve()
RAW = DATA_ROOT / "raw"
LOGS = DATA_ROOT / "download_logs"
DEFAULT_ENV_FILE = ROOT / "data" / "flood" / ".env.local"


REGIONS = {
    "west_africa_niger_benue": {
        "label": "West Africa: Niger / Benue basin",
        "bbox": [15.0, -12.0, 4.0, 15.0],
        "mechanisms": ["Sahel rainfall", "urban flood exposure", "data-sparse African validation"],
    },
    "southern_africa_limpopo_zambezi": {
        "label": "Southern Africa: Limpopo / Zambezi",
        "bbox": [-8.0, 20.0, -27.0, 37.0],
        "mechanisms": ["tropical cyclones", "reservoir influence", "drought-flood transitions"],
    },
    "east_africa_nile_headwaters": {
        "label": "East Africa: Nile headwaters / Sudan-Ethiopia corridor",
        "bbox": [16.0, 28.0, -4.0, 40.0],
        "mechanisms": ["seasonal rainfall", "large-basin routing", "high exposure"],
    },
    "ganges_brahmaputra_meghna": {
        "label": "South Asia: Ganges-Brahmaputra-Meghna",
        "bbox": [32.0, 72.0, 20.0, 93.0],
        "mechanisms": ["monsoon floods", "floodplain exposure", "large transboundary basin"],
    },
    "indus": {
        "label": "Pakistan / India: Indus basin",
        "bbox": [37.0, 66.0, 23.0, 82.0],
        "mechanisms": ["snowmelt", "monsoon", "recent catastrophic floods"],
    },
    "rhine_meuse": {
        "label": "Europe: Rhine / Meuse",
        "bbox": [53.5, 2.0, 45.0, 12.0],
        "mechanisms": ["dense observations", "2021 benchmark floods", "temperate extremes"],
    },
    "mississippi_missouri_texas_gulf": {
        "label": "North America: Mississippi/Missouri and Texas Gulf",
        "bbox": [49.5, -106.0, 25.0, -88.0],
        "mechanisms": ["strong gauge coverage", "compound rainfall", "urban/coastal interactions"],
    },
    "mekong": {
        "label": "Southeast Asia: Mekong",
        "bbox": [34.0, 93.0, 8.0, 109.0],
        "mechanisms": ["monsoon", "reservoir operations", "floodplain agriculture"],
    },
}


SOURCES = {
    "usgs_streamflow": {
        "category": "observed river discharge",
        "mode": "api",
        "requires": "internet only",
        "description": "USGS NWIS daily streamflow values for selected gauge sites.",
        "source_url": "https://api.waterdata.usgs.gov/",
        "default_size": "small to medium, depending on sites and years",
    },
    "noaa_daily_precip": {
        "category": "rainfall / meteorology",
        "mode": "api",
        "requires": "NOAA_CDO_TOKEN environment variable",
        "description": "NOAA Climate Data Online daily precipitation.",
        "source_url": "https://www.ncei.noaa.gov/cdo-web/webservices/v2",
        "default_size": "small to medium, depending on stations and years",
    },
    "noaa_stations_region": {
        "category": "rainfall / meteorology",
        "mode": "api",
        "requires": "NOAA_CDO_TOKEN environment variable",
        "description": "NOAA GHCND station inventory for one selected region.",
        "source_url": "https://www.ncei.noaa.gov/cdo-web/webservices/v2",
        "default_size": "small JSON metadata",
    },
    "hydrolakes_points": {
        "category": "reservoirs / lakes",
        "mode": "direct",
        "requires": "internet only",
        "description": "HydroLAKES lake pour points shapefile.",
        "source_url": "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_points_v10_shp.zip",
        "default_size": "79 MB",
    },
    "hydrolakes_polys": {
        "category": "reservoirs / lakes",
        "mode": "direct_large",
        "requires": "--allow-large",
        "description": "HydroLAKES lake polygons shapefile.",
        "source_url": "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip",
        "default_size": "820 MB",
    },
    "era5_template": {
        "category": "rainfall / meteorological reanalysis",
        "mode": "template",
        "requires": "Copernicus CDS account, accepted licence, cdsapi installed",
        "description": "ERA5 precipitation and temperature request template.",
        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        "default_size": "depends on area/time subset",
    },
    "era5_hourly_region": {
        "category": "rainfall / meteorological reanalysis",
        "mode": "api",
        "requires": "Copernicus CDS account, accepted licence, cdsapi installed",
        "description": "ERA5 hourly single-level variables for one selected region and month.",
        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        "default_size": "tens to hundreds of MB per region-month",
    },
    "global_flood_database_template": {
        "category": "satellite flood extent",
        "mode": "template",
        "requires": "Google Earth Engine account and initialized CLI",
        "description": "Global Flood Database v1 export guidance.",
        "source_url": "https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1",
        "default_size": "depends on exported events and region",
    },
    "gfd_event_metadata": {
        "category": "satellite flood extent",
        "mode": "api",
        "requires": "Earth Engine account and registered project",
        "description": "Global Flood Database event metadata and selected-region intersections.",
        "source_url": "https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1",
        "default_size": "small JSON metadata",
    },
    "gfd_region_summary": {
        "category": "satellite flood extent",
        "mode": "api",
        "requires": "Earth Engine account and registered project",
        "description": "Global Flood Database event counts and event IDs intersecting each selected study region.",
        "source_url": "https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1",
        "default_size": "small JSON and CSV metadata",
    },
    "worldcover_template": {
        "category": "land cover",
        "mode": "template",
        "requires": "AWS CLI for bulk sync, no AWS credentials required",
        "description": "ESA WorldCover 2021 v200 land-cover map download commands.",
        "source_url": "https://esa-worldcover.org/en/data-access",
        "default_size": "about 117 GB for the global map layer",
    },
    "smap_template": {
        "category": "soil moisture",
        "mode": "template",
        "requires": "NASA Earthdata Login",
        "description": "SMAP soil-moisture access notes.",
        "source_url": "https://www.earthdata.nasa.gov/data/instruments/smap-l-band-radiometer/near-real-time-data",
        "default_size": "depends on product and date range",
    },
    "copernicus_dem_template": {
        "category": "topography",
        "mode": "template",
        "requires": "AWS CLI, no AWS credentials required for public bucket",
        "description": "Copernicus DEM open-data access notes.",
        "source_url": "https://registry.opendata.aws/copernicus-dem/",
        "default_size": "large; tile by area of interest",
    },
    "ghsl_template": {
        "category": "urbanization / exposure",
        "mode": "template",
        "requires": "internet only",
        "description": "JRC Global Human Settlement Layer access notes.",
        "source_url": "https://data.jrc.ec.europa.eu/collection/ghsl/",
        "default_size": "depends on product and epoch",
    },
    "grand_template": {
        "category": "reservoirs / dams",
        "mode": "template",
        "requires": "manual download / licence review",
        "description": "GRanD global reservoir and dam database access notes.",
        "source_url": "https://water-future.org/gwsp-archive-grand-database/",
        "default_size": "small to medium vector data",
    },
    "emdat_template": {
        "category": "historical disaster inventory",
        "mode": "template",
        "requires": "EM-DAT registration for non-commercial access",
        "description": "EM-DAT public flood/disaster table access notes.",
        "source_url": "https://doc.emdat.be/docs/data-accessibility/",
        "default_size": "small table",
    },
    "nex_gddp_cmip6_template": {
        "category": "climate scenarios",
        "mode": "template",
        "requires": "internet only for THREDDS subset, AWS CLI for S3",
        "description": "NASA NEX-GDDP-CMIP6 precipitation scenario subset template.",
        "source_url": "https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6",
        "default_size": "depends on model/scenario/year/area",
    },
    "region_manifest": {
        "category": "study design",
        "mode": "template",
        "requires": "none",
        "description": "Selected representative regions and bounding boxes for subset downloads.",
        "source_url": "local study design",
        "default_size": "small JSON",
    },
}


def ensure_dirs() -> None:
    for path in [RAW, LOGS]:
        path.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "streamflow",
        "precipitation",
        "reanalysis",
        "satellite_flood_extent",
        "land_cover",
        "soil_moisture",
        "topography",
        "reservoirs",
        "urbanization",
        "disaster_inventory",
        "climate_scenarios",
        "manual",
    ]:
        (RAW / subdir).mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def configure_data_root(path: str | None) -> None:
    global DATA_ROOT, RAW, LOGS
    if not path:
        return
    DATA_ROOT = Path(path).expanduser().resolve()
    RAW = DATA_ROOT / "raw"
    LOGS = DATA_ROOT / "download_logs"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_metadata(name: str, output: Path, extra: dict[str, object] | None = None) -> None:
    meta = {
        "dataset": name,
        "source": SOURCES[name],
        "access_date_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "output": display_path(output),
    }
    if extra:
        meta.update(extra)
    meta_path = LOGS / f"{name}_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def download_url(url: str, output: Path, dry_run: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"DRY RUN: would download {url}")
        print(f"         to {output}")
        return
    print(f"Downloading {url}")
    print(f" -> {output}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "base-attentive-flood-paper-data-downloader/0.1",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request) as response, output.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_int:
                pct = downloaded / total_int * 100
                print(f"\r   {downloaded / 1e6:,.1f} MB / {total_int / 1e6:,.1f} MB ({pct:4.1f}%)", end="")
        print()


def fetch_usgs(args: argparse.Namespace) -> None:
    if not args.usgs_sites:
        raise SystemExit("--usgs-sites is required for usgs_streamflow, e.g. --usgs-sites 01646500,01491000")
    params = {
        "format": "json",
        "sites": args.usgs_sites,
        "parameterCd": args.usgs_parameter,
        "startDT": args.start,
        "endDT": args.end,
        "siteStatus": "all",
    }
    url = "https://waterservices.usgs.gov/nwis/dv/?" + urllib.parse.urlencode(params)
    stem = f"usgs_daily_{args.usgs_parameter}_{args.start}_{args.end}_{args.usgs_sites.replace(',', '-')}.json"
    output = RAW / "streamflow" / stem
    download_url(url, output, args.dry_run)
    if not args.dry_run:
        write_metadata("usgs_streamflow", output, {"sites": args.usgs_sites, "parameter": args.usgs_parameter})


def fetch_noaa(args: argparse.Namespace) -> None:
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        raise SystemExit("Set NOAA_CDO_TOKEN before downloading noaa_daily_precip.")
    if not args.noaa_station:
        raise SystemExit("--noaa-station is required, e.g. --noaa-station GHCND:USW00014732")
    params = {
        "datasetid": "GHCND",
        "stationid": args.noaa_station,
        "datatypeid": "PRCP",
        "startdate": args.start,
        "enddate": args.end,
        "units": "metric",
        "limit": 1000,
    }
    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data?" + urllib.parse.urlencode(params)
    output = RAW / "precipitation" / f"noaa_cdo_prcp_{args.noaa_station.replace(':', '_')}_{args.start}_{args.end}.json"
    if args.dry_run:
        print(f"DRY RUN: would request {url}")
        print(f"         to {output}")
        return
    request = urllib.request.Request(url, headers={"token": token})
    try:
        with urllib.request.urlopen(request) as response:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"NOAA CDO request failed: HTTP {exc.code} {exc.reason}") from exc
    write_metadata("noaa_daily_precip", output, {"station": args.noaa_station})


def noaa_request(url: str, token: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "token": token,
            "User-Agent": "base-attentive-flood-paper/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"NOAA CDO request failed: HTTP {exc.code} {exc.reason}") from exc


def fetch_noaa_stations_region(args: argparse.Namespace) -> None:
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        raise SystemExit("Set NOAA_CDO_TOKEN before downloading noaa_stations_region.")
    if not args.region:
        raise SystemExit("--region is required. Use one of: " + ", ".join(REGIONS))
    if args.region not in REGIONS:
        raise SystemExit(f"Unknown region {args.region!r}. Use one of: {', '.join(REGIONS)}")
    north, west, south, east = REGIONS[args.region]["bbox"]
    output = RAW / "precipitation" / "noaa_stations" / f"noaa_ghcnd_prcp_stations_{args.region}_{args.start}_{args.end}.json"
    if args.dry_run:
        print(f"DRY RUN: would fetch NOAA GHCND PRCP stations for {args.region} to {display_path(output)}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []
    offset = 1
    limit = 1000
    max_records = args.max_records or sys.maxsize
    while True:
        params = {
            "datasetid": "GHCND",
            "datatypeid": "PRCP",
            "startdate": args.start,
            "enddate": args.end,
            "extent": f"{south},{west},{north},{east}",
            "limit": limit,
            "offset": offset,
        }
        url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations?" + urllib.parse.urlencode(params)
        payload = noaa_request(url, token)
        results = payload.get("results", [])
        if isinstance(results, list):
            remaining = max_records - len(all_results)
            all_results.extend(results[:remaining])
            if len(all_results) >= max_records:
                break
        metadata = payload.get("metadata", {})
        resultset = metadata.get("resultset", {}) if isinstance(metadata, dict) else {}
        count = int(resultset.get("count", len(all_results))) if isinstance(resultset, dict) else len(all_results)
        if offset + limit > count:
            break
        offset += limit

    output.write_text(
        json.dumps(
            {
                "region": args.region,
                "region_label": REGIONS[args.region]["label"],
                "bbox": REGIONS[args.region]["bbox"],
                "start": args.start,
                "end": args.end,
                "station_count": len(all_results),
                "results": all_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_metadata(
        "noaa_stations_region",
        output,
        {
            "region": args.region,
            "station_count": len(all_results),
            "start": args.start,
            "end": args.end,
        },
    )
    print(f"NOAA stations for {args.region}: {len(all_results)} -> {display_path(output)}")


def fetch_era5_hourly_region(args: argparse.Namespace) -> None:
    if not args.region:
        raise SystemExit("--region is required. Use one of: " + ", ".join(REGIONS))
    if args.region not in REGIONS:
        raise SystemExit(f"Unknown region {args.region!r}. Use one of: {', '.join(REGIONS)}")
    if not os.environ.get("CDSAPI_URL") or not os.environ.get("CDSAPI_KEY"):
        raise SystemExit("CDSAPI_URL and CDSAPI_KEY must be set for ERA5 downloads.")

    try:
        import cdsapi  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("Install cdsapi first: python -m pip install cdsapi") from exc

    variables = [item.strip() for item in args.era5_variables.split(",") if item.strip()]
    days_in_month = calendar.monthrange(args.year, args.month)[1]
    year = f"{args.year:04d}"
    month = f"{args.month:02d}"
    region = REGIONS[args.region]
    output = RAW / "reanalysis" / "era5" / args.region / f"era5_hourly_{args.region}_{year}_{month}.nc"
    output.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [year],
        "month": [month],
        "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": region["bbox"],
    }
    if args.dry_run:
        print(json.dumps({"dataset": "reanalysis-era5-single-levels", "request": request, "output": display_path(output)}, indent=2))
        return

    client = cdsapi.Client(url=os.environ["CDSAPI_URL"], key=os.environ["CDSAPI_KEY"])
    client.retrieve("reanalysis-era5-single-levels", request, str(output))
    write_metadata(
        "era5_hourly_region",
        output,
        {
            "region": args.region,
            "region_label": region["label"],
            "year": args.year,
            "month": args.month,
            "variables": variables,
            "bbox": region["bbox"],
        },
    )


def fetch_gfd_event_metadata(args: argparse.Namespace) -> None:
    try:
        import ee  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("Install earthengine-api first: python -m pip install earthengine-api") from exc

    project = args.ee_project or os.environ.get("EARTHENGINE_PROJECT") or "fair-future-496413-f5"
    ee.Initialize(project=project)

    collection_id = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
    collection = ee.ImageCollection(collection_id)
    event_count = int(collection.size().getInfo())
    asset_ids = [
        item["id"]
        for item in ee.data.listAssets({"parent": f"projects/earthengine-public/assets/{collection_id}"})["assets"]
    ]
    records: list[dict[str, object]] = []
    for asset_id in asset_ids:
        name = asset_id.rsplit("/", 1)[-1]
        parts = name.split("_")
        records.append(
            {
                "asset_id": asset_id,
                "event_name": name,
                "dfo_id": parts[1] if len(parts) > 1 else None,
                "start_date": parts[3] if len(parts) > 3 else None,
                "end_date": parts[5] if len(parts) > 5 else None,
            }
        )

    output = RAW / "satellite_flood_extent" / "global_flood_database" / "gfd_event_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": collection_id,
                "project": project,
                "event_count": event_count,
                "events": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_metadata("gfd_event_metadata", output, {"event_count": event_count, "ee_project": project})
    print(f"GFD metadata: {event_count} events -> {display_path(output)}")


def parse_gfd_name(name: str) -> dict[str, str | None]:
    parts = name.split("_")
    return {
        "event_name": name,
        "dfo_id": parts[1] if len(parts) > 1 else None,
        "start_date": parts[3] if len(parts) > 3 else None,
        "end_date": parts[5] if len(parts) > 5 else None,
    }


def fetch_gfd_region_summary(args: argparse.Namespace) -> None:
    try:
        import ee  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("Install earthengine-api first: python -m pip install earthengine-api") from exc

    project = args.ee_project or os.environ.get("EARTHENGINE_PROJECT") or "fair-future-496413-f5"
    ee.Initialize(project=project)

    collection_id = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
    collection = ee.ImageCollection(collection_id)
    out_dir = RAW / "satellite_flood_extent" / "global_flood_database"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_output = out_dir / "gfd_region_summary.json"
    csv_output = out_dir / "gfd_region_summary.csv"

    summaries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for region_key, region in REGIONS.items():
        north, west, south, east = region["bbox"]
        geom = ee.Geometry.Rectangle([west, south, east, north], None, False)
        filtered = collection.filterBounds(geom)
        event_names = filtered.aggregate_array("system:index").getInfo()
        event_names = sorted(str(name) for name in event_names)
        parsed_events = [parse_gfd_name(name) for name in event_names]
        summaries.append(
            {
                "region": region_key,
                "region_label": region["label"],
                "bbox": region["bbox"],
                "event_count": len(parsed_events),
                "events": parsed_events,
            }
        )
        for event in parsed_events:
            rows.append(
                {
                    "region": region_key,
                    "region_label": region["label"],
                    "dfo_id": event["dfo_id"],
                    "event_name": event["event_name"],
                    "start_date": event["start_date"],
                    "end_date": event["end_date"],
                }
            )
        print(f"GFD region {region_key}: {len(parsed_events)} events")

    json_output.write_text(
        json.dumps(
            {
                "dataset": collection_id,
                "project": project,
                "regions": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["region", "region_label", "dfo_id", "event_name", "start_date", "end_date"],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_metadata(
        "gfd_region_summary",
        json_output,
        {
            "region_count": len(summaries),
            "row_count": len(rows),
            "csv_output": display_path(csv_output),
            "ee_project": project,
        },
    )
    print(f"GFD region summary -> {display_path(json_output)}")
    print(f"GFD region table -> {display_path(csv_output)}")


def write_templates() -> None:
    manual = RAW / "manual"
    manual.mkdir(parents=True, exist_ok=True)
    output = manual / "download_templates.md"
    output.write_text(
        textwrap.dedent(
            f"""\
            # Flood Data Download Templates

            Generated: {dt.datetime.utcnow().replace(microsecond=0).isoformat()}Z

            ## ERA5 / ERA5-Land

            Requires a Copernicus CDS account, accepted dataset licence, and `cdsapi`.

            ```python
            import cdsapi

            c = cdsapi.Client()
            c.retrieve(
                "reanalysis-era5-single-levels",
                {{
                    "product_type": ["reanalysis"],
                    "variable": ["total_precipitation", "2m_temperature"],
                    "year": ["2020"],
                    "month": ["01"],
                    "day": [f"{{day:02d}}" for day in range(1, 32)],
                    "time": [f"{{hour:02d}}:00" for hour in range(24)],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "area": [50, -130, 20, -60],
                }},
                "data/flood/raw/reanalysis/era5_precip_t2m_2020_01.nc",
            )
            ```

            ## Global Flood Database v1

            Requires Google Earth Engine.

            ```javascript
            var gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1");
            var events = gfd.filterDate("2000-01-01", "2018-12-31");
            Export.table.toDrive({{
              collection: events.map(function(img) {{ return ee.Feature(null, img.toDictionary()); }}),
              description: "global_flood_database_event_metadata",
              fileFormat: "CSV"
            }});
            ```

            ## ESA WorldCover 2021

            Global bulk download is about 117 GB. Prefer a regional subset first.

            ```powershell
            aws s3 sync s3://esa-worldcover/v200/2021/map data/flood/raw/land_cover/worldcover_2021_map --no-sign-request
            ```

            ## Copernicus DEM

            Use the AWS Open Data bucket and download only AOI tiles.

            ```powershell
            aws s3 ls s3://copernicus-dem-30m/ --no-sign-request
            ```

            ## NASA SMAP Soil Moisture

            Requires NASA Earthdata Login. Save downloads under:

            ```text
            data/flood/raw/soil_moisture/smap/
            ```

            ## EM-DAT

            Requires registration for non-commercial access. Save the public table as:

            ```text
            data/flood/raw/disaster_inventory/emdat_public.csv
            ```

            ## GRanD Reservoir / Dam Database

            Download after licence review and save under:

            ```text
            data/flood/raw/reservoirs/grand/
            ```

            ## NASA NEX-GDDP-CMIP6

            Example THREDDS NetCDF subset URL pattern for precipitation. Adjust model, scenario, year, and bounding box.

            ```text
            https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/pr/pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014.nc?var=pr&north=39&west=-77&east=-76&south=38.7&horizStride=1&time_start=2014-01-01T12:00:00Z&time_end=2014-12-31T12:00:00Z&accept=netcdf3&addLatLon=true
            ```
            """
        ),
        encoding="utf-8",
    )
    write_metadata("era5_template", output)
    print(f"Wrote {display_path(output)}")


def write_region_manifest() -> None:
    output = DATA_ROOT / "study_regions.json"
    output.write_text(json.dumps(REGIONS, indent=2) + "\n", encoding="utf-8")
    write_metadata("region_manifest", output, {"region_count": len(REGIONS)})
    print(f"Wrote {display_path(output)}")


def check_credentials() -> int:
    checks = {
        "FLOOD_DATA_ROOT": str(DATA_ROOT),
        "NOAA_CDO_TOKEN": "set" if os.environ.get("NOAA_CDO_TOKEN") else "missing",
        "CDSAPI_URL": "set" if os.environ.get("CDSAPI_URL") else "missing",
        "CDSAPI_KEY": "set" if os.environ.get("CDSAPI_KEY") else "missing",
        "EARTHDATA_USERNAME": "set" if os.environ.get("EARTHDATA_USERNAME") else "missing",
        "EARTHDATA_PASSWORD": "set" if os.environ.get("EARTHDATA_PASSWORD") else "missing",
    }
    for key, value in checks.items():
        print(f"{key}={value}")
    return 0


def list_sources() -> None:
    for name, info in SOURCES.items():
        print(f"{name}")
        print(f"  category: {info['category']}")
        print(f"  mode: {info['mode']}")
        print(f"  requires: {info['requires']}")
        print(f"  size: {info['default_size']}")
        print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or stage flood-paper datasets under data/flood/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/download_flood_data.py --list
              python scripts/download_flood_data.py --dataset templates
              python scripts/download_flood_data.py --dataset hydrolakes_points
              python scripts/download_flood_data.py --dataset usgs_streamflow --usgs-sites 01646500 --start 2000-01-01 --end 2024-12-31
              python scripts/download_flood_data.py --dataset noaa_daily_precip --noaa-station GHCND:USW00014732 --start 2020-01-01 --end 2020-12-31
            """
        ),
    )
    parser.add_argument("--list", action="store_true", help="List available dataset recipes.")
    parser.add_argument("--check-credentials", action="store_true", help="Report which provider credentials are available without printing secrets.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Optional .env file with provider credentials.")
    parser.add_argument("--dataset", choices=[*SOURCES.keys(), "templates", "all_open_small"], help="Dataset recipe to run.")
    parser.add_argument("--data-root", help="Directory for flood data. Defaults to FLOOD_DATA_ROOT or data/flood.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    parser.add_argument("--allow-large", action="store_true", help="Allow large direct downloads.")
    parser.add_argument("--start", default="2000-01-01", help="Start date for API downloads.")
    parser.add_argument("--end", default="2024-12-31", help="End date for API downloads.")
    parser.add_argument("--usgs-sites", help="Comma-separated USGS site IDs.")
    parser.add_argument("--usgs-parameter", default="00060", help="USGS parameter code, default 00060 discharge.")
    parser.add_argument("--noaa-station", help="NOAA CDO station ID, e.g. GHCND:USW00014732.")
    parser.add_argument("--max-records", type=int, help="Maximum records for paged metadata downloads.")
    parser.add_argument("--region", choices=REGIONS.keys(), help="Study region key for region-subset downloads.")
    parser.add_argument("--year", type=int, default=2020, help="Year for monthly region downloads.")
    parser.add_argument("--month", type=int, default=1, help="Month for monthly region downloads.")
    parser.add_argument(
        "--era5-variables",
        default="total_precipitation,2m_temperature",
        help="Comma-separated ERA5 single-level variables.",
    )
    parser.add_argument("--ee-project", help="Google Earth Engine project ID.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    load_env_file(Path(args.env_file))
    configure_data_root(args.data_root)
    ensure_dirs()
    if args.check_credentials:
        return check_credentials()
    if args.list:
        list_sources()
        return 0
    if not args.dataset:
        raise SystemExit("Choose --dataset or use --list.")
    if args.dataset == "templates":
        write_templates()
        write_region_manifest()
        return 0
    if args.dataset == "all_open_small":
        write_templates()
        output = RAW / "reservoirs" / "HydroLAKES_points_v10_shp.zip"
        download_url(SOURCES["hydrolakes_points"]["source_url"], output, args.dry_run)
        if not args.dry_run:
            write_metadata("hydrolakes_points", output)
        return 0
    if args.dataset == "usgs_streamflow":
        fetch_usgs(args)
        return 0
    if args.dataset == "noaa_daily_precip":
        fetch_noaa(args)
        return 0
    if args.dataset == "noaa_stations_region":
        fetch_noaa_stations_region(args)
        return 0
    if args.dataset == "era5_hourly_region":
        fetch_era5_hourly_region(args)
        return 0
    if args.dataset == "gfd_event_metadata":
        fetch_gfd_event_metadata(args)
        return 0
    if args.dataset == "gfd_region_summary":
        fetch_gfd_region_summary(args)
        return 0
    if args.dataset == "hydrolakes_points":
        output = RAW / "reservoirs" / "HydroLAKES_points_v10_shp.zip"
        download_url(SOURCES[args.dataset]["source_url"], output, args.dry_run)
        if not args.dry_run:
            write_metadata(args.dataset, output)
        return 0
    if args.dataset == "hydrolakes_polys":
        if not args.allow_large:
            raise SystemExit("hydrolakes_polys is about 820 MB. Re-run with --allow-large to download.")
        output = RAW / "reservoirs" / "HydroLAKES_polys_v10_shp.zip"
        download_url(SOURCES[args.dataset]["source_url"], output, args.dry_run)
        if not args.dry_run:
            write_metadata(args.dataset, output)
        return 0
    if SOURCES[args.dataset]["mode"] == "template":
        if args.dataset == "region_manifest":
            write_region_manifest()
        else:
            write_templates()
            print(f"{args.dataset} is credentialed or provider-specific; see {display_path(RAW / 'manual' / 'download_templates.md')}")
        return 0
    raise SystemExit(f"No runner implemented for {args.dataset}")


if __name__ == "__main__":
    raise SystemExit(main())
