# Flood Paper Linux Handoff

This note captures the reproducible state needed to continue the flood-paper
data acquisition and analysis on a faster Linux machine.

## Branch Purpose

Temporary collaboration branch for flood work:

```bash
git checkout flood-linux-handoff
```

The branch contains scripts and documentation only. Large raw data, credentials,
and local `.env` files are intentionally excluded from Git.

## Local Secrets

Do not commit credentials. Recreate a local environment file on the Linux
machine, for example:

```bash
mkdir -p data/flood
nano data/flood/.env.local
```

Expected variables:

```bash
FLOOD_DATA_ROOT=/path/to/external/FloodData
NOAA_CDO_TOKEN=...
CDSAPI_URL=...
CDSAPI_KEY=...
EARTHDATA_USERNAME=...
EARTHDATA_PASSWORD=...
```

The file is ignored by `.gitignore`.

## External Data Root

Windows data root used so far:

```text
F:\_DATA\FloodData
```

On Linux, mount the same external drive and set:

```bash
export FLOOD_DATA_ROOT=/media/$USER/<drive-name>/FloodData
```

Use the exact mount path reported by Linux.

## Current ERA5 Status

Complete years:

- 2010
- 2015
- 2018
- 2020

Remaining target years:

- 2013
- 2021
- 2022
- 2023
- 2024

Current verified ERA5 archive:

- 384 regional monthly ERA5 files.
- 96 files per complete year.
- All 2010 files were verified as valid ZIP containers with inner NetCDF
  streams for `tp` and `t2m`.

## Main Scripts

Credential check:

```bash
python scripts/check_flood_credentials.py
```

Download one ERA5 region-month:

```bash
python scripts/download_flood_data.py \
  --data-root "$FLOOD_DATA_ROOT" \
  --dataset era5_hourly_region \
  --region ganges_brahmaputra_meghna \
  --year 2013 \
  --month 1
```

Rebuild enriched harmonized event table:

```bash
python scripts/build_flood_event_table.py \
  --data-root "$FLOOD_DATA_ROOT" \
  --include-era5 \
  --event-window-days 7
```

Regenerate draft Figure 1 and Figure 2:

```bash
python scripts/make_flood_figures.py \
  --data-root "$FLOOD_DATA_ROOT" \
  --figure-dir paper/flood_nature_geoscience/figures
```

## Python Dependencies

Minimum analysis/download environment:

```bash
python -m pip install pandas numpy matplotlib xarray h5netcdf netCDF4 cdsapi earthengine-api openpyxl
```

Google Earth Engine should be initialized on the Linux machine:

```bash
earthengine authenticate
earthengine set_project fair-future-496413-f5
```

Copernicus CDS credentials must also be configured through `CDSAPI_URL` and
`CDSAPI_KEY` or `~/.cdsapirc`.

## Notes

- CDS ERA5 downloads are saved with `.nc` filenames but are ZIP containers.
- `scripts/build_flood_event_table.py` extracts the inner NetCDF streams into
  `interim/era5_zip_cache`.
- The current enriched table has 685 event-region rows and 109 rows with ERA5
  precipitation/temperature features after adding 2010.
- Current draft figures are pipeline checks, not Nature-grade final artwork.
