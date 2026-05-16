# Flood Nature Paper Working Plan

This file is the living project spine for building a robust, scientific flood paper suitable for a high-impact journal submission. We will use it to coordinate the narrative, the five main figures, the data acquisition, and the analysis scripts.

## One-Sentence Scientific Claim

Working target:

> We show that flood risk and predictability are controlled by interacting hydroclimatic memory, landscape exposure, and compound meteorological forcing, and that combining independent hydrological, meteorological, satellite, land-surface, and disaster-impact evidence improves flood estimation beyond rainfall-only or single-source approaches.

This sentence should become sharper after the first data audit. A Nature-level paper needs a discovery, not only a model.

## Core Scientific Question

What controls extreme flood occurrence, spatial exposure, and predictability across diverse hydrological regimes, and can those controls be quantified robustly from independent observational evidence?

## Paper Structure

1. Introduction
   - Flood extremes are rising in societal importance, but prediction and risk mapping remain limited by single-source evidence.
   - Existing methods often depend on rainfall thresholds, local gauge records, or static flood maps.
   - The knowledge gap is the lack of a cross-validated, multi-evidence framework connecting flood mechanisms, exposure, and prediction.

2. Data and Study Design
   - Combine observed discharge, rainfall/reanalysis, satellite flood extent, land-surface state, exposure layers, reservoirs, urbanization, historical disaster records, and climate scenario products.
   - Define flood events consistently across data streams.
   - Use held-out basins, held-out years, and held-out climate regimes for generalization tests.

3. Main Analysis
   - Build event labels and flood magnitudes from gauge and satellite observations.
   - Build predictors from antecedent rainfall, soil moisture, topography, land cover, urbanization, reservoir influence, and basin attributes.
   - Compare rainfall-only, hydrological-statistical, machine learning, and hybrid/attention-based models.
   - Quantify uncertainty with event-definition, dataset, spatial-resolution, and model sensitivity analyses.

4. Mechanistic Interpretation
   - Identify which factors explain failure modes of rainfall-only approaches.
   - Test whether soil moisture, terrain, storage, land cover, and storm sequencing explain regional differences in flood predictability.
   - Use ablations and interpretable model diagnostics to connect performance to hydrological mechanism.

5. Implications
   - Show where current flood risk is underestimated.
   - Show how the framework can support monitoring, early warning, and scenario stress testing.
   - Be careful not to overclaim operational readiness unless validated against operational products.

## Five Main Figures

### Figure 1: Study Design and Evidence Streams

Purpose: show the scientific architecture of the paper.

Panels:
- Map of selected basins/events and climate regimes.
- Data stream schematic: discharge, rainfall/reanalysis, satellite flood extent, land surface, exposure, historical disaster records, climate scenarios.
- Event definition workflow from raw observations to harmonized flood-event table.

Script to build later: `scripts/flood_fig01_study_design.py`

### Figure 2: Observed Flood Patterns and Data Agreement

Purpose: demonstrate that flood events are observable from multiple independent sources.

Panels:
- Gauge-derived extreme discharge events.
- Satellite-derived maximum flood extent or flood frequency.
- Disaster-inventory event density and reported impacts.
- Agreement/disagreement map across evidence streams.

Script to build later: `scripts/flood_fig02_observed_patterns.py`

### Figure 3: Main Prediction or Risk Result

Purpose: deliver the central quantitative result.

Panels:
- Performance comparison against baselines.
- Reliability/calibration curves.
- Event-level examples for major floods.
- Spatial distribution of skill improvements.

Script to build later: `scripts/flood_fig03_main_result.py`

### Figure 4: Generalization Across Space, Time, and Regimes

Purpose: prove the result is not a local case study.

Panels:
- Train-region to test-region transfer.
- Historical-period to recent-period transfer.
- Climate-zone or basin-type breakdown.
- Data-poor condition experiment.

Script to build later: `scripts/flood_fig04_generalization.py`

### Figure 5: Mechanism, Attribution, or Future-Risk Implication

Purpose: explain why the result matters scientifically.

Panels:
- Feature/driver contribution by flood regime.
- Counterfactual ablation: rainfall-only vs rainfall + antecedent soil moisture + terrain/storage/urbanization.
- Scenario stress test under climate projections, if used.
- Risk underestimation or exposure shift map.

Script to build later: `scripts/flood_fig05_mechanism_implications.py`

## Data Evidence Streams

Raw and intermediate data can live under local `data/flood/` for small pilots or on the external drive at `F:\_DATA\FloodData` for serious downloads. The repository `.gitignore` already excludes `data/`, and the external drive is outside the repository, so large files will stay out of git.

### 1. Observed River Discharge / Gauge Records

Primary options:
- USGS NWIS daily values for US basins.
- GRDC for global gauges, if we obtain access.
- Caravan/CAMELS-style basin datasets for benchmark hydrology.

Initial script target: download USGS daily streamflow for selected gauges using the NWIS daily values API.

### 2. Rainfall and Meteorological Reanalysis

Primary options:
- ERA5 / ERA5-Land from the Copernicus Climate Data Store.
- NOAA daily station precipitation through Climate Data Online.
- GPM IMERG for satellite precipitation if Earthdata access is configured.

Initial script target: provide a CDS API request template for ERA5 and NOAA CDO support when `NOAA_CDO_TOKEN` is set.

### 3. Satellite Flood Extent

Primary options:
- Global Flood Database v1 in Google Earth Engine.
- NASA MODIS/VIIRS near-real-time global flood products through Earthdata.
- Dartmouth Flood Observatory event archive as an event reference.

Initial script target: create instructions and export stubs for Earth Engine / NASA Earthdata products.

### 4. Land Cover, Soil Moisture, Topography, Reservoirs, Urbanization

Primary options:
- ESA WorldCover for land cover.
- SMAP for soil moisture.
- Copernicus DEM or SRTM for terrain.
- HydroLAKES and GRanD for lakes, reservoirs, and dams.
- GHSL for built-up area and population exposure.

Initial script target: download HydroLAKES layers directly and save command templates for the larger/provider-specific products.

### 5. Historical Flood Inventories / Disaster Databases

Primary options:
- EM-DAT public data after registration.
- Dartmouth Flood Observatory event archive.
- Global Flood Database metadata.

Initial script target: document required credentials and expected local file names.

### 6. Climate Model / Scenario Data

Primary options:
- NASA NEX-GDDP-CMIP6 for daily downscaled climate scenarios.
- CMIP6 cloud archives through Pangeo/ESGF for broader model ensembles.

Initial script target: provide a NASA NEX-GDDP-CMIP6 subset template for precipitation.

## Validation and Robustness Checklist

- Train/test split by basin, not only by random events.
- Hold out recent extreme events.
- Hold out climate regimes.
- Compare against rainfall-only, flood-frequency, simple ML, and existing operational/static products where available.
- Report confidence intervals by basin and by event.
- Run event-definition sensitivity tests.
- Run spatial-resolution sensitivity tests.
- Run missing-data and data-poor tests.
- Run ablation tests for rainfall, soil moisture, topography, land cover, reservoirs, and urbanization.

## Reproducibility Rules

- Raw data: `data/flood/raw/`
- Processed data: `data/flood/processed/`
- Interim event tables: `data/flood/interim/`
- Figures: `paper/flood_nature_geoscience/figures/`
- Tables: `paper/flood_nature_geoscience/tables/`
- Data download logs: `data/flood/download_logs/`

External drive equivalent:

```powershell
python scripts/download_flood_data.py --data-root F:\_DATA\FloodData --dataset templates
```

Every script should write a small metadata JSON next to its output with source name, source URL, access date, spatial extent, temporal extent, variables, processing decisions, and checksum where practical.

## Current Data Acquisition Status

External data root: `F:\_DATA\FloodData`

ERA5 hourly regional reanalysis status:
- Years complete: 2010, 2015, 2018, 2020.
- Spatial coverage: all eight study regions.
- File structure: `raw/reanalysis/era5/<region>/era5_hourly_<region>_<year>_<month>.nc`.
- Verification on 2026-05-16: 384 ERA5 files total; 96 files for 2010; no missing 2010 month-region files.
- Approximate ERA5 footprint: 3.01 GB total; 0.76 GB for 2010.

Other acquired evidence streams:
- Global Flood Database metadata and regional event summary are present.
- NOAA GHCND station inventories are present for all eight study regions.
- HydroLAKES point and polygon archives are present.
- EM-DAT public file is present.
- USGS pilot daily discharge file is present.

## Current Analysis Pipeline

Harmonized event table:

```powershell
python scripts/build_flood_event_table.py --data-root F:\_DATA\FloodData
```

Current outputs:
- `F:\_DATA\FloodData\interim\harmonized_flood_events.csv`
- `F:\_DATA\FloodData\interim\region_evidence_summary.csv`
- `F:\_DATA\FloodData\interim\region_year_event_counts.csv`
- `F:\_DATA\FloodData\interim\harmonized_flood_events.metadata.json`

Current table status:
- 685 event-region rows from the Global Flood Database regional intersections.
- Eight study regions represented.
- Target years encoded for planned downloads: 2010, 2013, 2015, 2018, 2020, 2021, 2022, 2023, 2024.
- Downloaded ERA5 years currently detected: 2015, 2018, 2020.
- Downloaded ERA5 years currently detected after the latest update: 2010, 2015, 2018, 2020.
- Event rows include region metadata, GFD identifiers, dates, duration, NOAA station-inventory count, local ERA5 month coverage, and availability flags for GFD, NOAA, HydroLAKES, and EM-DAT.
- NetCDF backends installed in the active Anaconda Python: `h5netcdf` and `netCDF4`.
- CDS ERA5 files are ZIP containers saved with `.nc` filenames; `scripts/build_flood_event_table.py` now extracts the inner `tp` and `t2m` NetCDF streams into `F:\_DATA\FloodData\interim\era5_zip_cache`.
- ERA5 event-window features are now written where local ERA5 overlaps the event window: total precipitation, mean hourly precipitation, max hourly precipitation, mean 2 m temperature, min 2 m temperature, and max 2 m temperature.
- Current enriched status after adding 2010: 109 event-region rows have ERA5 event-window features; 576 rows correctly report no local ERA5 files for the event window.
- Corrupt zero-byte Mekong July 2015 ERA5 file was refreshed on 2026-05-16.

Figure draft pipeline:

```powershell
python scripts/make_flood_figures.py --data-root F:\_DATA\FloodData --figure-dir paper\flood_nature_geoscience\figures
```

Current outputs:
- `paper/flood_nature_geoscience/figures/figure_1_region_evidence_design.png`
- `paper/flood_nature_geoscience/figures/figure_1_region_evidence_design.pdf`
- `paper/flood_nature_geoscience/figures/figure_2_event_timeline_and_coverage.png`
- `paper/flood_nature_geoscience/figures/figure_2_event_timeline_and_coverage.pdf`
- `paper/flood_nature_geoscience/figures/figure_manifest.json`

Figure 1 currently summarizes the eight-region design, GFD event counts, and NOAA station-inventory counts. Figure 2 currently summarizes event timing, local ERA5-year availability, and event-duration distribution. These are reproducible draft figures, not final Nature artwork.

## Immediate Next Steps

1. Download the remaining target years step by step: 2013, 2021, 2022, 2023, 2024.
2. Rerun the enriched event table after each new year lands:
   `python scripts/build_flood_event_table.py --data-root F:\_DATA\FloodData --include-era5 --event-window-days 7`
3. Design Nature-grade Figure 1 and Figure 2 concepts before redrawing the current drafts.
4. Start the Figure 3 analysis pipeline: rainfall anomaly / event severity model.
