# Scripts — PADR-Net Africa Analysis Pipeline

Run scripts in numbered order from the paper root directory.

| Script | Purpose | Key outputs |
|---|---|---|
| `common.py` | Shared paths, region defs, utilities | *(import only)* |
| `01_africa_data_audit.py` | Audit available data layers on `F:\\_DATA\\FloodData` | `tables/data_audit_africa.csv`, `results/data_audit_africa.json` |
| `02_build_africa_event_table.py` | Subset harmonised events to 3 Africa regions; add EM-DAT fields; assign severity tier + train/val/test split | `tables/africa_flood_events.csv`, `tables/africa_region_summary.csv` |
| `03_build_era5_covariates.py` | Extract ERA5 precipitation windows for each Africa event | `tables/africa_era5_covariates.csv` |
| `04_padrnet_training.py` | Train PADR-Net with λ=0 and λ>0; produce ablation metrics | `results/padrnet_ablation_metrics.csv` |
| `05_make_figures.py` | Generate all publication-ready figures | `figures/fig*.pdf` |

## Prerequisites

```bash
pip install pandas numpy xarray netCDF4 matplotlib cartopy openpyxl
```

Topography (FABDEM/SRTM) and ERA5 NetCDF files must be present under
`F:\_DATA\FloodData\raw\`.  Run script 01 first to check readiness.
