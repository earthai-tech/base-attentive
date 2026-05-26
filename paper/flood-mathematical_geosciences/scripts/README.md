# PADR-Net Scripts — Mathematical Geosciences Paper

Scripts for reproducing all numerical experiments, flood scenarios, and
publication figures in:

> **"A Physics-Informed Deep Learning Framework for Solving the 2D Shallow
> Water Equations in Non-Stationary Flood Environments"**
> K. L. Kouadio — *Mathematical Geosciences* (submitted 2026)

---

## Prerequisites

```
pip install numpy pandas scipy scikit-learn matplotlib
# optional but strongly recommended:
pip install xarray netcdf4 geopandas
```

Data at `F:\_DATA\FloodData` must include:
- `raw/disaster_inventory/public_emdat_2026-05-11.xlsx`
- `raw/satellite_flood_extent/global_flood_database/gfd_event_metadata.json`
- `interim/harmonized_flood_events.csv`
- `raw/reanalysis/era5/` (.nc files)

---

## Pipeline

Run everything in order:

```bash
python scripts/run_all.py
```

Or individual scripts:

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `common.py` | Shared paths, Africa region defs, utilities | *(imported)* |
| `01_africa_data_audit.py` | Audit all data layers on F: drive | `tables/data_audit_africa.csv` |
| `02_build_africa_event_table.py` | Subset 243 Africa events, assign severity tiers and splits | `tables/africa_flood_events.csv` |
| `03_build_era5_covariates.py` | Extract ERA5 precipitation/meteo features per event | `tables/era5_covariates.csv` |
| `04_padrnet_training.py` | Full PADR-Net training, ablation, nested predictors, LORO/LOYO, bootstrap CI | `tables/ablation_results.csv`, `tables/nested_results.csv`, etc. |
| `05_make_flood_scenarios.py` | Generate S1/S2/S3 flood benchmark scenarios | `tables/scenario_results.csv`, `results/scenarios/*.npy` |
| `06_make_figures.py` | Produce all publication figures (PNG + SVG + EPS) | `figures/fig01_*.{png,svg,eps}` ... |
| `run_all.py` | Master runner (calls scripts 01-06 in order) | -- |

Resume from a specific step:

```bash
python scripts/run_all.py --from 04
```

Run only figures:

```bash
python scripts/run_all.py --only 06
```

---

## Model description

**PADR-Net** (Physically-Aware Deep Reservoir Network):

```
L_total = L_data  +  lambda * L_phys
        = MSE(y_hat, y)  +  lambda * ||F(y_hat)||^2
```

where `F(y_hat)` is the residual of the linearised 2D shallow-water equation:

```
F(h_t) = (h_t - h_{t-1})/dt  +  C_f * h_t  -  P_t
```

The reservoir satisfies the Echo State Property: `rho(W_res) < 1` (Lemma 2).
The theoretical error bound is `C(lambda) = O(lambda^{-1/2})` (Theorem 1).

---

## Study regions

| Key | Label | ISO codes |
|-----|-------|-----------|
| `west_africa_niger_benue` | West Africa: Niger/Benue basin | BEN BFA CMR MLI NER NGA TCD |
| `east_africa_nile_headwaters` | East Africa: Nile headwaters | BDI ETH KEN RWA SDN SSD TZA UGA |
| `southern_africa_limpopo_zambezi` | Southern Africa: Limpopo/Zambezi | AGO BWA MWI MOZ NAM ZAF ZMB ZWE |

---

## Figures

| File | Caption |
|------|---------|
| `fig01_region_map` | Africa study region bounding boxes and ISO-country shading |
| `fig02_data_availability` | Flood event inventory heat-map by region and year |
| `fig03_architecture` | PADR-Net architecture schematic |
| `fig04_lambda_sensitivity` | Performance vs. lambda; theoretical error bound verification |
| `fig05_ablation` | PADR-Net-0 vs. PADR-Net-lambda on all metrics |
| `fig06_nested_predictors` | Incremental gain from nested predictor sets |
| `fig07_scenario_s1` | S1 extreme monsoon event: precip + depth time series |
| `fig08_scenario_s2` | S2 rapid-onset flash flood |
| `fig09_scenario_s3` | S3 180-day seasonal sequence |
| `fig10_loro_radar` | Leave-one-region-out transfer skill (radar chart) |
| `fig11_bootstrap_ci` | Bootstrap 95% confidence intervals (violin + box) |
| `fig12_error_bound` | Theorem 1 verification on log-log scale |
| `suppfig01_loyo` | Leave-one-year-out transfer per year |
| `suppfig02_feature_correlation` | Spearman rank correlation heat-map of covariates |

---

## Author

Kouao Laurent Kouadio
ORCID: 0000-0001-7259-7254
Email: etanoyau@gmail.com
