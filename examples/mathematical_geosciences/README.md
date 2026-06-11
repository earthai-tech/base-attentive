# PADR-Net — Mathematical Geosciences reproducibility

**Paper:** Physics-Informed Reservoir Learning for Shallow-Water Flood Modelling  
**Journal:** Mathematical Geosciences (submitted 2026)  
**Author:** Kouao Laurent Kouadio — [etanoyau@gmail.com](mailto:etanoyau@gmail.com)

This folder contains the scripts used to produce every result, table, and
figure in the manuscript.  Pre-computed result tables are included in the
companion Zenodo archive so that figures can be regenerated without
re-running the full training pipeline.

---

## Quick start — reproduce Tables 1–2 and Figures 4–12

No external data downloads required.  The pre-computed CSVs are bundled in
the Zenodo archive.

```bash
# 1. Clone the repository at the exact version used for submission
git clone https://github.com/earthai-tech/base-attentive
cd base-attentive
git checkout v1.0.0-matg-resubmission

# 2. Create the environment
conda env create -f examples/mathematical_geosciences/environment.yml
conda activate padrnet

# 3. Download the Zenodo archive and unpack it next to the repo
#    (replace the DOI path once the archive is published)
#    https://doi.org/10.5281/zenodo.XXXXXXX

# 4. Point the scripts at the archive data (one-time)
export DATA_ROOT=/path/to/padrnet-matg-resubmission/data    # Linux/macOS
# $env:DATA_ROOT = "C:\path\to\padrnet-matg-resubmission\data"  # Windows PS

# 5. Regenerate all publication figures
cd examples/mathematical_geosciences/scripts
python 06_make_figures.py
```

Figures are written to `results/figures/` as PNG, SVG, and EPS.

---

## Full pipeline — train from scratch

Scripts 01–03 require the external datasets listed in
`data/raw_public_links/download_sources.csv` of the Zenodo archive.
Scripts 04–06 are self-contained once `DATA_ROOT` is set.

```bash
# Step 1: Confirm available data layers
python scripts/01_africa_data_audit.py

# Step 2: Build the Africa flood event inventory (needs EM-DAT + GFD)
python scripts/02_build_africa_event_table.py

# Step 3: Extract ERA5 climate features (needs ~7 GB ERA5 NetCDF files)
#          See data/raw_public_links/ERA5_DOWNLOAD.md in the Zenodo archive
python scripts/03_build_era5_covariates.py

# Step 4: Train PADR-Net and run the M0–M8 ablation study
#          Produces: ablation_results.csv, nested_results.csv,
#                    lambda_sensitivity.csv, transfer_results.csv, bootstrap_ci.csv
python scripts/04_padrnet_training.py

# Step 5: Generate flood scenario time series
python scripts/05_make_flood_scenarios.py

# Step 6: Generate all 14 publication figures
python scripts/06_make_figures.py

# Or run everything at once:
python scripts/run_all.py
```

---

## Script reference

| Script | Purpose | External data needed? |
|---|---|---|
| `common.py` | Shared paths, region definitions, utilities | — |
| `01_africa_data_audit.py` | Audit available data layers per region/year | EM-DAT, GFD, ERA5 |
| `02_build_africa_event_table.py` | Build the 243-event flood inventory | EM-DAT, GFD |
| `03_build_era5_covariates.py` | Extract ERA5 climate features | ERA5 NetCDF (~7 GB) |
| `04_padrnet_training.py` | Train PADR-Net; M0–M8 ablation; LORO/LOYO transfer | processed tables |
| `05_make_flood_scenarios.py` | Generate scenario time series arrays | processed tables |
| `06_make_figures.py` | Reproduce all 14 publication figures | result CSVs |
| `run_all.py` | Run the full pipeline in sequence | see above |

---

## Key results (Table 2)

| Model | Predictor set | Spearman ρ | PR-AUC | NSE_depth | MAE |
|---|---|---|---|---|---|
| M0 | R (rainfall only) | 0.194 | 0.403 | 0.380 | 3.891 |
| M1 | R + M (memory) | 0.614 | 0.620 | 0.380 | 3.749 |
| M4 | R + M + E (exposure) | 0.699 | 0.675 | 0.380 | 3.642 |
| M6 | R + M + E + H (full, λ*=0.1) | 0.671 | 0.703 | **0.643** | 3.569 |

The severity ranking metrics (Spearman ρ, PR-AUC) are invariant to the
physics weight λ — see Proposition 1 in the manuscript.  Depth reconstruction
(NSE_depth) improves only when hydrodynamic terrain features (H) are included.

---

## Data availability

| Dataset | Role | Availability |
|---|---|---|
| EM-DAT flood inventory | Event labels | Register at https://www.emdat.be |
| Global Flood Database | Spatial extent | Register at https://global-flood-database.cloudtostreet.ai |
| ERA5 reanalysis | Climate covariates | Free at https://cds.climate.copernicus.eu |
| Pre-computed feature tables | Direct training input | Zenodo archive (CC-BY 4.0) |

The Zenodo archive contains all pre-computed tables so that scripts 04–06
can be run without downloading any raw data.

---

## Citation

```bibtex
@article{Kouadio2026padrnet,
  author  = {Kouadio, Kouao Laurent},
  title   = {Physics-Informed Reservoir Learning for
             Shallow-Water Flood Modelling},
  journal = {Mathematical Geosciences},
  year    = {2026},
  note    = {Submitted. DOI: 10.5281/zenodo.XXXXXXX}
}
```
