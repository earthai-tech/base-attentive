# Mathematical Geosciences Paper

**Title:** A Physics-Informed Deep Learning Framework for Solving the 2D Shallow Water
Equations in Non-Stationary Flood Environments

**Author:** Kouao Laurent Kouadio (ORCID: 0000-0001-7259-7254)

**Journal:** Mathematical Geosciences (Springer)

---

## Folder Structure

```
flood-mathematical_geosciences/
├── MG.main.tex                  # Main LaTeX manuscript (compile at this level)
├── MG.bst                       # Springer Mathematical Geosciences bibliography style
├── flood-references-cleaned.bib # Curated bibliography
├── MG_template/                 # Original Springer template (reference only)
│
├── figures/                     # Generated paper figures (PDF/PNG, 300+ dpi)
│   └── fig01_africa_study_regions.pdf
│
├── tables/                      # CSV tables produced by analysis scripts
│   └── tab01_africa_regions.csv
│
├── results/                     # Model metrics, JSON outputs, checkpoints
│
├── supplementary/               # Supplementary figures and text
│
└── scripts/                     # Reproducible analysis pipeline
    ├── common.py                # Shared paths, Africa region defs, utilities
    ├── 01_africa_data_audit.py  # Audit F:\_DATA\FloodData for Africa layers
    ├── 02_build_africa_event_table.py  # Build Africa-only flood event table
    ├── 03_build_era5_covariates.py     # Extract ERA5 precipitation for study regions
    ├── 04_padrnet_training.py          # Train PADR-Net (λ=0 and λ>0 ablation)
    └── 05_make_figures.py             # Produce publication-ready figures
```

---

## Data

All flood data live on the external drive at `F:\_DATA\FloodData/`.

| Layer | Path | Used for |
|---|---|---|
| EM-DAT events | `raw/disaster_inventory/public_emdat_2026-05-11.xlsx` | Event labels |
| GFD metadata | `raw/satellite_flood_extent/global_flood_database/` | Flood extent GT |
| ERA5 reanalysis | `raw/reanalysis/era5/` | Precipitation forcing |
| Harmonised events | `interim/harmonized_flood_events.csv` | Pre-built event table |

Africa sub-regions used:
- **West Africa — Niger/Benue** (`west_africa_niger_benue`)
- **East Africa — Nile headwaters** (`east_africa_nile_headwaters`)
- **Southern Africa — Limpopo/Zambezi** (`southern_africa_limpopo_zambezi`)

---

## Compile LaTeX

```bash
cd paper/flood-mathematical_geosciences
pdflatex MG.main.tex
bibtex MG.main
pdflatex MG.main.tex
pdflatex MG.main.tex
```

---

## Script Pipeline (run in order)

```bash
cd paper/flood-mathematical_geosciences
python scripts/01_africa_data_audit.py        # Confirm available layers
python scripts/02_build_africa_event_table.py # Build Africa event table
python scripts/03_build_era5_covariates.py    # Extract ERA5 for 3 regions
python scripts/04_padrnet_training.py         # Train + ablation study
python scripts/05_make_figures.py             # Generate all figures
```
