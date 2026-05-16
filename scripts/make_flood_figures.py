"""Create Figure 1 and Figure 2 drafts from the harmonized flood tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


DEFAULT_DATA_ROOT = Path(r"F:\_DATA\FloodData")
DEFAULT_FIGURE_DIR = Path("paper/flood_nature_geoscience/figures")
REGION_SHORT = {
    "west_africa_niger_benue": "W. Africa",
    "southern_africa_limpopo_zambezi": "S. Africa",
    "east_africa_nile_headwaters": "E. Africa",
    "ganges_brahmaputra_meghna": "GBM",
    "indus": "Indus",
    "rhine_meuse": "Rhine-Meuse",
    "mississippi_missouri_texas_gulf": "Mississippi",
    "mekong": "Mekong",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    return parser.parse_args()


def load_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interim = data_root / "interim"
    events = pd.read_csv(interim / "harmonized_flood_events.csv")
    summary = pd.read_csv(interim / "region_evidence_summary.csv")
    yearly = pd.read_csv(interim / "region_year_event_counts.csv")
    return events, summary, yearly


def detect_era5_years(data_root: Path) -> list[int]:
    years: set[int] = set()
    for path in (data_root / "raw" / "reanalysis" / "era5").glob("*/*.nc"):
        parts = path.stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            years.add(int(parts[1]))
    return sorted(years)


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str, formats: list[str]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(figure_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")


def make_figure_1(events: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path, formats: list[str]) -> None:
    fig = plt.figure(figsize=(11, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 0.08, 1.15], height_ratios=[1, 1])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_bar = fig.add_subplot(grid[0, 2])
    ax_station = fig.add_subplot(grid[1, 2])

    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-60, 75)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Figure 1a. Eight-region flood evidence design", loc="left", fontweight="bold")
    ax_map.grid(color="#dddddd", linewidth=0.6)
    ax_map.axhline(0, color="#999999", linewidth=0.7)
    ax_map.axvline(0, color="#999999", linewidth=0.7)

    count_by_region = summary.set_index("region")["gfd_event_count"].to_dict()
    norm = plt.Normalize(summary["gfd_event_count"].min(), summary["gfd_event_count"].max())
    cmap = plt.get_cmap("viridis")

    region_rows = events.drop_duplicates("region").copy()
    for _, row in region_rows.iterrows():
        west = row["bbox_west"]
        east = row["bbox_east"]
        south = row["bbox_south"]
        north = row["bbox_north"]
        count = count_by_region[row["region"]]
        rect = Rectangle(
            (west, south),
            east - west,
            north - south,
            facecolor=cmap(norm(count)),
            edgecolor="#222222",
            alpha=0.45,
            linewidth=1.2,
        )
        ax_map.add_patch(rect)
        ax_map.text(
            (west + east) / 2,
            (south + north) / 2,
            REGION_SHORT.get(row["region"], row["region"]),
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
        )

    ordered = summary.sort_values("gfd_event_count", ascending=True)
    labels = [REGION_SHORT.get(region, region) for region in ordered["region"]]
    ax_bar.barh(labels, ordered["gfd_event_count"], color="#287c8e")
    ax_bar.set_title("Figure 1b. GFD events", loc="left", fontweight="bold")
    ax_bar.set_xlabel("Event-region count")
    ax_bar.spines[["top", "right"]].set_visible(False)

    station_ordered = summary.sort_values("noaa_prcp_station_count", ascending=True)
    station_labels = [REGION_SHORT.get(region, region) for region in station_ordered["region"]]
    ax_station.barh(station_labels, station_ordered["noaa_prcp_station_count"], color="#b77f24")
    ax_station.set_title("Figure 1c. NOAA station inventory", loc="left", fontweight="bold")
    ax_station.set_xlabel("Stations")
    ax_station.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Multi-region flood catalogue and evidence coverage", fontsize=14, fontweight="bold")
    save_figure(fig, figure_dir, "figure_1_region_evidence_design", formats)
    plt.close(fig)


def make_figure_2(events: pd.DataFrame, yearly: pd.DataFrame, data_root: Path, figure_dir: Path, formats: list[str]) -> None:
    era5_years = detect_era5_years(data_root)
    min_year = min(int(events["start_year"].min()), min(era5_years or [9999]))
    max_year = max(int(events["start_year"].max()), max(era5_years or [0]))
    years = list(range(min_year, max_year + 1))
    regions = list(events.drop_duplicates("region")["region"])
    matrix = np.zeros((len(regions), len(years)))
    lookup = yearly.set_index(["region", "year"])["event_count"].to_dict()
    for i, region in enumerate(regions):
        for j, year in enumerate(years):
            matrix[i, j] = lookup.get((region, year), 0)

    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.1, 1])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[1, 0])

    im = ax_heat.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax_heat.set_title("Figure 2a. Regional GFD event timing and ERA5 availability", loc="left", fontweight="bold")
    ax_heat.set_yticks(range(len(regions)))
    ax_heat.set_yticklabels([REGION_SHORT.get(region, region) for region in regions])
    tick_step = 2 if len(years) < 30 else 5
    tick_positions = list(range(0, len(years), tick_step))
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels([years[i] for i in tick_positions], rotation=45, ha="right")
    for year in era5_years:
        if year in years:
            ax_heat.axvline(years.index(year), color="#d73027", linewidth=1.8, alpha=0.9)
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.01)
    cbar.set_label("GFD events")

    durations = events["duration_days"].clip(upper=120)
    ax_hist.hist(durations, bins=24, color="#4c78a8", edgecolor="white")
    ax_hist.set_title("Figure 2b. Event-duration distribution", loc="left", fontweight="bold")
    ax_hist.set_xlabel("Duration days, clipped at 120")
    ax_hist.set_ylabel("Event-region rows")
    ax_hist.spines[["top", "right"]].set_visible(False)

    note = "Red vertical lines mark local ERA5 years: " + ", ".join(str(year) for year in era5_years)
    fig.text(0.01, 0.01, note, fontsize=8, color="#444444")
    fig.suptitle("Event-table readiness for harmonized regional analysis", fontsize=14, fontweight="bold")
    save_figure(fig, figure_dir, "figure_2_event_timeline_and_coverage", formats)
    plt.close(fig)


def write_figure_manifest(data_root: Path, figure_dir: Path, formats: list[str]) -> None:
    manifest = {
        "data_root": str(data_root),
        "figure_dir": str(figure_dir.resolve()),
        "formats": formats,
        "inputs": [
            str(data_root / "interim" / "harmonized_flood_events.csv"),
            str(data_root / "interim" / "region_evidence_summary.csv"),
            str(data_root / "interim" / "region_year_event_counts.csv"),
        ],
        "outputs": [
            str(figure_dir / f"figure_1_region_evidence_design.{fmt}") for fmt in formats
        ]
        + [str(figure_dir / f"figure_2_event_timeline_and_coverage.{fmt}") for fmt in formats],
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    figure_dir = args.figure_dir.resolve()
    events, summary, yearly = load_tables(data_root)
    make_figure_1(events, summary, figure_dir, args.formats)
    make_figure_2(events, yearly, data_root, figure_dir, args.formats)
    write_figure_manifest(data_root, figure_dir, args.formats)
    print(f"Wrote Figure 1 and Figure 2 drafts to {figure_dir}")


if __name__ == "__main__":
    main()
