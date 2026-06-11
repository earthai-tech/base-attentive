"""Create manuscript figures from the harmonized flood tables."""

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

REGION_ORDER = [
    "west_africa_niger_benue",
    "southern_africa_limpopo_zambezi",
    "east_africa_nile_headwaters",
    "ganges_brahmaputra_meghna",
    "indus",
    "rhine_meuse",
    "mississippi_missouri_texas_gulf",
    "mekong",
]

REGION_COLORS = {
    "west_africa_niger_benue": "#7a3b2e",
    "southern_africa_limpopo_zambezi": "#b06c49",
    "east_africa_nile_headwaters": "#d8a45f",
    "ganges_brahmaputra_meghna": "#4f9a8b",
    "indus": "#2e6f9e",
    "rhine_meuse": "#5c5f86",
    "mississippi_missouri_texas_gulf": "#8c6bb1",
    "mekong": "#3f7f5f",
}

SOURCE_COLORS = {
    "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1": "#355c7d",
    "EM-DAT Public Table": "#d08b39",
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


def _load_world() -> object | None:
    try:
        import geopandas as gpd

        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        return None


def _source_label(source: str) -> str:
    if source.startswith("GLOBAL_FLOOD_DB"):
        return "GFD"
    if source.startswith("EM-DAT"):
        return "EM-DAT"
    return source


def make_figure_1(events: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path, formats: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    events = events.copy()
    events["start_year"] = events["start_year"].astype(int)
    events["has_era5"] = events["era5_window_total_precip_mm"].notna()
    target_years = [2010, 2013, 2015, 2018, 2020, 2021, 2022, 2023, 2024]
    ordered_regions = [region for region in REGION_ORDER if region in set(events["region"])]
    region_rows = events.drop_duplicates("region").set_index("region").loc[ordered_regions].reset_index()

    fig = plt.figure(figsize=(7.2, 8.1), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[2.25, 1.05, 1.1],
        width_ratios=[1.45, 1.0],
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.075,
        hspace=0.34,
        wspace=0.23,
    )
    ax_map = fig.add_subplot(grid[0, :])
    ax_source = fig.add_subplot(grid[1, 0])
    ax_era5 = fig.add_subplot(grid[1, 1])
    ax_timeline = fig.add_subplot(grid[2, :])

    world = _load_world()
    ax_map.set_facecolor("#f7fbfd")
    if world is not None:
        world.plot(ax=ax_map, color="#efeee9", edgecolor="#c7c9c7", linewidth=0.25)
    ax_map.set_xlim(-170, 150)
    ax_map.set_ylim(-38, 62)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    for idx, row in enumerate(region_rows.itertuples(index=False), start=1):
        region = row.region
        west, east = float(row.bbox_west), float(row.bbox_east)
        south, north = float(row.bbox_south), float(row.bbox_north)
        color = REGION_COLORS[region]
        rect = Rectangle(
            (west, south),
            east - west,
            north - south,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.4,
        )
        ax_map.add_patch(rect)
        cx, cy = (west + east) / 2, (south + north) / 2
        ax_map.scatter(cx, cy, s=105, color=color, edgecolor="white", linewidth=1.1, zorder=4)
        ax_map.text(cx, cy, str(idx), ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=5)
        label_x = min(max(cx + 7, -160), 140)
        label_y = min(max(cy + 4, -33), 58)
        ax_map.text(
            label_x,
            label_y,
            REGION_SHORT.get(region, region),
            color="#202020",
            fontsize=7,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
        )

    ax_map.text(-166, 58, "a", fontsize=11, fontweight="bold", va="top")
    ax_map.text(
        -151,
        58,
        "Eight-region design linking satellite events, disaster inventories and ERA5 windows",
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax_map.text(
        -151,
        50,
        "Boxes mark analysis domains; numbered points mark domain centroids used for the regional summaries.",
        fontsize=7,
        color="#4b4b4b",
    )

    source_counts = (
        events.groupby(["region", "event_inventory_source"])
        .size()
        .unstack(fill_value=0)
        .reindex(ordered_regions)
    )
    y = np.arange(len(ordered_regions))
    left = np.zeros(len(ordered_regions))
    for source in SOURCE_COLORS:
        values = source_counts[source].to_numpy() if source in source_counts else np.zeros(len(ordered_regions))
        ax_source.barh(y, values, left=left, color=SOURCE_COLORS[source], height=0.62, label=_source_label(source))
        left += values
    ax_source.set_yticks(y)
    ax_source.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_source.invert_yaxis()
    ax_source.set_xlabel("Event-region rows")
    ax_source.set_title("b  Event inventory composition", loc="left", fontweight="bold", pad=6)
    ax_source.spines[["top", "right"]].set_visible(False)
    ax_source.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax_source.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.2), handlelength=1.2)

    era5_region = (
        events.assign(has_era5=events["era5_window_total_precip_mm"].notna())
        .groupby("region")["has_era5"]
        .agg(["sum", "count"])
        .reindex(ordered_regions)
    )
    era5_pct = (era5_region["sum"] / era5_region["count"] * 100).fillna(0)
    ax_era5.barh(y, era5_pct.to_numpy(), color="#506d84", height=0.62)
    ax_era5.set_yticks(y)
    ax_era5.set_yticklabels([])
    ax_era5.invert_yaxis()
    ax_era5.set_xlim(0, 105)
    ax_era5.set_xlabel("Rows with ERA5 window (%)")
    ax_era5.set_title("c  Hydroclimate feature coverage", loc="left", fontweight="bold", pad=6)
    ax_era5.spines[["top", "right"]].set_visible(False)
    ax_era5.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    for yi, value in zip(y, era5_pct):
        ax_era5.text(min(value + 2, 101), yi, f"{value:.0f}%", va="center", fontsize=7, color="#333333")

    timeline = (
        events[events["start_year"].isin(target_years)]
        .groupby(["region", "start_year"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ordered_regions, columns=target_years, fill_value=0)
    )
    matrix = timeline.to_numpy(dtype=float)
    vmax = max(float(matrix.max()), 1.0)
    im = ax_timeline.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
    ax_timeline.set_yticks(np.arange(len(ordered_regions)))
    ax_timeline.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_timeline.set_xticks(np.arange(len(target_years)))
    ax_timeline.set_xticklabels(target_years)
    ax_timeline.set_title("d  Target-year events now linked to local ERA5 archives", loc="left", fontweight="bold", pad=6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value > 0:
                ax_timeline.text(j, i, str(value), ha="center", va="center", fontsize=6.5, color="#17202a")
    for spine in ax_timeline.spines.values():
        spine.set_visible(False)
    ax_timeline.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax_timeline, fraction=0.018, pad=0.012)
    cbar.set_label("Events")
    cbar.outline.set_visible(False)

    total_rows = len(events)
    gfd_rows = int((events["event_inventory_source"] == "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").sum())
    emdat_rows = int((events["event_inventory_source"] == "EM-DAT Public Table").sum())
    era5_rows = int(events["has_era5"].sum())
    fig.text(
        0.055,
        0.985,
        f"Figure 1 | Multi-evidence flood-event design ({total_rows} rows; {gfd_rows} GFD, {emdat_rows} EM-DAT; {era5_rows} ERA5-enriched).",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.025,
        "GFD provides satellite event metadata through 2018; EM-DAT extends disaster-event coverage for 2020-2024. "
        "ERA5 summaries use 7-day pre/post event windows.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
    )

    save_figure(fig, figure_dir, "figure_1_region_evidence_design", formats)
    plt.close(fig)


def make_figure_2(events: pd.DataFrame, yearly: pd.DataFrame, data_root: Path, figure_dir: Path, formats: list[str]) -> None:
    events = events.copy()
    events["start_year"] = events["start_year"].astype(int)
    for column in [
        "era5_window_total_precip_mm",
        "era5_window_max_hourly_precip_mm",
        "era5_window_mean_t2m_c",
        "duration_days",
        "emdat_total_affected",
    ]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    enriched = events[events["era5_window_total_precip_mm"].notna()].copy()
    ordered_regions = [region for region in REGION_ORDER if region in set(enriched["region"])]
    target_years = [2010, 2013, 2015, 2018, 2020, 2021, 2022, 2023, 2024]

    fig = plt.figure(figsize=(7.2, 8.2), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.55, 1.2, 1.25],
        width_ratios=[1.1, 1.0],
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.075,
        hspace=0.56,
        wspace=0.34,
    )
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    ax_lollipop = fig.add_subplot(grid[1, :])
    ax_heat = fig.add_subplot(grid[2, :])

    source_markers = {
        "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1": "o",
        "EM-DAT Public Table": "^",
    }
    for source, marker in source_markers.items():
        subset = enriched[enriched["event_inventory_source"] == source]
        if subset.empty:
            continue
        colors = [REGION_COLORS.get(region, "#666666") for region in subset["region"]]
        affected = subset["emdat_total_affected"].fillna(0)
        sizes = 25 + 28 * np.log10(affected.clip(lower=0) + 1) / max(np.log10(affected.max() + 1), 1)
        ax_scatter.scatter(
            subset["duration_days"].clip(lower=1),
            subset["era5_window_total_precip_mm"].clip(lower=0.1),
            c=colors,
            s=sizes,
            marker=marker,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.78,
            label=_source_label(source),
        )
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Flood duration (days, log scale)")
    ax_scatter.set_ylabel("Event-window precipitation (mm, log scale)")
    ax_scatter.set_title("a  Event hydroclimate phase space", loc="left", fontweight="bold", pad=6)
    ax_scatter.grid(True, which="major", color="#e6e6e6", linewidth=0.6)
    ax_scatter.spines[["top", "right"]].set_visible(False)
    ax_scatter.legend(frameon=False, loc="lower right")

    box_data = [
        enriched.loc[enriched["region"] == region, "era5_window_total_precip_mm"].dropna().to_numpy()
        for region in ordered_regions
    ]
    box = ax_box.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#8a8a8a", "linewidth": 0.8},
        capprops={"color": "#8a8a8a", "linewidth": 0.8},
    )
    for patch, region in zip(box["boxes"], ordered_regions):
        patch.set_facecolor(REGION_COLORS[region])
        patch.set_alpha(0.72)
        patch.set_edgecolor("none")
    ax_box.set_yticks(np.arange(1, len(ordered_regions) + 1))
    ax_box.set_yticklabels([REGION_SHORT[region] for region in ordered_regions])
    ax_box.invert_yaxis()
    ax_box.set_xscale("log")
    ax_box.set_xlabel("Precipitation (mm, log scale)")
    ax_box.set_title("b  Regional precipitation distributions", loc="left", fontweight="bold", pad=6)
    ax_box.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax_box.spines[["top", "right"]].set_visible(False)

    stats = (
        enriched.groupby("region")
        .agg(
            total_median=("era5_window_total_precip_mm", "median"),
            total_q75=("era5_window_total_precip_mm", lambda s: s.quantile(0.75)),
            peak_median=("era5_window_max_hourly_precip_mm", "median"),
            n=("era5_window_total_precip_mm", "count"),
        )
        .reindex(ordered_regions)
    )
    y = np.arange(len(ordered_regions))
    ax_lollipop.hlines(y, stats["total_median"], stats["total_q75"], color="#b8b8b8", linewidth=2.2)
    ax_lollipop.scatter(
        stats["total_median"],
        y,
        s=78,
        color=[REGION_COLORS[r] for r in ordered_regions],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Median total",
    )
    ax_lollipop.set_yticks(y)
    ax_lollipop.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_lollipop.invert_yaxis()
    ax_lollipop.set_xlabel("Median total precipitation; line extends to 75th percentile (mm)")
    ax_lollipop.set_title("c  Regional event-window intensity fingerprints", loc="left", fontweight="bold", pad=12)
    ax_lollipop.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax_lollipop.spines[["top", "right"]].set_visible(False)
    for yi, row in enumerate(stats.itertuples()):
        ax_lollipop.text(
            row.total_q75 + 8,
            yi,
            f"n={int(row.n)}; peak={row.peak_median:.2f} mm h$^{{-1}}$",
            va="center",
            fontsize=6.5,
            color="#555555",
        )

    heat = (
        enriched[enriched["start_year"].isin(target_years)]
        .groupby(["region", "start_year"])["era5_window_total_precip_mm"]
        .median()
        .unstack()
        .reindex(index=ordered_regions, columns=target_years)
    )
    matrix = heat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#f0f0f0")
    vmax = np.nanpercentile(matrix, 95) if np.isfinite(matrix).any() else 1.0
    im = ax_heat.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax_heat.set_yticks(np.arange(len(ordered_regions)))
    ax_heat.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_heat.set_xticks(np.arange(len(target_years)))
    ax_heat.set_xticklabels(target_years)
    ax_heat.set_title("d  Median event-window precipitation by target year", loc="left", fontweight="bold", pad=6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax_heat.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=6.3, color="#17202a")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.012)
    cbar.set_label("Median precipitation (mm)")
    cbar.outline.set_visible(False)

    fig.text(
        0.075,
        0.985,
        f"Figure 2 | ERA5 hydroclimate fingerprints of {len(enriched)} enriched flood-event windows.",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.025,
        "Each row summarizes a flood event-region window using ERA5 total precipitation, maximum hourly precipitation and 2 m temperature. "
        "Panels use only rows with successful ERA5 summaries.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    save_figure(fig, figure_dir, "figure_2_era5_hydroclimate_fingerprints", formats)
    plt.close(fig)


def make_figure_3(events: pd.DataFrame, figure_dir: Path, formats: list[str]) -> None:
    events = events.copy()
    events["start_year"] = events["start_year"].astype(int)
    for column in [
        "era5_window_total_precip_mm",
        "era5_window_max_hourly_precip_mm",
        "duration_days",
        "emdat_total_deaths",
        "emdat_total_affected",
    ]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    impact = events[
        (events["event_inventory_source"] == "EM-DAT Public Table")
        & events["era5_window_total_precip_mm"].notna()
    ].copy()
    ordered_regions = [region for region in REGION_ORDER if region in set(impact["region"])]
    recent_years = [2020, 2021, 2022, 2023, 2024]

    fig = plt.figure(figsize=(7.2, 8.0), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.35, 1.15, 1.25],
        width_ratios=[1.12, 1.0],
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.075,
        hspace=0.48,
        wspace=0.34,
    )
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_deaths = fig.add_subplot(grid[0, 1])
    ax_bars = fig.add_subplot(grid[1, :])
    ax_heat = fig.add_subplot(grid[2, :])

    scatter = impact[impact["emdat_total_affected"].notna()].copy()
    colors = [REGION_COLORS.get(region, "#666666") for region in scatter["region"]]
    deaths = scatter["emdat_total_deaths"].fillna(0)
    sizes = 26 + 42 * np.log10(deaths.clip(lower=0) + 1) / max(np.log10(deaths.max() + 1), 1)
    ax_scatter.scatter(
        scatter["era5_window_total_precip_mm"].clip(lower=0.1),
        scatter["emdat_total_affected"].clip(lower=1),
        c=colors,
        s=sizes,
        marker="o",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.82,
    )
    if len(scatter) > 2:
        x = np.log10(scatter["era5_window_total_precip_mm"].clip(lower=0.1).to_numpy())
        y_log = np.log10(scatter["emdat_total_affected"].clip(lower=1).to_numpy())
        slope, intercept = np.polyfit(x, y_log, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = 10 ** (intercept + slope * xx)
        ax_scatter.plot(10**xx, yy, color="#202020", linewidth=1.2, alpha=0.85)
        corr = np.corrcoef(x, y_log)[0, 1]
        ax_scatter.text(
            0.03,
            0.94,
            f"log-log r={corr:.2f}",
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#333333",
        )
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Event-window precipitation (mm)")
    ax_scatter.set_ylabel("Reported affected people")
    ax_scatter.set_title("a  Hydroclimate forcing and reported impact", loc="left", fontweight="bold", pad=6)
    ax_scatter.grid(True, which="major", color="#e6e6e6", linewidth=0.6)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    death_data = [
        impact.loc[impact["region"] == region, "emdat_total_deaths"].dropna().clip(lower=1).to_numpy()
        for region in ordered_regions
    ]
    box = ax_deaths.boxplot(
        death_data,
        vert=False,
        patch_artist=True,
        widths=0.56,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#8a8a8a", "linewidth": 0.8},
        capprops={"color": "#8a8a8a", "linewidth": 0.8},
    )
    for patch, region in zip(box["boxes"], ordered_regions):
        patch.set_facecolor(REGION_COLORS[region])
        patch.set_alpha(0.72)
        patch.set_edgecolor("none")
    ax_deaths.set_yticks(np.arange(1, len(ordered_regions) + 1))
    ax_deaths.set_yticklabels([REGION_SHORT[region] for region in ordered_regions])
    ax_deaths.invert_yaxis()
    ax_deaths.set_xscale("log")
    ax_deaths.set_xlabel("Reported deaths, non-zero events")
    ax_deaths.set_title("b  Mortality distribution by region", loc="left", fontweight="bold", pad=6)
    ax_deaths.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax_deaths.spines[["top", "right"]].set_visible(False)

    regional = (
        impact.groupby("region")
        .agg(
            affected_sum=("emdat_total_affected", "sum"),
            deaths_sum=("emdat_total_deaths", "sum"),
            precip_median=("era5_window_total_precip_mm", "median"),
            n=("event_region_id", "count"),
        )
        .reindex(ordered_regions)
        .fillna(0)
    )
    y = np.arange(len(ordered_regions))
    affected_m = regional["affected_sum"] / 1_000_000.0
    ax_bars.barh(y, affected_m, color=[REGION_COLORS[r] for r in ordered_regions], height=0.58)
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_bars.invert_yaxis()
    ax_bars.set_xlabel("Reported affected people, 2020-2024 total (millions)")
    ax_bars.set_title("c  Regional impact burden in the recent inventory period", loc="left", fontweight="bold", pad=6)
    ax_bars.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax_bars.spines[["top", "right"]].set_visible(False)
    for yi, row in enumerate(regional.itertuples()):
        ax_bars.text(
            affected_m.iloc[yi] + 0.25,
            yi,
            f"deaths={row.deaths_sum:.0f}; n={int(row.n)}",
            va="center",
            fontsize=6.7,
            color="#555555",
        )

    heat = (
        impact[impact["start_year"].isin(recent_years)]
        .groupby(["region", "start_year"])["emdat_total_affected"]
        .sum(min_count=1)
        .unstack()
        .reindex(index=ordered_regions, columns=recent_years)
    )
    matrix = np.log10(heat.to_numpy(dtype=float) + 1)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")
    vmax = np.nanpercentile(matrix, 95) if np.isfinite(matrix).any() else 1.0
    im = ax_heat.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax_heat.set_yticks(np.arange(len(ordered_regions)))
    ax_heat.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_heat.set_xticks(np.arange(len(recent_years)))
    ax_heat.set_xticklabels(recent_years)
    ax_heat.set_title("d  Reported affected population by region and year", loc="left", fontweight="bold", pad=6)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            if pd.notna(value):
                label = f"{value / 1_000_000:.1f}M" if value >= 1_000_000 else f"{value / 1_000:.0f}k"
                ax_heat.text(j, i, label, ha="center", va="center", fontsize=6.2, color="#202020")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.012)
    cbar.set_label("log10(affected + 1)")
    cbar.outline.set_visible(False)

    affected_rows = int(impact["emdat_total_affected"].notna().sum())
    death_rows = int(impact["emdat_total_deaths"].notna().sum())
    fig.text(
        0.075,
        0.985,
        f"Figure 3 | Recent flood impacts linked to ERA5 forcing ({affected_rows} affected records; {death_rows} mortality records).",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.025,
        "Impact fields are from EM-DAT public records for 2020-2024 and are matched to the same regional ERA5 event windows used in Figure 2.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    save_figure(fig, figure_dir, "figure_3_recent_impact_forcing", formats)
    plt.close(fig)


def _log_corr(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    subset = frame[[x_col, y_col]].dropna().copy()
    subset = subset[(subset[x_col] > 0) & (subset[y_col] > 0)]
    if len(subset) < 3:
        return np.nan
    x = np.log10(subset[x_col].to_numpy())
    y = np.log10(subset[y_col].to_numpy())
    return float(np.corrcoef(x, y)[0, 1])


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)
    return (values - lo) / (hi - lo)


def make_figure_4(events: pd.DataFrame, figure_dir: Path, formats: list[str]) -> None:
    events = events.copy()
    events["start_year"] = events["start_year"].astype(int)
    for column in [
        "era5_window_total_precip_mm",
        "era5_window_max_hourly_precip_mm",
        "duration_days",
        "era5_window_mean_t2m_c",
        "emdat_total_deaths",
        "emdat_total_affected",
    ]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    enriched = events[events["era5_window_total_precip_mm"].notna()].copy()
    impact = enriched[
        (enriched["event_inventory_source"] == "EM-DAT Public Table")
        & enriched["emdat_total_affected"].notna()
    ].copy()
    ordered_regions = [region for region in REGION_ORDER if region in set(enriched["region"])]
    recent_years = [2020, 2021, 2022, 2023, 2024]

    fig = plt.figure(figsize=(7.2, 7.7), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0, 1.0],
        left=0.08,
        right=0.985,
        top=0.925,
        bottom=0.135,
        hspace=0.42,
        wspace=0.32,
    )
    ax_source = fig.add_subplot(grid[0, 0])
    ax_region = fig.add_subplot(grid[0, 1])
    ax_year = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1])

    source_labels = ["GFD\n2010-2018", "EM-DAT\n2020-2024"]
    source_values = [
        enriched.loc[enriched["event_inventory_source"] == "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1", "era5_window_total_precip_mm"]
        .dropna()
        .to_numpy(),
        enriched.loc[enriched["event_inventory_source"] == "EM-DAT Public Table", "era5_window_total_precip_mm"]
        .dropna()
        .to_numpy(),
    ]
    box = ax_source.boxplot(
        source_values,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#8a8a8a", "linewidth": 0.8},
        capprops={"color": "#8a8a8a", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], [SOURCE_COLORS["GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"], SOURCE_COLORS["EM-DAT Public Table"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("none")
    ax_source.set_yscale("log")
    ax_source.set_xticks([1, 2])
    ax_source.set_xticklabels(source_labels)
    ax_source.set_ylabel("Event-window precipitation (mm)")
    ax_source.set_title("a  Source-period precipitation stability", loc="left", fontweight="bold", pad=6)
    ax_source.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax_source.spines[["top", "right"]].set_visible(False)

    region_rows = []
    for region in ordered_regions:
        subset = impact[impact["region"] == region]
        region_rows.append(
            {
                "region": region,
                "within": _log_corr(subset, "era5_window_total_precip_mm", "emdat_total_affected"),
                "without": _log_corr(impact[impact["region"] != region], "era5_window_total_precip_mm", "emdat_total_affected"),
                "n": int(len(subset)),
            }
        )
    region_df = pd.DataFrame(region_rows)
    y = np.arange(len(region_df))
    ax_region.axvline(_log_corr(impact, "era5_window_total_precip_mm", "emdat_total_affected"), color="#202020", linewidth=1.0, alpha=0.8)
    ax_region.scatter(region_df["without"], y, color="#506d84", s=48, zorder=3, label="Leave region out")
    ax_region.scatter(region_df["within"], y, color=[REGION_COLORS[r] for r in region_df["region"]], s=42, zorder=3, label="Within region")
    for yi, row in enumerate(region_df.itertuples()):
        ax_region.plot([row.without, row.within], [yi, yi], color="#c7c7c7", linewidth=1.0, zorder=1)
        ax_region.text(1.03, yi, f"n={row.n}", va="center", fontsize=6.5, color="#555555")
    ax_region.set_yticks(y)
    ax_region.set_yticklabels([REGION_SHORT[r] for r in region_df["region"]])
    ax_region.invert_yaxis()
    ax_region.set_xlim(-0.35, 1.08)
    ax_region.set_xlabel("log-log correlation")
    ax_region.set_title("b  Regional sensitivity of impact-forcing link", loc="left", fontweight="bold", pad=6)
    ax_region.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax_region.spines[["top", "right"]].set_visible(False)

    year_rows = []
    for year in recent_years:
        subset = impact[impact["start_year"] == year]
        year_rows.append(
            {
                "year": year,
                "within": _log_corr(subset, "era5_window_total_precip_mm", "emdat_total_affected"),
                "without": _log_corr(impact[impact["start_year"] != year], "era5_window_total_precip_mm", "emdat_total_affected"),
                "n": int(len(subset)),
            }
        )
    year_df = pd.DataFrame(year_rows)
    x = np.arange(len(year_df))
    width = 0.34
    ax_year.axhline(_log_corr(impact, "era5_window_total_precip_mm", "emdat_total_affected"), color="#202020", linewidth=1.0, alpha=0.8)
    ax_year.bar(x - width / 2, year_df["within"], width=width, color="#d08b39", label="Within year")
    ax_year.bar(x + width / 2, year_df["without"], width=width, color="#506d84", label="Leave year out")
    for xi, row in enumerate(year_df.itertuples()):
        ax_year.text(xi, max(row.within, row.without) + 0.04, f"n={row.n}", ha="center", fontsize=6.5, color="#555555")
    ax_year.set_xticks(x)
    ax_year.set_xticklabels(year_df["year"])
    ax_year.set_ylim(-0.15, 0.95)
    ax_year.set_ylabel("log-log correlation")
    ax_year.set_title("c  Temporal sensitivity across recent years", loc="left", fontweight="bold", pad=6)
    ax_year.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax_year.spines[["top", "right"]].set_visible(False)
    ax_year.legend(frameon=False, loc="upper left", fontsize=6.5)

    corr_frame = impact[
        [
            "era5_window_total_precip_mm",
            "era5_window_max_hourly_precip_mm",
            "duration_days",
            "era5_window_mean_t2m_c",
            "emdat_total_deaths",
            "emdat_total_affected",
        ]
    ].dropna(how="all")
    log_frame = pd.DataFrame(
        {
            "Total precip": np.log10(corr_frame["era5_window_total_precip_mm"].clip(lower=0.1)),
            "Hourly peak": np.log10(corr_frame["era5_window_max_hourly_precip_mm"].clip(lower=0.001)),
            "Duration": np.log10(corr_frame["duration_days"].clip(lower=1)),
            "Temperature": corr_frame["era5_window_mean_t2m_c"],
            "Deaths": np.log10(corr_frame["emdat_total_deaths"].clip(lower=1)),
            "Affected": np.log10(corr_frame["emdat_total_affected"].clip(lower=1)),
        }
    )
    corr = log_frame.corr(min_periods=10)
    im = ax_matrix.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = list(corr.columns)
    ax_matrix.set_xticks(np.arange(len(labels)))
    ax_matrix.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax_matrix.set_yticks(np.arange(len(labels)))
    ax_matrix.set_yticklabels(labels)
    ax_matrix.set_title("d  Cross-variable consistency matrix", loc="left", fontweight="bold", pad=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                ax_matrix.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.1, color="#202020")
    ax_matrix.tick_params(length=0)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    cbar.outline.set_visible(False)

    overall = _log_corr(impact, "era5_window_total_precip_mm", "emdat_total_affected")
    fig.text(
        0.08,
        0.985,
        f"Figure 4 | Robustness of the hydroclimate-impact relationship across sources, regions and years (overall r={overall:.2f}).",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.035,
        "Robustness diagnostics use log-transformed precipitation and EM-DAT affected-population fields. "
        "Sensitivity panels are descriptive and precede formal predictive modelling.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    save_figure(fig, figure_dir, "figure_4_robustness_sensitivity", formats)
    plt.close(fig)


def make_figure_5(events: pd.DataFrame, figure_dir: Path, formats: list[str]) -> None:
    events = events.copy()
    events["start_year"] = pd.to_numeric(events["start_year"], errors="coerce")
    events["start_month"] = pd.to_numeric(events["start_month"], errors="coerce")
    for column in [
        "era5_window_total_precip_mm",
        "era5_window_max_hourly_precip_mm",
        "duration_days",
        "emdat_total_deaths",
        "emdat_total_affected",
    ]:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    enriched = events[events["era5_window_total_precip_mm"].notna()].copy()
    impact = enriched[
        (enriched["event_inventory_source"] == "EM-DAT Public Table")
        & enriched["emdat_total_affected"].notna()
    ].copy()
    ordered_regions = [region for region in REGION_ORDER if region in set(enriched["region"])]

    rows = []
    for region in ordered_regions:
        all_region = enriched[enriched["region"] == region]
        recent_region = impact[impact["region"] == region]
        recent_n = max(len(recent_region), 1)
        rows.append(
            {
                "region": region,
                "era5_n": len(all_region),
                "gfd_n": int((all_region["event_inventory_source"] == "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").sum()),
                "emdat_n": int((all_region["event_inventory_source"] == "EM-DAT Public Table").sum()),
                "median_precip": all_region["era5_window_total_precip_mm"].median(),
                "p90_peak": all_region["era5_window_max_hourly_precip_mm"].quantile(0.90),
                "median_duration": all_region["duration_days"].median(),
                "affected_per_event": recent_region["emdat_total_affected"].sum() / recent_n,
                "deaths_per_event": recent_region["emdat_total_deaths"].sum() / recent_n,
            }
        )
    regional = pd.DataFrame(rows)
    regional["log_precip"] = np.log10(regional["median_precip"].clip(lower=0.1))
    regional["log_peak"] = np.log10(regional["p90_peak"].clip(lower=0.001))
    regional["log_duration"] = np.log10(regional["median_duration"].clip(lower=1))
    regional["log_affected"] = np.log10(regional["affected_per_event"].clip(lower=1))
    regional["log_deaths"] = np.log10(regional["deaths_per_event"].clip(lower=1))
    regional["compound_index"] = pd.concat(
        [
            _minmax(regional["log_precip"]),
            _minmax(regional["log_peak"]),
            _minmax(regional["log_duration"]),
            _minmax(regional["log_affected"]),
            _minmax(regional["log_deaths"]),
        ],
        axis=1,
    ).mean(axis=1)

    heat_cols = [
        ("Rain total", "log_precip", "median_precip", "{:.0f}"),
        ("Peak rain", "log_peak", "p90_peak", "{:.1f}"),
        ("Duration", "log_duration", "median_duration", "{:.0f}"),
        ("Affected", "log_affected", "affected_per_event", "{:.0e}"),
        ("Deaths", "log_deaths", "deaths_per_event", "{:.1f}"),
        ("Evidence", "era5_n", "era5_n", "{:.0f}"),
    ]
    heat = pd.DataFrame({label: _minmax(regional[score]) for label, score, _, _ in heat_cols})
    heat_values = heat.to_numpy()

    fig = plt.figure(figsize=(7.2, 8.6), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.1, 1.05, 0.92],
        width_ratios=[1.05, 1.0],
        left=0.085,
        right=0.985,
        top=0.92,
        bottom=0.09,
        hspace=0.46,
        wspace=0.34,
    )
    ax_heat = fig.add_subplot(grid[0:2, 0])
    ax_type = fig.add_subplot(grid[0, 1])
    ax_mosaic = fig.add_subplot(grid[1, 1])
    ax_evidence = fig.add_subplot(grid[2, 0])
    ax_rank = fig.add_subplot(grid[2, 1])

    im = ax_heat.imshow(heat_values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(heat_cols)))
    ax_heat.set_xticklabels([item[0] for item in heat_cols], rotation=38, ha="right")
    ax_heat.set_yticks(np.arange(len(regional)))
    ax_heat.set_yticklabels([REGION_SHORT[r] for r in regional["region"]])
    ax_heat.set_title("a  Regional compound flood signatures", loc="left", fontweight="bold", pad=8)
    for i, row in regional.iterrows():
        for j, (_, _, raw_col, fmt) in enumerate(heat_cols):
            value = row[raw_col]
            text = "NA" if pd.isna(value) else fmt.format(value)
            ax_heat.text(j, i, text, ha="center", va="center", fontsize=5.9, color="#202020")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("Relative regional rank")
    cbar.outline.set_visible(False)

    size = 45 + 34 * _minmax(regional["deaths_per_event"].clip(lower=0))
    ax_type.scatter(
        regional["median_precip"],
        regional["affected_per_event"].clip(lower=1),
        s=size,
        color=[REGION_COLORS[r] for r in regional["region"]],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for row in regional.itertuples():
        ax_type.text(row.median_precip * 1.03, max(row.affected_per_event, 1) * 1.03, REGION_SHORT[row.region], fontsize=6.2)
    ax_type.axvline(regional["median_precip"].median(), color="#777777", linewidth=0.8, linestyle="--")
    ax_type.axhline(regional["affected_per_event"].clip(lower=1).median(), color="#777777", linewidth=0.8, linestyle="--")
    ax_type.set_xscale("log")
    ax_type.set_yscale("log")
    ax_type.set_xlim(38, 470)
    ax_type.set_xticks([50, 100, 200, 400])
    ax_type.set_xticklabels(["50", "100", "200", "400"])
    ax_type.set_xlabel("Median event precipitation (mm)")
    ax_type.set_ylabel("Affected people per event")
    ax_type.set_title("b  Hazard-impact typology", loc="left", fontweight="bold", pad=8)
    ax_type.grid(color="#e8e8e8", linewidth=0.6)
    ax_type.spines[["top", "right"]].set_visible(False)

    mosaic = impact.sort_values("emdat_total_affected", ascending=False).head(80).copy()
    mosaic["x"] = mosaic["start_year"] + (mosaic["start_month"].fillna(6).clip(1, 12) - 0.5) / 12
    region_to_y = {region: idx for idx, region in enumerate(ordered_regions)}
    mosaic["y"] = mosaic["region"].map(region_to_y)
    sc = ax_mosaic.scatter(
        mosaic["x"],
        mosaic["y"],
        s=18 + 42 * _minmax(np.log10(mosaic["emdat_total_affected"].clip(lower=1))),
        c=mosaic["era5_window_total_precip_mm"],
        cmap="PuBuGn",
        edgecolor="#ffffff",
        linewidth=0.35,
        alpha=0.92,
    )
    ax_mosaic.set_yticks(np.arange(len(ordered_regions)))
    ax_mosaic.set_yticklabels([REGION_SHORT[r] for r in ordered_regions])
    ax_mosaic.invert_yaxis()
    ax_mosaic.set_xlim(2019.85, 2025.05)
    ax_mosaic.set_xticks([2020, 2021, 2022, 2023, 2024])
    ax_mosaic.set_title("c  Recent high-burden event mosaic", loc="left", fontweight="bold", pad=8)
    ax_mosaic.set_xlabel("Event start year")
    ax_mosaic.grid(axis="x", color="#e8e8e8", linewidth=0.6)
    ax_mosaic.spines[["top", "right"]].set_visible(False)
    cbar_mosaic = fig.colorbar(sc, ax=ax_mosaic, fraction=0.045, pad=0.02)
    cbar_mosaic.set_label("Rain total (mm)")
    cbar_mosaic.outline.set_visible(False)

    y = np.arange(len(regional))
    ax_evidence.barh(y, regional["gfd_n"], color=SOURCE_COLORS["GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"], label="GFD")
    ax_evidence.barh(
        y,
        regional["emdat_n"],
        left=regional["gfd_n"],
        color=SOURCE_COLORS["EM-DAT Public Table"],
        label="EM-DAT",
    )
    for yi, row in enumerate(regional.itertuples()):
        ax_evidence.text(row.gfd_n + row.emdat_n + 1, yi, f"ERA5 n={row.era5_n}", va="center", fontsize=6.2, color="#555555")
    ax_evidence.set_yticks(y)
    ax_evidence.set_yticklabels([REGION_SHORT[r] for r in regional["region"]])
    ax_evidence.invert_yaxis()
    ax_evidence.set_xlabel("Event-window records")
    ax_evidence.set_title("d  Evidence support behind the synthesis", loc="left", fontweight="bold", pad=8)
    ax_evidence.grid(axis="x", color="#e8e8e8", linewidth=0.6)
    ax_evidence.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=2)
    ax_evidence.spines[["top", "right"]].set_visible(False)

    ranked = regional.sort_values("compound_index", ascending=True).copy()
    yrank = np.arange(len(ranked))
    ax_rank.barh(
        yrank,
        ranked["compound_index"],
        color=[REGION_COLORS[r] for r in ranked["region"]],
        height=0.58,
    )
    for yi, row in enumerate(ranked.itertuples()):
        ax_rank.text(row.compound_index + 0.015, yi, f"{row.compound_index:.2f}", va="center", fontsize=6.3)
    ax_rank.set_yticks(yrank)
    ax_rank.set_yticklabels([REGION_SHORT[r] for r in ranked["region"]])
    ax_rank.set_xlim(0, 1.08)
    ax_rank.set_xlabel("Composite hazard-impact index")
    ax_rank.set_title("e  Regional synthesis ranking", loc="left", fontweight="bold", pad=8)
    ax_rank.grid(axis="x", color="#e8e8e8", linewidth=0.6)
    ax_rank.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.085,
        0.985,
        "Figure 5 | Compound flood-burden typology integrating event forcing, impacts and evidence support.",
        ha="left",
        va="top",
        fontsize=9.3,
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.025,
        "The synthesis index combines normalized regional precipitation, peak intensity, duration, affected population and mortality. "
        "It is a descriptive prioritization tool, not a causal estimate.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    save_figure(fig, figure_dir, "figure_5_compound_flood_burden_typology", formats)
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
        + [str(figure_dir / f"figure_2_era5_hydroclimate_fingerprints.{fmt}") for fmt in formats]
        + [str(figure_dir / f"figure_3_recent_impact_forcing.{fmt}") for fmt in formats]
        + [str(figure_dir / f"figure_4_robustness_sensitivity.{fmt}") for fmt in formats]
        + [str(figure_dir / f"figure_5_compound_flood_burden_typology.{fmt}") for fmt in formats],
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
    make_figure_3(events, figure_dir, args.formats)
    make_figure_4(events, figure_dir, args.formats)
    make_figure_5(events, figure_dir, args.formats)
    write_figure_manifest(data_root, figure_dir, args.formats)
    print(f"Wrote Figure 1, Figure 2, Figure 3, Figure 4, and Figure 5 drafts to {figure_dir}")


if __name__ == "__main__":
    main()
