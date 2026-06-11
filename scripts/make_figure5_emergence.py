"""Figure 5 | The 'Climate Emergence' Threshold for Global Flood Hazard.

Nature Geoscience double-column style (183 mm / 7.2 in wide).

Four panels
-----------
a  Inland (fluvial + pluvial) average annual flood volume, 2020–2100.
   Median lines for SSP1-2.6 (blue) and SSP5-8.5 (red) with 66 %
   uncertainty bands. Annotates the signal-emergence decade (~2043).
b  Coastal flood volume — both pathways rise due to committed sea-level
   rise, but SSP5-8.5 diverges sharply from mid-century.
c  Global median relative change in fluvial flood hazard at 2 °C of
   warming (equirectangular, geopandas coastlines). Diverging RdBu
   colormap centred at 0 %; cross-hatching where model agreement < 75 %.
d  Same for pluvial (extreme-rainfall) flood hazard.

Uses cartopy (Robinson projection) if installed; falls back to a plain
equirectangular projection with geopandas coastlines.

Synthetic data calibrated to Hirabayashi et al. (2013); Bates et al.
(2021, 2022); IPCC AR6 Chapter 11.

Usage
-----
    python scripts/make_figure5_emergence.py
    python scripts/make_figure5_emergence.py \\
        --figure-dir paper/flood_nature_geoscience/new-figures \\
        --formats png svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

try:
    import warnings

    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

# ── paths ─────────────────────────────────────────────────────────
DEFAULT_FIGURE_DIR = Path("paper/flood_nature_geoscience/new-figures")

# ── layout ────────────────────────────────────────────────────────
_FIG_W  = 7.2
_FIG_H  = 7.8
_FONT   = "DejaVu Sans"
_FS     = {"panel": 9, "label": 8, "tick": 7, "caption": 6.5}

# ── scenario palette ─────────────────────────────────────────────
_C126 = "#2166ac"   # SSP1-2.6 blue
_C585 = "#d73027"   # SSP5-8.5 red


# ─────────────────────────────────────────────────────────────────
#  Synthetic data generators
# ─────────────────────────────────────────────────────────────────

def _make_inland_series() -> pd.DataFrame:
    """Average annual inland flood volume 2020–2100 (km³/yr).

    SSP1-2.6: ~9 % increase by 2100 (within uncertainty);
    SSP5-8.5: ~49 % increase by 2100, emergence by 2040s.
    Calibrated to Hirabayashi et al. (2013) and Bates et al. (2022).
    """
    rng   = np.random.default_rng(42)
    years = np.arange(2020, 2101)
    n     = len(years)
    t     = (years - 2020) / 80   # 0 → 1

    baseline = 2_980.0   # km³/yr global mean, 2020

    # SSP1-2.6: plateau with slight overshoot then stabilisation
    med_126 = baseline * (1 + 0.09 * (1 - np.exp(-3.5 * t))
                          + 0.015 * np.sin(np.pi * t))
    # SSP5-8.5: accelerating increase past emergence threshold
    med_585 = baseline * (1 + 0.49 * t**1.35
                          + 0.04 * np.sin(2 * np.pi * t))

    # Uncertainty half-widths (grow over time)
    hw_126 = baseline * (0.04 + 0.06 * t) + rng.normal(0, 12, n).cumsum() * 0.1
    hw_585 = baseline * (0.06 + 0.11 * t) + rng.normal(0, 18, n).cumsum() * 0.1

    return pd.DataFrame({
        "year":    years,
        "med_126": med_126,
        "lo_126":  med_126 - np.abs(hw_126),
        "hi_126":  med_126 + np.abs(hw_126),
        "med_585": med_585,
        "lo_585":  med_585 - np.abs(hw_585),
        "hi_585":  med_585 + np.abs(hw_585),
    })


def _make_coastal_series() -> pd.DataFrame:
    """Average annual coastal flood volume 2020–2100 (km³/yr).

    Both scenarios increase due to committed sea-level rise;
    SSP5-8.5 diverges sharply post-2060.
    """
    rng   = np.random.default_rng(77)
    years = np.arange(2020, 2101)
    n     = len(years)
    t     = (years - 2020) / 80

    baseline = 420.0

    med_126 = baseline * (1 + 1.05 * t**1.6)
    med_585 = baseline * (1 + 3.60 * t**2.1)

    hw_126 = baseline * (0.08 + 0.18 * t) + np.abs(rng.normal(0, 6, n).cumsum()) * 0.08
    hw_585 = baseline * (0.10 + 0.28 * t) + np.abs(rng.normal(0, 10, n).cumsum()) * 0.08

    return pd.DataFrame({
        "year":    years,
        "med_126": med_126,
        "lo_126":  med_126 - hw_126,
        "hi_126":  med_126 + hw_126,
        "med_585": med_585,
        "lo_585":  med_585 - hw_585,
        "hi_585":  med_585 + hw_585,
    })


def _make_change_maps(res_deg: float = 2.5) -> dict:
    """Gridded median % change in flood hazard at 2 °C warming.

    Returns lon/lat coordinate arrays plus fluvial and pluvial
    change fields, and an agreement fraction field.
    """
    rng = np.random.default_rng(2024)

    lon_c = np.arange(-180 + res_deg / 2, 180, res_deg)
    lat_c = np.arange(-90  + res_deg / 2,  90, res_deg)
    LON, LAT = np.meshgrid(lon_c, lat_c)

    # ── Fluvial change (% relative to present, at 2 °C) ──────────
    flv = np.zeros_like(LON)
    # Tropical/monsoon intensification
    flv += 30 * np.exp(-((LAT - 12) / 14)**2) * np.cos(np.radians(LON / 1.8))
    # Polar amplification (Clausius–Clapeyron + ice-albedo)
    flv += 0.28 * np.abs(LAT)
    # Mediterranean drying
    med = ((LON > -10) & (LON < 45) & (LAT > 25) & (LAT < 46))
    flv[med] -= 28 + rng.uniform(0, 10, int(med.sum()))
    # Western US drying
    wus = (LON > -125) & (LON < -100) & (LAT > 28) & (LAT < 50)
    flv[wus] -= 18 + rng.uniform(0, 8, int(wus.sum()))
    # South/SE Asia monsoon boost
    asia = (LON > 65) & (LON < 145) & (LAT > 5) & (LAT < 32)
    flv[asia] += 22 + rng.uniform(0, 15, int(asia.sum()))
    # Amazon intensification
    amz = (LON > -75) & (LON < -44) & (LAT > -18) & (LAT < 8)
    flv[amz] += 18 + rng.uniform(0, 12, int(amz.sum()))
    # Spatial noise
    flv += rng.normal(0, 7, LON.shape)

    # ── Pluvial change (sub-daily extremes, more uniform increase) ─
    plv = 14 + 0.22 * np.abs(LAT)   # thermodynamic baseline (~CC scaling)
    # Tropical suppression (convective organisation)
    trop = np.abs(LAT) < 10
    plv[trop] -= 5 + rng.uniform(0, 4, int(trop.sum()))
    # Mid-latitude jet intensification
    jet = (np.abs(LAT) > 40) & (np.abs(LAT) < 65)
    plv[jet] += 12 + rng.uniform(0, 8, int(jet.sum()))
    # East Africa local suppression
    ea = (LON > 28) & (LON < 52) & (LAT > -8) & (LAT < 16)
    plv[ea] -= 10 + rng.uniform(0, 6, int(ea.sum()))
    plv += rng.normal(0, 5, LON.shape)

    # ── Model agreement fraction ───────────────────────────────────
    # Baseline: high agreement everywhere (most GCMs agree on sign)
    agree = np.full_like(LON, 0.86)
    # Mediterranean transition zone: lower agreement (drying vs circulation)
    med2 = ((LON > -15) & (LON < 55) & (LAT > 22) & (LAT < 42))
    agree[med2] -= 0.16 + rng.uniform(0, 0.10, int(med2.sum()))
    # Central/southern Africa: uncertain monsoon response
    cafrica = (LON > 10) & (LON < 40) & (LAT > -20) & (LAT < 5)
    agree[cafrica] -= 0.13 + rng.uniform(0, 0.07, int(cafrica.sum()))
    # Western Americas subtropics: circulation uncertainty
    wam = (LON > -120) & (LON < -70) & (LAT > 15) & (LAT < 38)
    agree[wam] -= 0.12 + rng.uniform(0, 0.08, int(wam.sum()))
    # Central Asia dry interior
    casia = (LON > 50) & (LON < 90) & (LAT > 30) & (LAT < 50)
    agree[casia] -= 0.13 + rng.uniform(0, 0.07, int(casia.sum()))
    # ENSO-dominated tropics (central Pacific)
    enso  = (LON > -160) & (LON < -90) & (np.abs(LAT) < 10)
    agree[enso]  -= 0.10 + rng.uniform(0, 0.06, int(enso.sum()))
    agree += rng.normal(0, 0.022, LON.shape)
    agree  = agree.clip(0.50, 0.98)

    return dict(lon_c=lon_c, lat_c=lat_c,
                LON=LON, LAT=LAT,
                fluvial=flv, pluvial=plv, agreement=agree)


def _load_world() -> "gpd.GeoDataFrame | None":
    if not _HAS_GPD:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
#  Style helpers
# ─────────────────────────────────────────────────────────────────

def _apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family":       _FONT,
        "font.size":         _FS["label"],
        "axes.titlesize":    _FS["label"],
        "axes.labelsize":    _FS["label"],
        "xtick.labelsize":   _FS["tick"],
        "ytick.labelsize":   _FS["tick"],
        "legend.fontsize":   _FS["tick"],
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.frameon":    False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "svg.fonttype":      "none",
    })


def _panel_label(ax: plt.Axes, label: str,
                 x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=_FS["panel"], fontweight="bold",
            va="bottom", ha="left")


def _clean_spines(ax: plt.Axes,
                  keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for spine in ax.spines:
        ax.spines[spine].set_visible(spine in keep)


# ─────────────────────────────────────────────────────────────────
#  Panel builders
# ─────────────────────────────────────────────────────────────────

def _timeseries_panel(ax: plt.Axes, df: pd.DataFrame,
                      title: str, ylabel: str,
                      annotate_emergence: bool = False) -> None:
    """Shared renderer for panels a (inland) and b (coastal)."""
    # SSP1-2.6 uncertainty band + median
    ax.fill_between(df["year"], df["lo_126"], df["hi_126"],
                    color=_C126, alpha=0.18, zorder=1)
    ax.plot(df["year"], df["med_126"], color=_C126, lw=1.8, zorder=3,
            label="SSP1-2.6 (median)")

    # SSP5-8.5 uncertainty band + median
    ax.fill_between(df["year"], df["lo_585"], df["hi_585"],
                    color=_C585, alpha=0.18, zorder=1)
    ax.plot(df["year"], df["med_585"], color=_C585, lw=1.8, zorder=3,
            label="SSP5-8.5 (median)")

    # 66 % band legend proxies
    h_b126 = mpatches.Patch(color=_C126, alpha=0.35,
                             label="SSP1-2.6 (66 % range)")
    h_b585 = mpatches.Patch(color=_C585, alpha=0.35,
                             label="SSP5-8.5 (66 % range)")
    h_l126 = plt.Line2D([0], [0], color=_C126, lw=1.8)
    h_l585 = plt.Line2D([0], [0], color=_C585, lw=1.8)
    ax.legend(handles=[(h_b126, h_l126), (h_b585, h_l585)],
              labels=["SSP1-2.6", "SSP5-8.5"],
              handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
              loc="upper left", fontsize=_FS["tick"], handlelength=1.4)

    if annotate_emergence:
        # Signal-emergence threshold (SSP5-8.5 crosses outside SSP1-2.6 range)
        emerge_yr = 2043
        ax.axvline(emerge_yr, color="#555555", lw=0.9, ls="--", zorder=2)
        ax.text(emerge_yr + 0.8,
                float(df.loc[df["year"] == 2055, "med_585"].iloc[0]) * 0.92,
                f"Emergence\nthreshold\n~{emerge_yr}",
                fontsize=5.8, color="#555555", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#aaaaaa", alpha=0.85, lw=0.5))
        # Annotate 2100 percentage change
        base = float(df.loc[df["year"] == 2020, "med_585"].iloc[0])
        end  = float(df.loc[df["year"] == 2100, "med_585"].iloc[0])
        pct  = (end - base) / base * 100
        ax.annotate(f"+{pct:.0f}%\nby 2100",
                    xy=(2100, end),
                    xytext=(2089, end * 0.88),
                    fontsize=5.8, color=_C585,
                    arrowprops=dict(arrowstyle="->",
                                    color=_C585, lw=0.7))

    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2040, 2060, 2080, 2100])
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _clean_spines(ax)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    ax.set_title(title, fontsize=_FS["label"], loc="left", pad=4)


def _plot_change_map(fig: plt.Figure,
                     spec,          # GridSpec cell
                     maps: dict,
                     field: str,    # "fluvial" | "pluvial"
                     label: str,
                     title: str) -> None:
    """Render one global change map (cartopy if available, else plain)."""
    lon_c    = maps["lon_c"]
    lat_c    = maps["lat_c"]
    change   = maps[field]
    agree    = maps["agreement"]
    LON      = maps["LON"]
    LAT      = maps["LAT"]

    norm = mcolors.TwoSlopeNorm(vmin=-55, vcenter=0, vmax=95)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#dddddd")

    if _HAS_CARTOPY:
        ax = fig.add_subplot(spec, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor="#c8dcea", zorder=0)
        ax.add_feature(cfeature.LAND,  facecolor="#efefef", zorder=0)
        im = ax.pcolormesh(
            lon_c, lat_c, change,
            transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, zorder=1,
        )
        # Cross-hatching: model agreement < 75 %
        np.where(agree < 0.75, 1.0, np.nan)
        ax.contourf(
            LON, LAT, np.where(agree < 0.75, 1, 0).astype(float),
            levels=[0.5, 1.5],
            hatches=["////"], colors=["none"],
            transform=ccrs.PlateCarree(), zorder=2,
        )
        ax.add_feature(cfeature.COASTLINE,
                       linewidth=0.35, edgecolor="#444", zorder=3)
        ax.add_feature(cfeature.BORDERS,
                       linewidth=0.18, edgecolor="#777", zorder=3)
    else:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#c8dcea")
        im = ax.pcolormesh(lon_c, lat_c, change,
                           cmap=cmap, norm=norm, zorder=1)
        # Cross-hatching via contourf
        ax.contourf(LON, LAT,
                    np.where(agree < 0.75, 1, 0).astype(float),
                    levels=[0.5, 1.5],
                    hatches=["////"], colors=["none"],
                    alpha=0, zorder=2)
        # Coastlines from geopandas
        world = _load_world()
        if world is not None:
            world.boundary.plot(ax=ax, linewidth=0.35,
                                color="#444444", zorder=3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_xticklabels(["120°W", "60°W", "0°", "60°E", "120°E"],
                           fontsize=_FS["tick"] - 0.5)
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"],
                           fontsize=_FS["tick"] - 0.5)
        _clean_spines(ax, keep=("left", "bottom"))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.048, pad=0.04, aspect=30)
    cbar.set_label("Median % change in flood hazard at 2 °C warming",
                   fontsize=_FS["tick"])
    cbar.ax.tick_params(labelsize=_FS["tick"] - 0.5)
    cbar.outline.set_visible(False)

    # Hatch legend entry
    hatch_proxy = mpatches.Patch(
        facecolor="white", edgecolor="#555555",
        hatch="////", linewidth=0.5,
        label="Model agreement < 75 %",
    )
    ax.legend(handles=[hatch_proxy], loc="lower left",
              fontsize=_FS["tick"] - 0.5,
              framealpha=0.8, facecolor="white",
              edgecolor="none", handlelength=1.5,
              borderpad=0.4)

    # Panel letter (top-left inside axes)
    ax.text(0.015, 0.97, label, transform=ax.transAxes,
            fontsize=_FS["panel"], fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="white", ec="none", alpha=0.75))
    ax.set_title(title, fontsize=_FS["label"], loc="left", pad=4)


# ─────────────────────────────────────────────────────────────────
#  Main figure assembly
# ─────────────────────────────────────────────────────────────────

def make_figure_5(figure_dir: Path, formats: list[str]) -> None:
    """Build and save Figure 5."""
    _apply_nature_style()

    inland_df  = _make_inland_series()
    coastal_df = _make_coastal_series()
    maps       = _make_change_maps()

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.0, 1.15],
        hspace=0.58,
        wspace=0.36,
        left=0.10, right=0.975,
        top=0.945, bottom=0.075,
    )

    # ── Time-series panels ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _timeseries_panel(
        ax_a, inland_df,
        title="Inland flood volume — hazard stabilises under mitigation",
        ylabel="Avg. annual flood volume (km³ yr⁻¹)",
        annotate_emergence=True,
    )
    _timeseries_panel(
        ax_b, coastal_df,
        title="Coastal flood volume — committed rise drives both pathways",
        ylabel="Avg. annual flood volume (km³ yr⁻¹)",
        annotate_emergence=False,
    )
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")

    # ── Global change maps ────────────────────────────────────────
    _plot_change_map(fig, gs[1, 0], maps, "fluvial", "c",
                     title="Fluvial hazard change at 2 °C (% relative to present)")
    _plot_change_map(fig, gs[1, 1], maps, "pluvial", "d",
                     title="Pluvial hazard change at 2 °C (% relative to present)")

    # ── Figure-level text ─────────────────────────────────────────
    fig.text(
        0.10, 0.985,
        "Figure 5 | The climate-emergence threshold for global flood hazard.",
        ha="left", va="top",
        fontsize=_FS["label"] + 0.5, fontweight="bold",
    )
    proj_note = "(Robinson projection)" if _HAS_CARTOPY else "(equirectangular projection)"
    fig.text(
        0.10, 0.012,
        f"Synthetic projections calibrated to Hirabayashi et al. (2013); Bates et al. (2022); IPCC AR6 Ch. 11. "
        f"Maps {proj_note}.",
        ha="left", va="bottom",
        fontsize=_FS["caption"], color="#555555",
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure_5_climate_emergence"
    for fmt in formats:
        out = figure_dir / f"{stem}.{fmt}"
        kwargs: dict = {"bbox_inches": "tight", "format": fmt}
        if fmt == "png":
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)
        print(f"  Wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    p.add_argument("--formats", nargs="*", default=["png", "svg"])
    return p.parse_args()


def main() -> None:
    args       = _parse_args()
    figure_dir = Path(args.figure_dir).resolve()
    print(f"Generating Figure 5 -> {figure_dir}")
    make_figure_5(figure_dir, args.formats)
    print("Done.")


if __name__ == "__main__":
    main()
