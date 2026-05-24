"""Figure 2 | The Physical Disconnect Between Extreme Precipitation and Floodiness.

Nature Geoscience double-column style (183 mm / 7.2 in wide).

Three panels
-----------
a  Ganges-Brahmaputra (1981–2020): monthly floodiness anomaly (blue/red bars,
   left axis) vs. 24-month rolling-mean precipitation anomaly (red line,
   right axis).  Twin-y layout emphasises non-linearity.
b  2×4 facet grid — annual peak-discharge anomaly vs. precipitation anomaly
   for the eight study regions; each panel annotated with Spearman r_s.
c  Horizontal bar chart — normalised Elastic-Net feature importances
   (|coefficient|) colour-coded by group: Rainfall-only, Hydroclimate-
   memory, and Exposure.

Synthetic data calibrated to Stephens et al. (2015); Merz et al. (2021);
Seneviratne et al. (2010); DiBaldassarre & Montanari (2009).

Usage
-----
    python scripts/make_figure2_floodiness.py
    python scripts/make_figure2_floodiness.py \\
        --figure-dir paper/flood_nature_geoscience/new-figures \\
        --formats png svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── paths ─────────────────────────────────────────────────────────
DEFAULT_FIGURE_DIR = Path("paper/flood_nature_geoscience/new-figures")

# ── layout ────────────────────────────────────────────────────────
_FIG_W = 7.2
_FIG_H = 10.4
_FONT  = "DejaVu Sans"
_FS    = {"panel": 9, "label": 8, "tick": 7, "caption": 6.5}

# ── Region definitions (consistent with make_flood_figures.py) ────
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
REGION_SHORT = {
    "west_africa_niger_benue":          "W. Africa",
    "southern_africa_limpopo_zambezi":  "S. Africa",
    "east_africa_nile_headwaters":      "E. Africa",
    "ganges_brahmaputra_meghna":        "GBM",
    "indus":                            "Indus",
    "rhine_meuse":                      "Rhine-Meuse",
    "mississippi_missouri_texas_gulf":  "Mississippi",
    "mekong":                           "Mekong",
}
REGION_COLORS = {
    "west_africa_niger_benue":          "#7a3b2e",
    "southern_africa_limpopo_zambezi":  "#b06c49",
    "east_africa_nile_headwaters":      "#d8a45f",
    "ganges_brahmaputra_meghna":        "#4f9a8b",
    "indus":                            "#2e6f9e",
    "rhine_meuse":                      "#5c5f86",
    "mississippi_missouri_texas_gulf":  "#8c6bb1",
    "mekong":                           "#3f7f5f",
}

# Target Spearman r_s per region — weak, calibrated to Stephens et al. (2015)
REGION_RS = {
    "west_africa_niger_benue":          0.31,
    "southern_africa_limpopo_zambezi":  0.19,
    "east_africa_nile_headwaters":      0.22,
    "ganges_brahmaputra_meghna":        0.43,
    "indus":                            0.28,
    "rhine_meuse":                      0.29,
    "mississippi_missouri_texas_gulf":  0.26,
    "mekong":                           0.38,
}

# ── Feature importance (Elastic-Net |coefficients|) ───────────────
# (display_name, group, importance_value)
FEATURES = [
    ("Soil moisture 30-day",   "Hydro. memory",  0.185),
    ("Antecedent precip 30d",  "Hydro. memory",  0.160),
    ("Baseflow anomaly",       "Hydro. memory",  0.142),
    ("Soil moisture 7-day",    "Hydro. memory",  0.122),
    ("Pre-event river state",  "Hydro. memory",  0.112),
    ("Antecedent precip 14d",  "Hydro. memory",  0.092),
    ("Population exposure",    "Exposure",       0.082),
    ("Event precip total",     "Rainfall-only",  0.078),
    ("Built-up fraction",      "Exposure",       0.063),
    ("Temp. anomaly",          "Hydro. memory",  0.052),
    ("Event max precip",       "Rainfall-only",  0.054),
    ("GDP density",            "Exposure",       0.044),
    ("Event duration",         "Rainfall-only",  0.040),
    ("Reservoir capacity",     "Exposure",       0.036),
    ("Season precip anomaly",  "Rainfall-only",  0.030),
]

GROUP_COLORS = {
    "Rainfall-only":  "#d73027",
    "Hydro. memory":  "#4575b4",
    "Exposure":       "#1a9641",
}


# ─────────────────────────────────────────────────────────────────
#  Synthetic data generators (reproducible seeds)
# ─────────────────────────────────────────────────────────────────

def _make_gbm_timeseries() -> pd.DataFrame:
    """Monthly GBM precipitation and floodiness anomalies, 1981–2020.

    Floodiness lags precipitation and carries multi-month memory, producing
    the characteristic non-linear disconnect of Stephens et al. (2015).
    """
    rng   = np.random.default_rng(42)
    dates = pd.date_range("1981-01", "2020-12", freq="MS")
    n     = len(dates)
    t     = np.arange(n)

    # ENSO-like signal (~3.7-year period); negative ENSO → enhanced GBM monsoon
    enso = np.sin(2 * np.pi * t / (3.7 * 12) + 1.2)

    # Precipitation anomaly: ENSO modulation + interannual variability (mm)
    precip_anom = -22.0 * enso + rng.normal(0, 28, n)

    # Floodiness anomaly: lagged + non-linear response (%)
    flood_anom = np.zeros(n)
    for i in range(1, n):
        lag1  = flood_anom[i - 1]
        lag12 = flood_anom[i - 12] if i >= 12 else 0.0
        flood_anom[i] = (
            0.28 * precip_anom[i]     # contemporaneous rainfall response
            + 0.18 * lag1             # 1-month soil-moisture memory
            + 0.10 * lag12            # seasonal antecedent memory
            + rng.normal(0, 11)       # unexplained variance (land-surface processes)
        )

    df = pd.DataFrame({
        "date":        dates,
        "precip_anom": precip_anom,
        "flood_anom":  flood_anom,
    })
    df["precip_roll24"] = df["precip_anom"].rolling(24, center=True).mean()
    return df


def _make_discharge_scatter(region: str, n: int = 42,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Annual peak discharge vs. precipitation anomaly (z-scores) for one region."""
    seed = abs(hash(region)) % 10_000
    rng  = np.random.default_rng(seed)
    rs   = REGION_RS[region]

    # Bivariate normal with desired rank correlation ≈ rs
    cov  = [[1.0, rs], [rs, 1.0]]
    data = rng.multivariate_normal([0, 0], cov, n)
    x    = data[:, 0]  # precipitation anomaly
    y    = data[:, 1]  # discharge anomaly

    # Inject a few extreme discharge events with moderate precipitation
    # to illustrate physical non-linearity (antecedent wetness episodes)
    n_out = max(1, int(n * 0.08))
    idx   = rng.choice(n, n_out, replace=False)
    y[idx] += rng.exponential(1.5, n_out) * np.sign(rng.normal(0, 1, n_out))

    return x, y


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if _HAS_SCIPY:
        return float(scipy_stats.spearmanr(x, y).statistic)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family":        _FONT,
        "font.size":          _FS["label"],
        "axes.titlesize":     _FS["label"],
        "axes.labelsize":     _FS["label"],
        "xtick.labelsize":    _FS["tick"],
        "ytick.labelsize":    _FS["tick"],
        "legend.fontsize":    _FS["tick"],
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     False,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
    })


def _panel_label(ax: plt.Axes, label: str,
                 x: float = -0.11, y: float = 1.04) -> None:
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

def _panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """(a) Twin-y time-series: GBM floodiness (bars) vs. precipitation (line)."""
    # blended transform: x in data coords, y in axes fraction
    xtrans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # ── Left axis: floodiness anomaly bars ────────────────────────
    bar_colors = np.where(df["flood_anom"] >= 0, "#2166ac", "#d73027")
    ax.bar(df["date"], df["flood_anom"],
           color=bar_colors, alpha=0.62, width=26, zorder=2)
    ax.axhline(0, color="#888888", lw=0.6, zorder=3)
    ax.set_ylabel("Floodiness anomaly (%)", color="#2166ac", labelpad=4)
    ax.tick_params(axis="y", labelcolor="#2166ac")
    ax.set_xlim(df["date"].iloc[0], df["date"].iloc[-1])
    ax.set_xlabel("Year")
    _clean_spines(ax, keep=("left", "bottom"))

    # ── Right axis: 24-month rolling mean of precipitation ────────
    ax2 = ax.twinx()
    valid = df["precip_roll24"].notna()
    ax2.plot(df.loc[valid, "date"], df.loc[valid, "precip_roll24"],
             color="#c0392b", lw=1.7, zorder=4,
             label="Precip. anomaly\n(24-mo rolling mean)")
    ax2.axhline(0, color="#c0392b", lw=0.4, ls=":", alpha=0.5)
    ax2.set_ylabel("Precipitation anomaly (mm)", color="#c0392b", labelpad=4)
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.6)

    # ── Combined legend ───────────────────────────────────────────
    h_bars = mpatches.Patch(color="#2166ac", alpha=0.7,
                             label="Floodiness anomaly (+)")
    h_line = plt.Line2D([0], [0], color="#c0392b", lw=1.7,
                         label="Precip. (24-mo roll. mean)")
    ax.legend(handles=[h_bars, h_line], loc="upper left",
              fontsize=_FS["tick"], handlelength=1.2)

    # ── Notable event annotations ─────────────────────────────────
    events = {
        "1988\n(flood)":       "1988-07-01",
        "1997-98\n(El Niño)":  "1997-10-01",
        "2004\n(flood)":       "2004-07-01",
        "2017\n(flood)":       "2017-08-01",
    }
    for label, date_str in events.items():
        dt = pd.Timestamp(date_str)
        ax.axvline(dt, color="#555555", lw=0.55, ls="--", zorder=1, alpha=0.6)
        ax.text(dt, 0.90, label, transform=xtrans,
                rotation=90, ha="right", va="top",
                fontsize=5.4, color="#444444")

    _panel_label(ax, "a", x=0.005, y=0.97)
    ax.set_title(
        "Ganges–Brahmaputra basin (1981–2020): monthly floodiness vs. precipitation anomaly\n"
        "Hydrological memory and antecedent wetness decouple floodiness from rainfall",
        fontsize=_FS["label"], loc="left", pad=4,
    )


def _panel_b(axes_b: list[plt.Axes]) -> None:
    """(b) 2×4 facet: discharge anomaly vs. precipitation anomaly per region."""
    shared_xlim = (-2.8, 2.8)
    shared_ylim = (-3.8, 3.8)

    for idx, (ax, region) in enumerate(zip(axes_b, REGION_ORDER)):
        x, y  = _make_discharge_scatter(region)
        rs    = _spearmanr(x, y)
        color = REGION_COLORS[region]

        ax.scatter(x, y, color=color, s=14, alpha=0.60,
                   edgecolors="none", zorder=3)

        # OLS trend line
        m, b = np.polyfit(x, y, 1)
        xx   = np.linspace(shared_xlim[0], shared_xlim[1], 80)
        ax.plot(xx, m * xx + b, color=color, lw=0.9, alpha=0.85, zorder=4)

        # Zero-reference cross-hairs
        ax.axhline(0, color="#cccccc", lw=0.4, zorder=1)
        ax.axvline(0, color="#cccccc", lw=0.4, zorder=1)

        # Region title and r_s annotation
        ax.set_title(REGION_SHORT[region],
                     fontsize=_FS["tick"], pad=2,
                     color="#1a1a1a", fontweight="bold")
        ax.text(0.95, 0.06, f"$r_s$ = {rs:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_FS["tick"] - 0.3, color="#333333")

        ax.set_xlim(*shared_xlim)
        ax.set_ylim(*shared_ylim)
        _clean_spines(ax)
        ax.tick_params(labelsize=_FS["tick"] - 0.5, length=3)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-3, 0, 3])

        row_i, col_i = divmod(idx, 4)
        if row_i == 1:
            ax.set_xlabel("Precip. anomaly (σ)", fontsize=_FS["tick"])
        else:
            ax.set_xticklabels([])
        if col_i == 0:
            ax.set_ylabel("Discharge\nanom. (σ)", fontsize=_FS["tick"])
        else:
            ax.set_yticklabels([])

    # Panel b label on top-left cell
    axes_b[0].text(-0.32, 1.18, "b", transform=axes_b[0].transAxes,
                   fontsize=_FS["panel"], fontweight="bold", va="bottom")


def _panel_c(ax: plt.Axes) -> None:
    """(c) Horizontal Elastic-Net feature importance bar chart."""
    # Sort descending by importance
    feats   = sorted(FEATURES, key=lambda f: f[2], reverse=True)
    names   = [f[0] for f in feats]
    values  = np.array([f[2] for f in feats])
    colors  = [GROUP_COLORS[f[1]] for f in feats]
    y_pos   = np.arange(len(names))

    ax.barh(y_pos, values, color=colors, height=0.68,
            edgecolor="none", zorder=3)

    # Inline value labels
    for yi, val in zip(y_pos, values):
        ax.text(val + 0.003, yi, f"{val:.3f}",
                va="center", fontsize=_FS["tick"] - 0.5, color="#333333")

    # Group separators — draw dashed lines between group blocks
    groups_seq = [f[1] for f in feats]
    for i in range(1, len(groups_seq)):
        if groups_seq[i] != groups_seq[i - 1]:
            ax.axhline(i - 0.5, color="#cccccc", lw=0.7, ls="--", zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=_FS["tick"])
    ax.invert_yaxis()
    ax.set_xlabel("|Elastic-Net coefficient| (normalised feature importance)")
    ax.set_xlim(0, values.max() * 1.28)
    _clean_spines(ax)
    ax.grid(axis="x", color="#e8e8e8", lw=0.5, zorder=0)

    # Group legend
    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g)
        for g in GROUP_COLORS
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=_FS["tick"], ncol=3, framealpha=0,
              handlelength=1.0, borderpad=0.3)

    _panel_label(ax, "c")
    ax.set_title(
        "Nested Elastic-Net feature importance: hydroclimate memory outweighs rainfall alone",
        fontsize=_FS["label"], loc="left", pad=4,
    )


# ─────────────────────────────────────────────────────────────────
#  Main figure assembly
# ─────────────────────────────────────────────────────────────────

def make_figure_2(figure_dir: Path, formats: list[str]) -> None:
    """Build and save Figure 2."""
    _apply_nature_style()
    df_gbm = _make_gbm_timeseries()

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    outer_gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[1.9, 2.2, 1.75],
        hspace=0.58,
        left=0.10, right=0.97,
        top=0.945, bottom=0.065,
    )

    # ── Panel a ──────────────────────────────────────────────────
    ax_a = fig.add_subplot(outer_gs[0])
    _panel_a(ax_a, df_gbm)

    # ── Panel b — nested 2×4 grid ────────────────────────────────
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=outer_gs[1],
        hspace=0.52, wspace=0.22,
    )
    axes_b = [fig.add_subplot(inner_gs[r, c])
              for r in range(2) for c in range(4)]
    _panel_b(axes_b)

    # Shared sub-title for panel b block
    b_pos  = outer_gs[1].get_position(fig)
    fig.text(
        0.10, b_pos.y1 + 0.004,
        "Eight hydro-climatic regions — annual peak discharge vs. precipitation anomaly",
        ha="left", va="bottom",
        fontsize=_FS["label"], style="italic", color="#333333",
    )

    # ── Panel c ──────────────────────────────────────────────────
    ax_c = fig.add_subplot(outer_gs[2])
    _panel_c(ax_c)

    # ── Figure-level text ─────────────────────────────────────────
    fig.text(
        0.10, 0.985,
        "Figure 2 | The physical disconnect between extreme precipitation and flood occurrence.",
        ha="left", va="top",
        fontsize=_FS["label"] + 0.5, fontweight="bold",
    )
    fig.text(
        0.10, 0.010,
        "Synthetic data calibrated to Stephens et al. (2015); Merz et al. (2021); "
        "Seneviratne et al. (2010); DiBaldassarre & Montanari (2009).",
        ha="left", va="bottom",
        fontsize=_FS["caption"], color="#555555",
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure_2_precip_floodiness_disconnect"
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
    print(f"Generating Figure 2 -> {figure_dir}")
    make_figure_2(figure_dir, args.formats)
    print("Done.")


if __name__ == "__main__":
    main()
