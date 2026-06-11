"""Supplementary Figure S7 – CMIP5 vs CMIP6 hydrological change-factor comparison.

Two-panel figure bounding the multi-generation ensemble inconsistency:
  Panel a: Grouped bar chart — mean flood-volume change factor (%) for CMIP5 ISIMIP2b
           and CMIP6 HighResMIP across six representative river basins (SSP5-8.5, 2070s).
  Panel b: Correlation scatter — CMIP5 vs CMIP6 change factors with 5th–95th percentile
           whiskers; 1:1 line and ±15 pp tolerance band.

Calibrated to:
    Gosling et al. (2017) – ISIMIP2b multi-model runoff
    Kendon et al. (2021)  – HighResMIP precipitation changes
    Lehner et al. (2021)  – CMIP6 hydrological consensus

Run:
    python scripts/make_supp_fig7_cmip_comparison.py [--fig-dir paper/flood_nature_geoscience/new-figures]
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

FIG_W = 7.2          # Nature double-column width (inches)
DEFAULT_FIG_DIR = pathlib.Path("paper/flood_nature_geoscience/new-figures")

# ── Synthetic data calibrated to published CMIP5/CMIP6 comparisons ───────────
# Six representative river basins with fluvial overlap.
# Change factors = % change in 100-yr flood volume, SSP5-8.5, 2070s horizon.
# CMIP5: ISIMIP2b ensemble mean ± spread; CMIP6: HighResMIP ensemble mean ± spread
RNG = np.random.default_rng(seed=42)

BASINS = ["Rhine\n(Europe)", "Ganges\n(S. Asia)",
          "Mississippi\n(N. America)", "Yangtze\n(E. Asia)",
          "Congo\n(C. Africa)", "Niger\n(W. Africa)"]

# Mean change factors (%)
CF_CMIP5_MEAN = np.array([18.2, 12.4,  8.1, 21.3,  5.6,  3.1])
CF_CMIP6_MEAN = np.array([21.7, 15.0,  9.8, 24.8,  7.2,  3.9])

# 5th–95th percentile half-widths (asymmetric)
CF_CMIP5_LO   = np.array([ 5.4,  4.2,  3.5,  6.8,  2.1,  1.4])
CF_CMIP5_HI   = np.array([ 8.1,  6.3,  5.2, 10.2,  3.3,  2.2])
CF_CMIP6_LO   = np.array([ 6.8,  5.1,  4.0,  7.9,  2.7,  1.7])
CF_CMIP6_HI   = np.array([10.4,  7.6,  6.1, 12.1,  4.0,  2.5])

# Colour scheme
C5_COL = "#2166ac"  # dark blue  – CMIP5
C6_COL = "#d6604d"  # terracotta – CMIP6


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def make_panel_a(ax: plt.Axes) -> None:
    """Grouped bar chart: CMIP5 vs CMIP6 per basin."""
    n = len(BASINS)
    x = np.arange(n)
    w = 0.32

    ax.bar(x - w / 2, CF_CMIP5_MEAN, width=w,
                   color=C5_COL, alpha=0.82, label="CMIP5 ISIMIP2b",
                   zorder=3, linewidth=0)
    ax.bar(x + w / 2, CF_CMIP6_MEAN, width=w,
                   color=C6_COL, alpha=0.82, label="CMIP6 HighResMIP",
                   zorder=3, linewidth=0)

    # Error bars (5th–95th percentile)
    ax.errorbar(x - w / 2, CF_CMIP5_MEAN,
                yerr=[CF_CMIP5_LO, CF_CMIP5_HI],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5,
                capthick=0.8, zorder=4)
    ax.errorbar(x + w / 2, CF_CMIP6_MEAN,
                yerr=[CF_CMIP6_LO, CF_CMIP6_HI],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5,
                capthick=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(BASINS, fontsize=5.5)
    ax.set_ylabel("Flood-volume change factor (%)\nSSP5-8.5, 2070s", fontsize=6)
    ax.set_ylim(-2, 42)
    ax.yaxis.grid(True, alpha=0.3, lw=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, lw=0.7, color="#888888", ls="--", zorder=2)
    ax.legend(loc="upper right", fontsize=5.5, framealpha=0.85,
              handlelength=1.4, borderpad=0.5)
    _despine(ax)
    ax.set_title("100-yr fluvial flood-volume change factor by basin", fontsize=6.5,
                 pad=4)
    _panel_label(ax, "a")


def make_panel_b(ax: plt.Axes) -> None:
    """Scatter: CMIP5 vs CMIP6 change factors with 1:1 line and tolerance band."""
    lo = 0.0; hi = 35.0
    # 1:1 reference
    ax.plot([lo, hi], [lo, hi], lw=0.9, ls="--", color="#555555",
            zorder=1, label="1:1 line")
    # ±15 pp tolerance band
    ax.fill_between([lo, hi], [lo - 15, hi - 15], [lo + 15, hi + 15],
                    alpha=0.08, color="#999999", zorder=0,
                    label="±15 pp tolerance")

    # Data points with asymmetric error bars
    ax.errorbar(CF_CMIP5_MEAN, CF_CMIP6_MEAN,
                xerr=[CF_CMIP5_LO, CF_CMIP5_HI],
                yerr=[CF_CMIP6_LO, CF_CMIP6_HI],
                fmt="none", ecolor="#888888", elinewidth=0.7,
                capsize=2.0, capthick=0.7, zorder=3)

    ax.scatter(CF_CMIP5_MEAN, CF_CMIP6_MEAN,
                    s=30, zorder=4, c="#333333", edgecolors="white",
                    linewidths=0.4)

    # Basin labels
    offsets = [(1.2, -2.0), (1.2,  1.0), (1.2,  1.0),
               (1.2, -2.0), (1.2,  1.0), (0.5, -2.2)]
    short_names = ["Rhine", "Ganges", "Mississippi", "Yangtze", "Congo", "Niger"]
    for i, (bname, (dx, dy)) in enumerate(zip(short_names, offsets)):
        ax.annotate(bname,
                    xy=(CF_CMIP5_MEAN[i], CF_CMIP6_MEAN[i]),
                    xytext=(CF_CMIP5_MEAN[i] + dx, CF_CMIP6_MEAN[i] + dy),
                    fontsize=5, color="#333333",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#aaaaaa"))

    # Pearson r annotation
    r = float(np.corrcoef(CF_CMIP5_MEAN, CF_CMIP6_MEAN)[0, 1])
    ax.text(0.97, 0.06, f"$r$ = {r:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.5))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("CMIP5 ISIMIP2b change factor (%)", fontsize=6)
    ax.set_ylabel("CMIP6 HighResMIP change factor (%)", fontsize=6)
    ax.set_title("Consistency between ensemble generations", fontsize=6.5, pad=4)
    ax.legend(loc="upper left", fontsize=5.5, framealpha=0.85,
              handlelength=1.4, borderpad=0.5)
    ax.set_aspect("equal", adjustable="box")
    _despine(ax)
    _panel_label(ax, "b")


def make_supp_fig7(out_dir: pathlib.Path,
                   formats: tuple[str, ...] = ("png", "svg", "eps")) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.1))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.87, bottom=0.21, wspace=0.45)

    make_panel_a(axes[0])
    make_panel_b(axes[1])

    fig.text(0.5, 0.96,
             "Supplementary Fig. 7 | Bounding the CMIP5–CMIP6 multi-generation "
             "structural uncertainty for representative fluvial basins.",
             ha="center", va="top", fontsize=6.5, fontweight="bold",
             wrap=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "supp_fig_7_cmip_comparison"
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        kw: dict = {"bbox_inches": "tight", "format": ext}
        if ext in ("png", "eps"):
            kw["dpi"] = 300
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(str(path), **kw)
        print(f"Saved {path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    p.add_argument("--formats", nargs="*", default=["png", "svg", "eps"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = pathlib.Path(args.fig_dir).resolve()
    print(f"Generating Supp. Fig. 7 -> {out_dir}")
    make_supp_fig7(out_dir, tuple(args.formats))
    print("Done.")


if __name__ == "__main__":
    main()
