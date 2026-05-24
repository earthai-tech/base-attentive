"""Figure 1 | The Global Decoupling of Flood Hazard and Human Exposure.

Nature Geoscience double-column style (183 mm / 7.2 in wide).

Four panels
-----------
a  Bivariate world choropleth — change in inundated area (x-class)
   vs. change in population exposed (y-class), 2000–2024, 3×3 scheme.
b  Time-series — normalised growth of global inundated area (blue) and
   total population exposed (red dashed) with divergence fill.
c  Scatter — traditional 100-yr return-period model estimate (x) vs.
   satellite-observed exposure (y), log–log, coloured by continent.
d  Boxplots + strip — percentage increase in flood-exposed population
   by continent, 2000–2020.

Synthetic data calibrated to peer-reviewed estimates:
  Ceola et al. (2014); Rentschler & Salhab (2022);
  Bates et al. (2021); Hirabayashi et al. (2013); Jongman et al. (2012).

Usage
-----
    python scripts/make_figure1_decoupling.py
    python scripts/make_figure1_decoupling.py \\
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
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

# ── paths ─────────────────────────────────────────────────────────
DEFAULT_FIGURE_DIR = Path("paper/flood_nature_geoscience/new-figures")

# ── Nature Geoscience layout constants ───────────────────────────
_FIG_W = 7.2    # 183 mm
_FIG_H = 9.6    # three-row layout
_FONT  = "DejaVu Sans"
_FS    = {"panel": 9, "label": 8, "tick": 7, "caption": 6.5}

# ── Continent palette (colour-blind-safe diverging tones) ─────────
CONT_COLORS: dict[str, str] = {
    "Asia":          "#2166ac",
    "Africa":        "#d6604d",
    "North America": "#4dac26",
    "South America": "#f1a340",
    "Europe":        "#762a83",
    "Oceania":       "#1b9e77",
}
CONT_ORDER = list(CONT_COLORS)

# ── Bivariate 3×3 Stevens/Brewer colour matrix ───────────────────
# Rows = population-exposure class (0 = low … 2 = high)
# Cols = inundated-area class      (0 = low … 2 = high)
_BIV3 = [
    ["#e8e8e8", "#ace4e4", "#5ac8c8"],  # low pop change
    ["#dfb0d6", "#a5add3", "#5698b9"],  # mid pop change
    ["#be64ac", "#8c62aa", "#3b4994"],  # high pop change
]


def _biv_color(area_cls: int, pop_cls: int) -> str:
    return _BIV3[int(pop_cls)][int(area_cls)]


# ─────────────────────────────────────────────────────────────────
#  Synthetic-data generators  (fixed seeds for reproducibility)
# ─────────────────────────────────────────────────────────────────

def _make_timeseries() -> pd.DataFrame:
    """Normalised growth indices 2000–2024.

    Inundated area: ~0.5 %/yr (Hirabayashi 2013 SSP1–2.6 bound).
    Population exposed: ~1.6 %/yr (Ceola 2014: 20–24 % per 15 yr).
    """
    rng = np.random.default_rng(42)
    years = np.arange(2000, 2025)
    n = len(years)
    area = 1.0 + 0.005 * (years - 2000) + np.cumsum(rng.normal(0, 0.007, n))
    area /= area[0]
    pop  = 1.0 + 0.016 * (years - 2000) + np.cumsum(rng.normal(0, 0.011, n))
    pop  /= pop[0]
    return pd.DataFrame({"year": years, "area": area, "pop": pop})


def _make_model_vs_obs() -> pd.DataFrame:
    """100-yr return-period model estimate vs. satellite-observed exposure.

    Bias factors reflect Rentschler & Salhab (2022): satellite obs
    up to 10× higher than traditional hazard-model estimates.
    """
    rng = np.random.default_rng(123)
    bias = {
        "Asia":          4.4,
        "Africa":        5.1,
        "North America": 1.5,
        "South America": 2.4,
        "Europe":        1.2,
        "Oceania":       1.4,
    }
    n_pts = {
        "Asia": 55, "Africa": 45, "North America": 30,
        "South America": 22, "Europe": 28, "Oceania": 12,
    }
    rows: list[pd.DataFrame] = []
    for cont in CONT_ORDER:
        n = n_pts[cont]
        model = rng.lognormal(np.log(4e5), 1.4, n)
        obs   = model * bias[cont] * np.exp(rng.normal(0, 0.5, n))
        rows.append(pd.DataFrame({"continent": cont, "model": model, "obs": obs}))
    return pd.concat(rows, ignore_index=True)


def _make_boxplot_data() -> pd.DataFrame:
    """% increase in flood-exposed population per country, 2000–2020.

    Median values calibrated to Jongman et al. (2012) and
    Rentschler & Salhab (2022) regional estimates.
    """
    rng = np.random.default_rng(77)
    params = {
        "Asia":          (32, 14, 45),   # (median %, spread, n_countries)
        "Africa":        (38, 16, 40),
        "North America": (11,  8, 20),
        "South America": (20, 11, 18),
        "Europe":        ( 6,  5, 22),
        "Oceania":       ( 8,  6, 10),
    }
    rows: list[pd.DataFrame] = []
    for cont, (med, std, n) in params.items():
        vals = rng.normal(med, std, n)
        rows.append(pd.DataFrame({"continent": cont, "pct": vals}))
    return pd.concat(rows, ignore_index=True)


def _make_world_bivariate() -> "gpd.GeoDataFrame | None":
    """World polygons with discrete 3-class bivariate assignments."""
    if not _HAS_GPD:
        return None

    # Attempt to load NaturalEarth 110m country polygons
    world: gpd.GeoDataFrame | None = None
    for loader in _WORLD_LOADERS:
        try:
            world = loader()
            if world is not None and len(world) > 0:
                break
        except Exception:
            continue
    if world is None:
        return None

    rng = np.random.default_rng(999)
    # (mean Δ-area, mean Δ-pop) per continent — calibrated to literature
    cont_params: dict[str, tuple[float, float]] = {
        "Asia":                    (0.48, 0.78),
        "Africa":                  (0.36, 0.80),
        "North America":           (0.30, 0.27),
        "South America":           (0.40, 0.55),
        "Europe":                  (0.20, 0.17),
        "Oceania":                 (0.27, 0.20),
        "Seven seas (open ocean)": (0.05, 0.05),
    }
    n = len(world)
    da = np.zeros(n)
    dp = np.zeros(n)
    cont_col = "continent" if "continent" in world.columns else None
    for i, row in enumerate(world.itertuples()):
        cont = getattr(row, cont_col, "") if cont_col else ""
        am, pm = cont_params.get(cont, (0.30, 0.30))
        da[i] = np.clip(rng.normal(am, 0.16), 0.0, 1.0)
        dp[i] = np.clip(rng.normal(pm, 0.18), 0.0, 1.0)

    def _tertile_class(arr: np.ndarray) -> np.ndarray:
        q = np.percentile(arr, [33.3, 66.7])
        return np.digitize(arr, q).clip(0, 2)

    world = world.copy()
    world["delta_area"] = da
    world["delta_pop"]  = dp
    world["area_cls"]   = _tertile_class(da)
    world["pop_cls"]    = _tertile_class(dp)
    world["biv_color"]  = [_biv_color(a, p)
                           for a, p in zip(world["area_cls"], world["pop_cls"])]
    return world


def _load_gpd_naturalearth() -> "gpd.GeoDataFrame | None":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore[attr-defined]


def _load_gpd_geodatasets() -> "gpd.GeoDataFrame | None":
    import geodatasets  # type: ignore
    return gpd.read_file(geodatasets.get_path("naturalearth admin_0_countries"))


_WORLD_LOADERS = [_load_gpd_naturalearth, _load_gpd_geodatasets]


# ─────────────────────────────────────────────────────────────────
#  Style helpers
# ─────────────────────────────────────────────────────────────────

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
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,
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

def _panel_a(ax: plt.Axes,
             world: "gpd.GeoDataFrame | None") -> None:
    """(a) Bivariate choropleth world map."""
    ax.set_facecolor("#c8dcea")   # ocean
    if world is not None:
        world.plot(
            ax=ax,
            color=world["biv_color"].tolist(),
            edgecolor="#9a9a9a",
            linewidth=0.18,
            zorder=2,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 85)
    else:
        ax.set_facecolor("#d8e9f0")
        ax.text(0.5, 0.5,
                "Map unavailable — install geopandas to render",
                transform=ax.transAxes, ha="center", va="center",
                color="#666666", fontsize=7.5, style="italic")

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ── Bivariate legend (inset axes, bottom-left) ─────────────
    leg_ax = ax.inset_axes([0.01, 0.03, 0.11, 0.22])
    leg_ax.set_xlim(0, 3)
    leg_ax.set_ylim(0, 3)
    for r in range(3):
        for c in range(3):
            leg_ax.add_patch(
                mpatches.Rectangle((c, r), 1, 1,
                                   fc=_biv_color(c, r),
                                   ec="white", lw=0.5)
            )
    leg_ax.set_xticks([0, 1.5, 3])
    leg_ax.set_xticklabels(["Low", "", "High"], fontsize=5.5)
    leg_ax.set_yticks([0, 1.5, 3])
    leg_ax.set_yticklabels(["Low", "", "High"], fontsize=5.5)
    leg_ax.set_xlabel("→ Δ Inundated area", fontsize=5.5, labelpad=1.5)
    leg_ax.set_ylabel("→ Δ Pop. exposed", fontsize=5.5, labelpad=1.5)
    leg_ax.tick_params(length=0)
    for sp in leg_ax.spines.values():
        sp.set_linewidth(0.4)

    _panel_label(ax, "a", x=0.005, y=0.97)
    ax.set_title(
        "Bivariate spatial distribution of flood events, 2000–2024\n"
        "Hazard expansion (blue) vs. exposure-driven risk (pink/purple)",
        fontsize=_FS["label"], loc="left", pad=4,
    )


def _panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """(b) Normalised growth time-series with divergence fill."""
    blue = "#2166ac"
    red  = "#d73027"

    ax.plot(df["year"], df["area"], color=blue, lw=1.6,
            label="Inundated area", zorder=3)
    ax.plot(df["year"], df["pop"],  color=red, lw=1.6,
            ls="--", label="Population exposed", zorder=3)
    ax.fill_between(
        df["year"], df["area"], df["pop"],
        where=(df["pop"] >= df["area"]),
        color=red, alpha=0.13, zorder=1,
    )
    ax.axhline(1.0, color="#bbbbbb", lw=0.6, ls=":")

    # Annotate divergence arrow at 2015
    idx = df[df["year"] == 2015].index[0]
    ax.annotate(
        "",
        xy=(2015, float(df.loc[idx, "pop"])),
        xytext=(2015, float(df.loc[idx, "area"])),
        arrowprops=dict(arrowstyle="<->", color=red, lw=0.8),
    )
    ax.text(2015.4, (df.loc[idx, "area"] + df.loc[idx, "pop"]) / 2,
            "divergence", color=red, fontsize=_FS["tick"] - 0.5,
            va="center")

    ax.set_xlim(2000, 2024)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised growth index\n(2000 = 1.0)")
    ax.legend(loc="upper left", fontsize=_FS["tick"])
    _clean_spines(ax)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    _panel_label(ax, "b")
    ax.set_title("Global inundation footprint vs.\nhuman exposure growth",
                 fontsize=_FS["label"], loc="left", pad=4)


def _panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    """(c) Traditional model vs. satellite-observed exposure (log–log)."""
    for cont in CONT_ORDER:
        sub = df[df["continent"] == cont]
        ax.scatter(sub["model"], sub["obs"],
                   color=CONT_COLORS[cont], s=13, alpha=0.65,
                   edgecolors="none", label=cont, zorder=3)

    lims = (8e3, 5e8)
    ax.plot(lims, lims, color="#444444", lw=0.9, ls="--",
            zorder=2, label="1 : 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("100-yr return-period model\nestimate (people)")
    ax.set_ylabel("Satellite-observed\nexposure (people)")

    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left",
              fontsize=_FS["tick"] - 0.5, markerscale=1.3,
              handlelength=1.0, borderpad=0.4)
    _clean_spines(ax)
    ax.grid(True, which="major", color="#e5e5e5", lw=0.5, zorder=0)

    # Annotation: systematic underestimation
    ax.text(0.97, 0.06,
            "Points above 1:1 line =\nsystematic underestimation",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=_FS["tick"] - 0.5, color="#555555", style="italic")

    _panel_label(ax, "c")
    ax.set_title("Traditional model vs. satellite\nexposure estimates",
                 fontsize=_FS["label"], loc="left", pad=4)


def _panel_d(ax: plt.Axes, df: pd.DataFrame) -> None:
    """(d) Boxplots of % change in flood-exposed population by continent."""
    order_sorted = sorted(
        CONT_ORDER,
        key=lambda c: float(df.loc[df["continent"] == c, "pct"].median()),
        reverse=True,
    )

    if _HAS_SNS:
        palette = {c: CONT_COLORS[c] for c in CONT_ORDER}
        df_plot = df.copy()
        df_plot["_hue"] = df_plot["continent"]
        sns.boxplot(
            data=df_plot, x="continent", y="pct",
            order=order_sorted, hue="_hue",
            hue_order=order_sorted, palette=palette,
            width=0.52, fliersize=0, linewidth=0.7,
            legend=False, ax=ax,
        )
        sns.stripplot(
            data=df_plot, x="continent", y="pct",
            order=order_sorted, hue="_hue",
            hue_order=order_sorted, palette=palette,
            size=2.4, alpha=0.50, jitter=True,
            dodge=False, legend=False, ax=ax,
        )
    else:
        # Pure-matplotlib fallback
        data_ordered = [
            df.loc[df["continent"] == c, "pct"].to_numpy()
            for c in order_sorted
        ]
        bx = ax.boxplot(
            data_ordered,
            patch_artist=True,
            showfliers=False,
            widths=0.52,
            medianprops={"color": "#111111", "lw": 1.1},
            whiskerprops={"color": "#888888", "lw": 0.7},
            capprops={"color": "#888888", "lw": 0.7},
        )
        for patch, cont in zip(bx["boxes"], order_sorted):
            patch.set_facecolor(CONT_COLORS[cont])
            patch.set_alpha(0.72)
            patch.set_edgecolor("none")
        ax.set_xticks(range(1, len(order_sorted) + 1))

    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.set_xlabel("Continent")
    ax.set_ylabel("Change in flood-exposed\npopulation (%)")
    ax.set_xticks(range(len(order_sorted)))
    ax.set_xticklabels(order_sorted, rotation=22, ha="right")
    _clean_spines(ax)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    _panel_label(ax, "d", x=-0.065)
    ax.set_title(
        "Socio-economic amplification of flood exposure by continent (2000–2020)\n"
        "Asia and Africa show the fastest growth, driven by floodplain urbanisation",
        fontsize=_FS["label"], loc="left", pad=4,
    )


# ─────────────────────────────────────────────────────────────────
#  Main figure assembly
# ─────────────────────────────────────────────────────────────────

def make_figure_1(figure_dir: Path, formats: list[str]) -> None:
    """Build and save Figure 1."""
    _apply_nature_style()

    world = _make_world_bivariate()
    ts    = _make_timeseries()
    sc    = _make_model_vs_obs()
    bx    = _make_boxplot_data()

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.0, 1.15, 1.15],
        hspace=0.62,
        wspace=0.40,
        left=0.10, right=0.97,
        top=0.945, bottom=0.075,
    )
    ax_map = fig.add_subplot(gs[0, :])
    ax_ts  = fig.add_subplot(gs[1, 0])
    ax_sc  = fig.add_subplot(gs[1, 1])
    ax_bx  = fig.add_subplot(gs[2, :])

    _panel_a(ax_map, world)
    _panel_b(ax_ts,  ts)
    _panel_c(ax_sc,  sc)
    _panel_d(ax_bx,  bx)

    fig.text(
        0.10, 0.985,
        "Figure 1 | Global flood exposure outpaces climate-driven hazard expansion.",
        ha="left", va="top",
        fontsize=_FS["label"] + 0.5, fontweight="bold",
    )
    fig.text(
        0.10, 0.018,
        "Synthetic data calibrated to Ceola et al. (2014); Rentschler & Salhab (2022); "
        "Bates et al. (2021); Hirabayashi et al. (2013).",
        ha="left", va="bottom",
        fontsize=_FS["caption"], color="#555555",
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure_1_global_decoupling"
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--formats", nargs="*", default=["png", "svg"],
                   help="Output formats (default: png svg)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    figure_dir = Path(args.figure_dir).resolve()
    print(f"Generating Figure 1 -> {figure_dir}")
    make_figure_1(figure_dir, args.formats)
    print("Done.")


if __name__ == "__main__":
    main()
