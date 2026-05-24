"""Figure 4 | Cascading Hazards Exceed Meteorological Boundaries in Mountainous Terrains.

Nature Geoscience double-column style (183 mm / 7.2 in wide).

Four panels
-----------
a  3D perspective schematic of a GLOF process chain: glacial lake,
   lateral moraine failure, displacement wave, and downstream release.
b  Reconstructed breach hydrograph — liquid phase (water) and solid
   phase (sediment-laden slurry) vastly exceed the monsoon baseflow.
c  Longitudinal valley profile — pre-event (red dashed) vs. post-event
   (blue) thalweg elevation; red shading = erosion, blue = deposition.
d  Synthetic post-event overhead scene: sediment fan, channel avulsion,
   and annotated hydropower infrastructure damage.

Synthetic data calibrated to Veh et al. (2020); Sattar et al. (2025);
Zheng et al. (2021).

Usage
-----
    python scripts/make_figure4_glof.py
    python scripts/make_figure4_glof.py \\
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── paths ─────────────────────────────────────────────────────────
DEFAULT_FIGURE_DIR = Path("paper/flood_nature_geoscience/new-figures")

# ── layout ────────────────────────────────────────────────────────
_FIG_W = 7.2
_FIG_H = 7.6
_FONT  = "DejaVu Sans"
_FS    = {"panel": 9, "label": 8, "tick": 7, "caption": 6.5}


# ─────────────────────────────────────────────────────────────────
#  Synthetic-data generators
# ─────────────────────────────────────────────────────────────────

def _make_glof_dem(nx: int = 80, ny: int = 60) -> tuple[np.ndarray, ...]:
    """Synthetic glacial valley DEM with lake and moraine dam."""
    rng = np.random.default_rng(11)
    x   = np.linspace(0, 8, nx)    # km downstream
    y   = np.linspace(-3, 3, ny)   # km cross-valley
    X, Y = np.meshgrid(x, y)

    # V-shaped glacial valley deepening upstream
    valley_depth = 0.9 * (8 - X) + 0.12 * Y**2
    # Add rough glacial texture
    noise = rng.normal(0, 0.12, (ny, nx))
    Z = valley_depth + noise
    Z = np.maximum(Z, 0)

    # Moraine dam: transverse ridge at x≈2 km
    dam_mask = (X > 1.6) & (X < 2.2)
    Z[dam_mask] += 1.6 * np.exp(-((X[dam_mask] - 1.9) / 0.18)**2)

    # Lake surface: flat behind the moraine (x < 1.9, Z < lake_level)
    lake_level = Z[dam_mask].min() - 0.05
    lake_mask  = (X < 1.85) & (Z < lake_level + 0.4)
    Z_lake     = Z.copy()
    Z_lake[lake_mask] = lake_level

    return X, Y, Z, Z_lake, lake_mask, lake_level


def _make_hydrograph() -> pd.DataFrame:
    """GLOF breach hydrograph — liquid and solid phases.

    Peak calibrated to Sattar et al. (2025) Sikkim-class GLOF
    (~36 000 m³/s peak at breach).
    """
    t = np.linspace(0, 9, 900)   # hours

    # Liquid phase: rapid rise (~1.8 h to peak) then exponential recession
    def _hydro(t, t_peak, q_peak, rise_k, fall_k):
        rise = q_peak * (1 - np.exp(-rise_k * np.maximum(t - 0, 0)))
        fall = q_peak * np.exp(-fall_k * np.maximum(t - t_peak, 0))
        return np.where(t < t_peak, rise, fall)

    q_liquid = _hydro(t, 1.8, 36_500, 2.8, 0.85)
    # Solid phase (sediment slurry): slightly higher peak density, wider
    q_solid  = _hydro(t, 2.1, 28_000, 2.2, 0.70) * 0.92

    # Monsoon baseflow: roughly constant with small diurnal variation
    q_monsoon = 780 + 80 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(3).normal(0, 25, len(t))

    return pd.DataFrame({"t": t, "liquid": q_liquid,
                          "solid": q_solid, "monsoon": q_monsoon})


def _make_thalweg_profiles() -> pd.DataFrame:
    """Pre- and post-GLOF longitudinal thalweg elevation (m a.s.l.)."""
    rng  = np.random.default_rng(22)
    dist = np.linspace(0, 120, 300)   # km from dam

    # Pre-event: smooth concave-up profile
    pre = 4_200 - 28 * dist + 0.09 * dist**2 + rng.normal(0, 8, len(dist))
    pre = np.maximum(pre, 600)

    # Post-event: deep erosion near breach (0–15 km), deposition 50–90 km
    erosion_depth = 85 * np.exp(-((dist - 4) / 7)**2)  # up to –85 m near breach
    deposit_raise = 22 * np.exp(-((dist - 68) / 18)**2)  # +22 m deposition fan

    noise_post = rng.normal(0, 6, len(dist))
    post = pre - erosion_depth + deposit_raise + noise_post

    return pd.DataFrame({"dist": dist, "pre": pre, "post": post})


def _make_satellite_scene(nr: int = 240, nc: int = 280) -> np.ndarray:
    """Synthetic post-event overhead scene (RGB, uint8)."""
    rng = np.random.default_rng(55)
    img = np.full((nr, nc, 3), [215, 208, 195], dtype=np.uint8)

    # ── Valley walls (darker slope texture) ──────────────────────
    rows_2d, cols_2d = np.mgrid[:nr, :nc]
    slope_mask = (rows_2d < 40) | (rows_2d > nr - 40)
    img[slope_mask] = np.clip(img[slope_mask].astype(int) - 25, 0, 255)

    # ── Pre-avulsion channel (narrow, ~5px wide) ─────────────────
    ch_row_fn = lambda c: int(nr // 2 + 20 * np.sin(2 * np.pi * c / nc * 2.5))
    for c in range(nc):
        cr = ch_row_fn(c)
        img[max(0, cr - 2): cr + 3, c] = [100, 148, 200]   # water blue

    # ── Sediment fan (post-event deposit, tan/gray) ───────────────
    fan_cx, fan_cy = nc * 0.62, nr * 0.5
    fan_rx, fan_ry = nc * 0.28, nr * 0.22
    fan_mask = ((cols_2d - fan_cx)**2 / fan_rx**2
                + (rows_2d - fan_cy)**2 / fan_ry**2) < 1
    # Gradient: darker toward apex, lighter at toe
    dist_apex = np.sqrt(((cols_2d - fan_cx * 0.7) / fan_rx)**2
                        + ((rows_2d - fan_cy) / fan_ry)**2)
    fan_gray = (170 + 30 * dist_apex).clip(155, 200).astype(np.uint8)
    img[fan_mask, 0] = fan_gray[fan_mask]
    img[fan_mask, 1] = (fan_gray[fan_mask] * 0.92).astype(np.uint8)
    img[fan_mask, 2] = (fan_gray[fan_mask] * 0.80).astype(np.uint8)

    # ── Avulsed channel through fan ───────────────────────────────
    for c in range(int(nc * 0.45), nc):
        cr = int(nr // 2 + 15 + 18 * np.sin(2 * np.pi * c / nc * 3 + 1.2))
        img[max(0, cr - 2): cr + 3, c] = [80, 130, 185]

    # ── Debris / boulder deposits (speckle) ──────────────────────
    n_rocks = 180
    rx = rng.integers(int(nc * 0.40), int(nc * 0.75), n_rocks)
    ry = rng.integers(int(nr * 0.35), int(nr * 0.65), n_rocks)
    for rxi, ryi in zip(rx, ry):
        img[max(0, ryi - 1): ryi + 2, max(0, rxi - 1): rxi + 2] = [95, 85, 75]

    # ── Intact reservoir (upstream, blue) ─────────────────────────
    res_mask = (cols_2d < int(nc * 0.18)) & (np.abs(rows_2d - nr // 2) < 18)
    img[res_mask] = [65, 120, 185]

    return img


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
                 x: float = -0.14, y: float = 1.03) -> None:
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

def _panel_a(ax: "Axes3D", X, Y, Z, Z_lake, lake_mask,
             lake_level: float) -> None:
    """(a) 3D GLOF process-chain schematic."""
    # Terrain surface
    terrain_norm = (Z - Z.min()) / (Z.max() - Z.min())
    cmap_terrain = plt.get_cmap("terrain")
    face_colors  = cmap_terrain(terrain_norm)

    ax.plot_surface(X, Y, Z, facecolors=face_colors,
                    alpha=0.88, linewidth=0, antialiased=True,
                    rcount=40, ccount=40)

    # Lake surface (flat blue)
    lx = X[lake_mask].reshape(-1)
    ly = Y[lake_mask].reshape(-1)
    lz = np.full_like(lx, lake_level)
    ax.scatter(lx, ly, lz, c="#3a8fc7", s=1.5, alpha=0.55, zorder=3)

    # Failure zone arrow (moraine breach)
    ax.quiver(1.9, 0.0, lake_level + 0.8,
              1.2, 0.0, -1.2,
              color="#d73027", linewidth=1.8, arrow_length_ratio=0.28)

    # Annotations
    ax.text(0.6, 2.5, lake_level + 1.2, "Glacial\nlake",
            fontsize=6.0, color="#1a5e8f", fontweight="bold", ha="center")
    ax.text(1.9, 0.3, lake_level + 2.0, "Moraine\nfailure",
            fontsize=5.8, color="#d73027", fontweight="bold")
    ax.text(5.0, -2.5, 1.2, "Downstream\nchannel",
            fontsize=5.8, color="#333333")

    ax.set_xlabel("Distance (km)", fontsize=_FS["tick"], labelpad=2)
    ax.set_ylabel("Cross-valley (km)", fontsize=_FS["tick"], labelpad=2)
    ax.set_zlabel("Elev. (rel.)", fontsize=_FS["tick"], labelpad=2)
    ax.tick_params(labelsize=_FS["tick"] - 1, pad=0)
    ax.set_box_aspect([8, 6, 3])
    ax.view_init(elev=28, azim=-55)
    ax.grid(False)
    ax.set_title("GLOF process chain — moraine failure and breach",
                 fontsize=_FS["label"], pad=4)


def _panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """(b) Breach hydrograph: liquid vs. solid phase."""
    blue  = "#2166ac"
    brown = "#8c510a"
    gray  = "#888888"

    ax.fill_between(df["t"], df["solid"], alpha=0.18, color=brown, zorder=1)
    ax.fill_between(df["t"], df["liquid"], df["solid"],
                    alpha=0.22, color=blue, zorder=1)

    ax.plot(df["t"], df["liquid"], color=blue, lw=1.8, zorder=3,
            label="Liquid phase (water)")
    ax.plot(df["t"], df["solid"],  color=brown, lw=1.6, ls="-.", zorder=3,
            label="Solid phase (sediment slurry)")
    ax.plot(df["t"], df["monsoon"], color=gray, lw=1.2, ls="--", zorder=2,
            label="Monsoon baseflow")

    # Peak discharge annotation
    t_pk  = float(df.loc[df["liquid"].idxmax(), "t"])
    q_pk  = float(df["liquid"].max())
    ax.annotate(
        f"Q$_{{peak}}$ = {q_pk/1000:.0f} × 10³ m³/s",
        xy=(t_pk, q_pk), xytext=(t_pk + 1.5, q_pk * 0.88),
        fontsize=_FS["tick"], color=blue,
        arrowprops=dict(arrowstyle="->", color=blue, lw=0.8),
    )

    ax.set_xlim(0, 9)
    ax.set_ylim(-500, 42_000)
    ax.set_xlabel("Time since breach (hours)")
    ax.set_ylabel("Discharge (m³ s⁻¹)")
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=_FS["tick"] - 0.3,
              handlelength=1.2)
    _clean_spines(ax)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    _panel_label(ax, "b")
    ax.set_title("Breach hydrograph vastly exceeds\nclimatological flood magnitudes",
                 fontsize=_FS["label"], loc="left", pad=4)


def _panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    """(c) Longitudinal thalweg: pre- vs. post-GLOF with shaded E/D zones."""
    dist = df["dist"].to_numpy()
    pre  = df["pre"].to_numpy()
    post = df["post"].to_numpy()

    ax.fill_between(dist, post, pre,
                    where=(pre > post), interpolate=True,
                    color="#d73027", alpha=0.28, label="Erosion",  zorder=1)
    ax.fill_between(dist, post, pre,
                    where=(post > pre), interpolate=True,
                    color="#4575b4", alpha=0.28, label="Deposition", zorder=1)

    ax.plot(dist, pre,  color="#d73027", lw=1.5, ls="--",
            label="Pre-GLOF thalweg", zorder=3)
    ax.plot(dist, post, color="#1a6faf", lw=1.8,
            label="Post-GLOF thalweg", zorder=3)

    # Annotate key zones
    ax.annotate("Max. erosion\n(−85 m)", xy=(4, float(post[np.argmin(pre - post)])),
                xytext=(14, float(post[np.argmin(pre - post)]) + 280),
                fontsize=5.8, color="#d73027",
                arrowprops=dict(arrowstyle="->", color="#d73027", lw=0.7))
    ax.annotate("Deposition\nfan (+22 m)", xy=(68, float(post[150])),
                xytext=(78, float(post[150]) + 250),
                fontsize=5.8, color="#4575b4",
                arrowprops=dict(arrowstyle="->", color="#4575b4", lw=0.7))

    ax.set_xlim(0, 120)
    ax.set_xlabel("Distance downstream from dam (km)")
    ax.set_ylabel("Thalweg elevation (m a.s.l.)")
    ax.legend(loc="upper right", fontsize=_FS["tick"],
              handlelength=1.2, ncol=2)
    _clean_spines(ax)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    _panel_label(ax, "c")
    ax.set_title("Longitudinal valley profile — massive basal erosion and downstream deposition",
                 fontsize=_FS["label"], loc="left", pad=4)


def _panel_d(ax: plt.Axes, img: np.ndarray) -> None:
    """(d) Synthetic post-event overhead scene with infrastructure damage."""
    nr, nc = img.shape[:2]
    ax.imshow(img, origin="upper", aspect="auto", interpolation="bilinear")

    # ── Infrastructure annotations ────────────────────────────────
    infra = [
        # (col, row, label, color, offset_col, offset_row)
        (int(nc * 0.14), int(nr * 0.50), "Intact\nreservoir",
         "#2166ac", -28, -22),
        (int(nc * 0.55), int(nr * 0.50), "Destroyed\nrun-of-river dam",
         "#d73027", 10, -24),
        (int(nc * 0.72), int(nr * 0.54), "Sediment\nfan apex",
         "#8c510a", 8, 14),
        (int(nc * 0.82), int(nr * 0.52), "Avulsed\nchannel",
         "#1a6faf", 8, -22),
    ]
    for cx, cy, lbl, col, dx, dy in infra:
        ax.add_patch(mpatches.Circle((cx, cy), radius=7,
                                      ec=col, fc="none", lw=1.5, zorder=4))
        ax.annotate(
            lbl,
            xy=(cx, cy), xytext=(cx + dx, cy + dy),
            fontsize=5.6, color=col, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.75),
            zorder=5,
        )

    # Scale bar (~1 km equivalent)
    sb_start, sb_end = int(nc * 0.07), int(nc * 0.20)
    y_sb = nr - 12
    ax.plot([sb_start, sb_end], [y_sb, y_sb], "w-", lw=3.0,
            solid_capstyle="butt", zorder=5)
    ax.plot([sb_start, sb_end], [y_sb, y_sb], "-", color="#222222",
            lw=1.6, solid_capstyle="butt", zorder=6)
    ax.text((sb_start + sb_end) / 2, y_sb - 5, "1 km",
            ha="center", va="bottom",
            fontsize=_FS["tick"] - 0.5, color="white",
            fontweight="bold", zorder=7)

    # North arrow
    ax.annotate("N", xy=(nc - 12, 22), xytext=(nc - 12, 38),
                ha="center", fontsize=7, fontweight="bold",
                color="white",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2))

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(0.02, 0.97, "d", transform=ax.transAxes,
            fontsize=_FS["panel"], fontweight="bold",
            va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#333333", ec="none", alpha=0.65))
    ax.set_title("Post-event scene — sediment aggradation and infrastructure loss",
                 fontsize=_FS["label"], loc="left", pad=3)


# ─────────────────────────────────────────────────────────────────
#  Main figure assembly
# ─────────────────────────────────────────────────────────────────

def make_figure_4(figure_dir: Path, formats: list[str]) -> None:
    """Build and save Figure 4."""
    _apply_nature_style()

    X, Y, Z, Z_lake, lake_mask, lake_level = _make_glof_dem()
    hydro_df  = _make_hydrograph()
    thalweg_df = _make_thalweg_profiles()
    sat_img   = _make_satellite_scene()

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.52,
        wspace=0.38,
        left=0.10, right=0.97,
        top=0.945, bottom=0.075,
    )

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, X, Y, Z, Z_lake, lake_mask, lake_level)
    _panel_b(ax_b, hydro_df)
    _panel_c(ax_c, thalweg_df)
    _panel_d(ax_d, sat_img)

    # Panel letter for 3D axis (placed via figure coords)
    ax_a.text2D(-0.10, 1.03, "a", transform=ax_a.transAxes,
                fontsize=_FS["panel"], fontweight="bold",
                va="bottom", ha="left")
    _panel_label(ax_b, "b")
    _panel_label(ax_c, "c")
    # panel d label is inside the imshow

    fig.text(
        0.10, 0.985,
        "Figure 4 | Cascading hazards exceed meteorological boundaries in mountainous terrains.",
        ha="left", va="top",
        fontsize=_FS["label"] + 0.5, fontweight="bold",
    )
    fig.text(
        0.10, 0.012,
        "Synthetic data calibrated to Sattar et al. (2025); Veh et al. (2020); Zheng et al. (2021).",
        ha="left", va="bottom",
        fontsize=_FS["caption"], color="#555555",
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure_4_glof_cascading"
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
    print(f"Generating Figure 4 -> {figure_dir}")
    make_figure_4(figure_dir, args.formats)
    print("Done.")


if __name__ == "__main__":
    main()
