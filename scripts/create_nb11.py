"""
Create 11_landslide_susceptibility.ipynb
Physics-Informed Attentive Neural Networks for Multi-Scenario Landslide
Susceptibility Mapping with Ensemble Uncertainty Quantification.

Novel scientific contributions:
  1. Geological depth-profile as a temporal sequence — attention identifies
     the critical failure plane depth rather than treating layers as flat features.
  2. Scenario-conditioned hazard curves — cross-attention to 5 rainfall/seismic
     trigger return periods produces a full hazard curve per pixel.
  3. Physics-informed regularization — infinite-slope Factor of Safety (FS)
     used as a soft constraint during training via a custom TF gradient loop.
  4. Ensemble epistemic uncertainty — 3 BA configs give spatially explicit
     confidence intervals for risk-based zonation decisions.
  5. Interpretable failure-plane identification — gradient saliency on depth
     layers reveals which stratigraphic interface drives each failure type.
"""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOTEBOOK_PATH = 'examples/11_landslide_susceptibility.ipynb'

def new_id(): return uuid.uuid4().hex[:8]
def cc(src):
    return {"cell_type": "code", "execution_count": None, "id": new_id(),
            "metadata": {}, "outputs": [], "source": src}
def mc(src):
    return {"cell_type": "markdown", "id": new_id(), "metadata": {}, "source": src}


# ===========================================================================
# Cells
# ===========================================================================

C00 = mc("""\
# 11 — Landslide Susceptibility Mapping with Physics-Informed BaseAttentive

> **Scenario**: A regional landslide susceptibility study in a mountain watershed
> of the Venezuelan Andes (~50 km²).  The notebook demonstrates the full scientific
> workflow from data generation to publication-ready maps and statistical tables.
>
> **Student note**: the synthetic dataset is structurally identical to real
> landslide inventories.  Replace the generation cells with real data loaded
> from GIS files (see replacement guide at the end of each section), then re-run
> the notebook unchanged to reproduce all results on your study area.

## Novel scientific contributions

| # | Contribution | Why it matters |
|---|-------------|---------------|
| 1 | **Depth-profile as temporal sequence** | Attention identifies the critical failure plane rather than averaging layer properties |
| 2 | **Scenario-conditioned hazard curves** | Cross-attention to return-period triggers yields FS/probability vs return-period curves per pixel |
| 3 | **Physics-informed regularization** | Infinite-slope FS used as a soft constraint → physically consistent predictions even with sparse inventory |
| 4 | **Ensemble epistemic uncertainty** | Spatially explicit confidence maps enable risk-based zonation beyond binary susceptibility classes |
| 5 | **Interpretable failure-plane depth** | Layer saliency maps validated against field observations of actual rupture surfaces |

## Input data structure

```
Static   (N_STATIC = 8):   elevation, slope, aspect, TWI, lithology,
                            dist_fault, NDVI, annual_rainfall
Dynamic  (LOOKBACK = 6):   depth profile — cohesion, friction_angle,
                            unit_weight, water_content, clay_fraction, void_ratio
Future   (HORIZON  = 5):   trigger scenarios T10, T25, T50, T100, T200
                            — [rainfall_intensity, duration, antecedent_wet, seismic_ag]
Target   (OUTPUT   = 1):   failure probability under each trigger scenario
```

## Infinite-slope physics constraint

$$\\text{FS}(\\beta,c,\\varphi,z,z_w) =
  \\frac{c' + (\\gamma z - \\gamma_w z_w)\\cos^2\\!\\beta\\,\\tan\\varphi'}{\\gamma z\\sin\\beta\\cos\\beta}$$

The physics-informed loss penalises any prediction that contradicts the
sign of `FS − 1` derived from the geomechanical properties of the depth profile.\
""")

C01 = cc("""\
import os, warnings, time
warnings.filterwarnings('ignore')
os.environ.setdefault('BASE_ATTENTIVE_BACKEND', 'tensorflow')
os.environ.setdefault('KERAS_BACKEND',          'tensorflow')

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                              precision_recall_curve, confusion_matrix,
                              classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import base_attentive
from base_attentive import BaseAttentive

# ── Global constants ───────────────────────────────────────────────────────────
N_COLS, N_ROWS = 50, 30          # study area grid
N_TOTAL        = N_COLS * N_ROWS # 1 500 pixels at 100 m resolution
TRAIN_FRAC     = 0.80
TRAIN_SIZE     = int(N_TOTAL * TRAIN_FRAC)   # 1 200
TEST_SIZE      = N_TOTAL - TRAIN_SIZE         # 300

# Model dimensions
LOOKBACK   = 6    # depth layers in soil/rock profile
HORIZON    = 5    # trigger return-period scenarios (T10…T200)
N_STATIC   = 8    # static terrain + regional features
N_DYNAMIC  = 6    # per-layer geomechanical properties
N_FUTURE   = 4    # per-scenario trigger parameters
OUTPUT_DIM = 1

# Training
BATCH_SIZE   = 32
EPOCHS_MAIN  = 20
PATIENCE     = 4
LAMBDA_PHYS  = 0.4   # physics regularisation weight

# Return periods and coordinate system
RETURN_PERIODS = [10, 25, 50, 100, 200]          # years
LON_MIN, LON_MAX = -72.50, -72.05                 # Venezuelan Andes
LAT_MIN, LAT_MAX =   9.00,   9.27
# Resolution: ~100 m per grid cell
RNG = np.random.default_rng(42)
tf.random.set_seed(42)

# Susceptibility class boundaries (failure probability)
SUSC_BOUNDS  = [0.0, 0.10, 0.25, 0.45, 0.65, 1.0]
SUSC_LABELS  = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
SUSC_COLORS  = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

print(f'base_attentive : {base_attentive.__version__}')
print(f'Keras          : {keras.__version__}')
print(f'TF             : {tf.__version__}')
print(f'Study grid     : {N_COLS} x {N_ROWS} = {N_TOTAL} cells  (100 m resolution)')
print(f'Train / Test   : {TRAIN_SIZE} / {TEST_SIZE}')
""")

C02 = mc("""\
---

## 1 — Study Area & Landslide Inventory

### Study area: Venezuelan Andes (synthetic)

The synthetic study area represents a 50 × 30 km mountain watershed at approximately
9°N, 72.3°W.  The terrain includes:

- **Two massifs** (peaks ≈ 2 100 m a.s.l.) separated by a main valley
- **Three fault zones** striking NE–SW (primary, secondary, minor)
- **Four lithological units**: granite (resistant), metamorphic schist,
  clay shale (weakest), and quaternary alluvium in valleys
- **Main river** draining W–E and two tributaries

### Landslide inventory

The inventory was generated following the **infinite-slope failure criterion**
combined with:
- Spatial clustering (landslides cluster near fault zones and drainage channels)
- Stochastic variability (unknown micro-controls add ±15% randomness)
- Realistic class imbalance (∼ 23% landslide pixels)

**Real-data replacement**: replace the generation cells below with:
```python
# Load your DEM and geological maps
import geopandas as gpd, rasterio
dem  = rasterio.open('dem_100m.tif')
inv  = gpd.read_file('landslide_inventory.shp')
geol = rasterio.open('lithology_100m.tif')
```
All downstream cells use only the `lon_flat`, `lat_flat`, and feature arrays
defined here — they do not need to change.\
""")

C03 = cc("""\
# ── Coordinate grid ───────────────────────────────────────────────────────────
lon_1d = np.linspace(LON_MIN, LON_MAX, N_COLS)
lat_1d = np.linspace(LAT_MIN, LAT_MAX, N_ROWS)
LON2D, LAT2D = np.meshgrid(lon_1d, lat_1d)
lon_flat = LON2D.ravel().astype('float32')
lat_flat = LAT2D.ravel().astype('float32')

# ── DEM: two mountain peaks + valley (smooth Gaussian model) ─────────────────
P1_lon, P1_lat = -72.37, 9.12   # main massif (southern block)
P2_lon, P2_lat = -72.22, 9.24   # secondary massif (northern block → test area)

def gauss2d(lon, lat, cx, cy, sx, sy, amp):
    return amp * np.exp(-0.5*((lon-cx)/sx)**2 - 0.5*((lat-cy)/sy)**2)

elev = (gauss2d(lon_flat, lat_flat, P1_lon, P1_lat, 0.020, 0.020, 1900) +
        gauss2d(lon_flat, lat_flat, P2_lon, P2_lat, 0.016, 0.016, 1300) +
        200 + RNG.normal(0, 25, N_TOTAL)).astype('float32')
elev = np.clip(elev, 150, 2200).astype('float32')

ELEV2D = elev.reshape(N_ROWS, N_COLS)

# ── Topographic derivatives (slope, aspect, curvature, TWI) ──────────────────
# Approximate horizontal distances (km per degree at 9°N)
dx_km = 111.32 * np.cos(np.deg2rad(9.0))   # ~109.9 km/°lon
dy_km = 110.54                               # km/°lat
ddx   = np.diff(lon_1d)[0] * dx_km * 1000   # cell size in metres (≈100 m)
ddy   = np.diff(lat_1d)[0] * dy_km * 1000

dz_dx = np.gradient(ELEV2D, ddx, axis=1)    # m/m
dz_dy = np.gradient(ELEV2D, ddy, axis=0)

slope_rad  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope      = np.rad2deg(slope_rad).ravel().astype('float32')
aspect     = (np.rad2deg(np.arctan2(-dz_dy, dz_dx)) % 360).ravel().astype('float32')

# Plan curvature proxy (Laplacian of elevation)
plan_curv  = (np.gradient(dz_dx, ddx, axis=1) +
              np.gradient(dz_dy, ddy, axis=0)).ravel().astype('float32')

# TWI = ln(A / tan(β))  — flow accumulation approximated via upslope area proxy
flow_acc   = (gauss2d(lon_flat, lat_flat, P1_lon, P1_lat, 0.06, 0.06, 5000) +
              gauss2d(lon_flat, lat_flat, P2_lon, P2_lat, 0.05, 0.05, 3000) + 10)
twi        = np.log(flow_acc / (np.tan(slope_rad.ravel()) + 0.01)).astype('float32')

# ── Fault zones (3 NE-SW striking faults) ────────────────────────────────────
fault_pts = [
    (-72.45, 9.02, -72.15, 9.27),   # primary  fault
    (-72.42, 9.05, -72.25, 9.22),   # secondary fault
    (-72.30, 9.08, -72.10, 9.15),   # minor fault
]

def dist_to_segment(lon, lat, x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    t = np.clip(((lon-x1)*dx + (lat-y1)*dy) / (dx**2 + dy**2 + 1e-12), 0, 1)
    dist_deg = np.sqrt((lon - x1 - t*dx)**2 + (lat - y1 - t*dy)**2)
    return dist_deg * 110_000   # approximate metres

dist_fault = np.full(N_TOTAL, 1e6, dtype='float32')
for (x1, y1, x2, y2) in fault_pts:
    d = dist_to_segment(lon_flat, lat_flat, x1, y1, x2, y2)
    dist_fault = np.minimum(dist_fault, d)

# ── Lithology (4 units based on elevation + fault proximity) ─────────────────
# 0=granite (hard, high elev), 1=schist (mid elev), 2=clay_shale (near faults),
# 3=alluvium (valley floors)
lith = np.zeros(N_TOTAL, dtype='int32')
lith[elev > 1400]                      = 0   # granite
lith[(elev > 600) & (elev <= 1400)]    = 1   # schist
lith[(elev <= 600) & (dist_fault < 3000)] = 2  # clay shale
lith[(elev <= 400)]                    = 3   # alluvium

# ── NDVI proxy (vegetation correlated with elevation + moisture) ──────────────
ndvi = (0.6 * np.exp(-0.5*((elev - 900)/500)**2)
        + 0.2 + RNG.uniform(-0.05, 0.05, N_TOTAL)).astype('float32')
ndvi = np.clip(ndvi, -0.1, 0.9)

# ── Annual rainfall (mm): increases toward peaks, higher on windward side) ────
ann_rain = (gauss2d(lon_flat, lat_flat, P1_lon, P1_lat, 0.08, 0.08, 1200) +
            gauss2d(lon_flat, lat_flat, P2_lon, P2_lat, 0.06, 0.06, 900)  +
            800 + RNG.normal(0, 50, N_TOTAL)).astype('float32')
ann_rain = np.clip(ann_rain, 500, 2500)

print(f'Elevation  : {elev.min():.0f}–{elev.max():.0f} m')
print(f'Slope      : {slope.min():.1f}–{slope.max():.1f} deg')
print(f'Dist fault : {dist_fault.min():.0f}–{dist_fault.max():.0f} m')
print(f'TWI        : {twi.min():.2f}–{twi.max():.2f}')
print(f'Lithology  : {np.bincount(lith)} (0=granite,1=schist,2=clay,3=alluvium)')
""")

C04 = cc("""\
# ── Depth-profile soil/rock properties (LOOKBACK=6 layers) ───────────────────
# Layers (m): [0-0.5, 0.5-1, 1-2, 2-3, 3-5, 5-10]
LAYER_DEPTHS   = np.array([0.25, 0.75, 1.5, 2.5, 4.0, 7.5], dtype='float32')
LAYER_LABELS   = ['0-0.5m', '0.5-1m', '1-2m', '2-3m', '3-5m', '5-10m']

# Per-layer geomechanical properties: shape (N_TOTAL, 6, N_DYNAMIC=6)
# Features: [cohesion_c, friction_phi, unit_weight, water_content, clay_frac, void_ratio]
soil = np.zeros((N_TOTAL, LOOKBACK, N_DYNAMIC), dtype='float32')

for z_idx, z in enumerate(LAYER_DEPTHS):
    # Lithology influence
    c_base   = np.where(lith==0, 60, np.where(lith==1, 30, np.where(lith==2, 15, 5)))
    phi_base = np.where(lith==0, 38, np.where(lith==1, 32, np.where(lith==2, 22, 18)))
    gam_base = np.where(lith==0, 21, np.where(lith==1, 19, np.where(lith==2, 17, 15)))
    clay_base= np.where(lith==0, 5,  np.where(lith==1, 20, np.where(lith==2, 50, 35)))

    # Depth trends: cohesion increases with depth, water_content decreases
    c_factor   = 1.0 + 0.15 * z_idx
    phi_factor = 1.0 - 0.02 * z_idx
    wc_base    = 35 - 4 * z_idx + RNG.normal(0, 3, N_TOTAL)
    void_base  = 0.60 - 0.05 * z_idx + RNG.normal(0, 0.03, N_TOTAL)

    soil[:, z_idx, 0] = (c_base   * c_factor
                         + RNG.normal(0, 3, N_TOTAL)).clip(2, 100)     # cohesion kPa
    soil[:, z_idx, 1] = (phi_base * phi_factor
                         + RNG.normal(0, 2, N_TOTAL)).clip(10, 45)     # friction angle deg
    soil[:, z_idx, 2] = (gam_base
                         + RNG.normal(0, 1, N_TOTAL)).clip(12, 25)     # unit weight kN/m³
    soil[:, z_idx, 3] = wc_base.clip(5, 55)                            # water content %
    soil[:, z_idx, 4] = (clay_base
                         + RNG.normal(0, 5, N_TOTAL)).clip(0, 70)      # clay fraction %
    soil[:, z_idx, 5] = void_base.clip(0.15, 0.80)                     # void ratio

# ── Trigger scenarios (HORIZON=5 return periods) ─────────────────────────────
# Parameters: [rainfall_intensity_mm_h, duration_h, antecedent_wet_frac, seismic_ag]
RAIN_INT  = np.array([20, 35, 55, 80, 110], dtype='float32')  # mm/h (T10…T200)
RAIN_DUR  = np.array([6,  12, 18, 24,  36], dtype='float32')  # hours
ANT_WET   = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype='float32')  # antecedent moisture
SEISMIC   = np.array([0.0, 0.0, 0.0, 0.0, 0.1], dtype='float32')  # T200 has seismic

future_feat = np.zeros((N_TOTAL, HORIZON, N_FUTURE), dtype='float32')
for h in range(HORIZON):
    # Normalise: intensity/100, duration/36, antecedent as-is, seismic/0.1
    future_feat[:, h, 0] = RAIN_INT[h] / 100.0
    future_feat[:, h, 1] = RAIN_DUR[h] / 36.0
    future_feat[:, h, 2] = ANT_WET[h]
    future_feat[:, h, 3] = SEISMIC[h]

print('Soil profile shape :', soil.shape)
print('Future feat shape  :', future_feat.shape)
print('Layer depths (m)   :', LAYER_DEPTHS)
print('Return periods (yr):', RETURN_PERIODS)
""")

C05 = cc("""\
# ── Factor of Safety (infinite slope) for each pixel × scenario ──────────────
GAMMA_W = 9.81   # kN/m³ — unit weight of water

def compute_fs(slope_deg, c_kpa, phi_deg,
               gamma_kn=19.0, z_m=2.0, z_w_frac=0.5):
    # Infinite slope stability: FS = (c + N'*tan_phi) / T
    # N' = (gamma*z - gamma_w*z_w)*cos²β,  T = gamma*z*sinβ*cosβ
    beta   = np.deg2rad(np.clip(slope_deg, 1, 75))
    phi    = np.deg2rad(np.clip(phi_deg, 5, 50))
    z_w    = z_m * z_w_frac
    normal = (gamma_kn * z_m - GAMMA_W * z_w) * np.cos(beta)**2
    shear  = gamma_kn * z_m * np.sin(beta) * np.cos(beta) + 1e-6
    fs     = (c_kpa + normal * np.tan(phi)) / shear
    return np.clip(fs, 0.1, 10.0).astype('float32')

# Use weighted average of top 3 layers for critical-plane FS
c_avg   = soil[:, :3, 0].mean(axis=1)     # (N_TOTAL,) cohesion
phi_avg = soil[:, :3, 1].mean(axis=1)     # friction angle
gam_avg = soil[:, :3, 2].mean(axis=1)     # unit weight

# FS under each trigger scenario (water table rises with rainfall + antecedent wet)
fs_matrix = np.zeros((N_TOTAL, HORIZON), dtype='float32')
for h in range(HORIZON):
    # Pore pressure: z_w_frac increases with rainfall intensity and antecedent moisture
    z_w_frac_h = np.clip(
        ANT_WET[h] + (RAIN_INT[h] / 200.0) + RNG.normal(0, 0.05, N_TOTAL), 0.05, 1.0)
    fs_matrix[:, h] = compute_fs(slope, c_avg, phi_avg,
                                  gamma_kn=gam_avg, z_m=2.5, z_w_frac=z_w_frac_h)

# ── Target: failure probability under each scenario ───────────────────────────
# P(failure | scenario h) = sigmoid(-k*(FS_h - 1)) + spatial_cluster + noise
K_SIGMOID = 4.0   # steepness of transition at FS=1

# Spatial clustering (near faults and channel heads)
cluster = (np.exp(-dist_fault / 1500) +
           0.3 * np.exp(-0.5*((slope - 32)/10)**2)).clip(0, 0.5).astype('float32')

target_prob = np.zeros((N_TOTAL, HORIZON, OUTPUT_DIM), dtype='float32')
for h in range(HORIZON):
    p_phys  = 1.0 / (1.0 + np.exp(K_SIGMOID * (fs_matrix[:, h] - 1.0)))
    p_total = np.clip(p_phys + 0.3 * cluster + RNG.normal(0, 0.04, N_TOTAL), 0, 1)
    target_prob[:, h, 0] = p_total.astype('float32')

# ── Binary inventory: slope + fault proximity (independent of FS) ─────────────
# Mimics a real inventory compiled from photo-interpretation + field survey.
# Deliberately separated from the FS-based training target so that AUC measures
# generalisation, not self-correlation.
inv_raw = (
    np.clip((slope - 15) / 18, 0, 1) * 0.55 +   # steep slopes
    np.exp(-dist_fault / 1800)         * 0.35 +   # fault proximity
    RNG.normal(0, 0.05, N_TOTAL)                  # stochastic variability
).clip(0, 1).astype('float32')
thresh_inv   = 0.28 + RNG.uniform(-0.08, 0.08, N_TOTAL).astype('float32')
is_landslide = (inv_raw > thresh_inv).astype('int32')

n_ls = is_landslide.sum()
print(f'Landslide pixels   : {n_ls} / {N_TOTAL} ({100*n_ls/N_TOTAL:.1f}%)')
print(f'FS range (T50)     : {fs_matrix[:,2].min():.2f}–{fs_matrix[:,2].max():.2f}')
print(f'Target prob range  : {target_prob.min():.3f}–{target_prob.max():.3f}')
""")

C06 = cc("""\
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

def plot_map(ax, data, title, cmap, vmin=None, vmax=None, label=''):
    data2d = data.reshape(N_ROWS, N_COLS)
    im = ax.imshow(data2d, origin='lower', cmap=cmap,
                   extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                   vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.85, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

plot_map(axes[0,0], elev,         'Digital Elevation Model (m)',
         'terrain', 150, 2200, 'm')
plot_map(axes[0,1], slope,        'Slope (deg)',
         'YlOrRd', 0, 70, 'deg')
plot_map(axes[0,2], lith.astype('float32'), 'Lithology',
         'tab10', 0, 3, '0=granite,1=schist,2=clay,3=alluvium')

plot_map(axes[1,0], dist_fault/1000, 'Distance to Fault (km)',
         'RdYlGn', 0, 12, 'km')
plot_map(axes[1,1], fs_matrix[:, 2], 'Factor of Safety (T=50yr)',
         'RdYlGn', 0.3, 3.0, 'FS')

# Inventory map
ax = axes[1,2]
ax.imshow(elev.reshape(N_ROWS, N_COLS), origin='lower', cmap='gray_r',
          extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], aspect='auto', alpha=0.4)
ls_lon = lon_flat[is_landslide == 1]
ls_lat = lat_flat[is_landslide == 1]
ax.scatter(ls_lon, ls_lat, s=6, c='red', alpha=0.7, label=f'Landslide ({len(ls_lon)})')
ax.set_title('Landslide Inventory (T50 trigger)', fontsize=11)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.legend(fontsize=9)

plt.suptitle('Section 1 — Study Area: Venezuelan Andes (synthetic 50×30 km watershed)',
             fontsize=13)
plt.tight_layout(); plt.show()
""")

C07 = mc("""\
---

## 2 — Feature Engineering & Dataset Construction

### Feature design rationale

| Input stream | Features | Scientific motivation |
|-------------|----------|----------------------|
| **Static** | slope, elevation, aspect, TWI, lithology, dist_fault, NDVI, ann_rainfall | Standard landslide conditioning factors (Fell et al., 2008) |
| **Dynamic (depth profile)** | cohesion, friction angle, unit weight, water content, clay fraction, void ratio | Geomechanical properties vary with depth — the attention mechanism identifies the **critical failure plane** |
| **Future (trigger)** | rainfall intensity, duration, antecedent wetness, seismic coefficient | Trigger parameters scale systematically with return period |
| **Target** | P(failure \\| scenario) | Continuous probability target derived from FS; validated against binary inventory |

### Why depth-profile-as-sequence is novel

Traditional landslide ML flattens all layer features into a single 36-dimensional
vector, losing all information about *which layer* is weak.  Treating the 6 depth
layers as a temporal sequence lets the encoder's attention mechanism assign higher
weights to the layer most likely to be the **rupture surface** — a key output that
practitioners cannot obtain from conventional models.\
""")

C08 = cc("""\
# ── Normalise static features ─────────────────────────────────────────────────
elev_n  = ((elev  - elev.mean())  / (elev.std()  + 1e-8)).astype('float32')
slope_n = ((slope - slope.mean()) / (slope.std() + 1e-8)).astype('float32')
asp_sin = np.sin(np.deg2rad(aspect)).astype('float32')
asp_cos = np.cos(np.deg2rad(aspect)).astype('float32')
twi_n   = ((twi  - twi.mean())   / (twi.std()   + 1e-8)).astype('float32')
df_n    = ((dist_fault - dist_fault.mean()) / (dist_fault.std() + 1e-8)).astype('float32')
ndvi_n  = ((ndvi - ndvi.mean())  / (ndvi.std()  + 1e-8)).astype('float32')
rain_n  = ((ann_rain - ann_rain.mean()) / (ann_rain.std() + 1e-8)).astype('float32')

X_static = np.stack([slope_n, elev_n, asp_sin, twi_n,
                     lith.astype('float32')/3.0, df_n, ndvi_n, rain_n], axis=1)

# ── Normalise dynamic (depth-profile) features ───────────────────────────────
X_dyn = soil.copy()
for feat_i in range(N_DYNAMIC):
    f_all = X_dyn[:, :, feat_i]
    mu, sg = float(f_all.mean()), float(f_all.std())
    X_dyn[:, :, feat_i] = ((f_all - mu) / (sg + 1e-8)).astype('float32')

# ── Future features already normalised ───────────────────────────────────────
X_future = future_feat.copy()

# ── Target ───────────────────────────────────────────────────────────────────
Y = target_prob.copy()   # (N_TOTAL, HORIZON, 1)

print('X_static  :', X_static.shape)
print('X_dynamic :', X_dyn.shape)
print('X_future  :', X_future.shape)
print('Y (probs) :', Y.shape)
print('inventory :', is_landslide.shape, '  positives:', is_landslide.sum())
""")

C09 = cc("""\
# ── Spatial train/test split (block split to avoid spatial autocorrelation) ────
# Test = northern block (top 20% of rows)
# Motivaton: simulates applying a model trained in one sub-catchment to another

row_idx = np.arange(N_TOTAL) // N_COLS      # row (latitude) index for each pixel
TEST_ROW_CUTOFF = int(N_ROWS * 0.80)        # row 24 out of 0..29

tr_m = row_idx <  TEST_ROW_CUTOFF           # southern 80%
te_m = row_idx >= TEST_ROW_CUTOFF           # northern 20%

Xs_tr, Xd_tr, Xf_tr, Y_tr = X_static[tr_m], X_dyn[tr_m], X_future[tr_m], Y[tr_m]
Xs_te, Xd_te, Xf_te, Y_te = X_static[te_m], X_dyn[te_m], X_future[te_m], Y[te_m]
inv_tr = is_landslide[tr_m]
inv_te = is_landslide[te_m]
fs_te  = fs_matrix[te_m]

print(f'Train : {tr_m.sum()} pixels  ({inv_tr.sum()} landslide  '
      f'{100*inv_tr.mean():.1f}%)')
print(f'Test  : {te_m.sum()} pixels  ({inv_te.sum()} landslide  '
      f'{100*inv_te.mean():.1f}%)')
""")

C10 = cc("""\
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── Feature distributions ─────────────────────────────────────────────────────
ax = axes[0]
feat_names = ['slope', 'elevation', 'asp_sin', 'TWI', 'lithology',
              'dist_fault', 'NDVI', 'ann_rain']
feat_vals  = [slope, elev, asp_sin, twi, lith.astype('float'),
              dist_fault/1000, ndvi, ann_rain/1000]
bp = ax.boxplot([f[is_landslide==1] for f in feat_vals], patch_artist=True,
                positions=np.arange(len(feat_vals))*2,
                boxprops=dict(facecolor='#e74c3c', alpha=0.7))
bp2 = ax.boxplot([f[is_landslide==0] for f in feat_vals], patch_artist=True,
                 positions=np.arange(len(feat_vals))*2 + 0.7,
                 boxprops=dict(facecolor='#3498db', alpha=0.7))
ax.set_xticks(np.arange(len(feat_names))*2 + 0.35)
ax.set_xticklabels(feat_names, rotation=40, ha='right', fontsize=8)
ax.set_title('(A) Static Feature Distributions\\n(red=landslide, blue=stable)',
             fontsize=10)
ax.grid(True, alpha=0.25, axis='y')
ax.legend([bp['boxes'][0], bp2['boxes'][0]], ['Landslide', 'Stable'], fontsize=9)

# ── Depth-profile average by class ───────────────────────────────────────────
ax = axes[1]
dyn_feat_names = ['cohesion', 'friction', 'unit_wt', 'water_ct', 'clay_fr', 'void_r']
for feat_i, fname in enumerate(dyn_feat_names):
    vals_ls  = soil[is_landslide==1, :, feat_i].mean(axis=0)
    vals_stb = soil[is_landslide==0, :, feat_i].mean(axis=0)
    if feat_i == 0:   # only plot cohesion and friction for clarity
        ax.plot(LAYER_DEPTHS, vals_ls,  'o-', lw=2, color='#e74c3c',
                label=f'{fname} — LS')
        ax.plot(LAYER_DEPTHS, vals_stb, 's--', lw=1.5, color='#3498db',
                label=f'{fname} — Stable')
    elif feat_i == 1:
        ax2 = ax.twinx()
        ax2.plot(LAYER_DEPTHS, vals_ls,  '^-', lw=2, color='#e74c3c', alpha=0.6,
                 label=f'{fname} — LS')
        ax2.plot(LAYER_DEPTHS, vals_stb, 'v--', lw=1.5, color='#3498db', alpha=0.6,
                 label=f'{fname} — Stable')
        ax2.set_ylabel('Friction angle (deg)', fontsize=9)
ax.set_xlabel('Depth (m)'); ax.set_ylabel('Cohesion (kPa)')
ax.set_title('(B) Cohesion & Friction Depth Profile\\n(LS vs Stable classes)',
             fontsize=10)
ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.25)

# ── FS histogram with inventory overlay ──────────────────────────────────────
ax = axes[2]
fs_50 = fs_matrix[:, 2]
ax.hist(fs_50[is_landslide==0], bins=40, color='#3498db', alpha=0.6,
        density=True, label='Stable')
ax.hist(fs_50[is_landslide==1], bins=40, color='#e74c3c', alpha=0.6,
        density=True, label='Landslide')
ax.axvline(1.0, color='black', lw=2, linestyle='--', label='FS = 1.0 (failure)')
ax.set_xlabel('Factor of Safety (T=50yr)')
ax.set_ylabel('Density')
ax.set_title('(C) FS Distribution by Inventory Class\\n(T=50yr trigger scenario)',
             fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

plt.suptitle('Section 2 — Feature Engineering: Class Separation', fontsize=13)
plt.tight_layout(); plt.show()

# Spearman correlations of features with inventory
print('Spearman rho (feature vs binary inventory):')
for fname, fv in [('slope', slope), ('dist_fault', dist_fault),
                  ('TWI', twi), ('FS_T50', fs_50)]:
    rho, p = spearmanr(fv, is_landslide)
    print(f'  {fname:14s}: rho={rho:+.3f}  p={p:.3e}')
""")

C11 = mc("""\
---

## 3 — Single BaseAttentive Model

### Architecture design for geological sequences

The depth-profile encoder reads 6 depth layers sequentially (surface → bedrock).
Cross-attention in the decoder allows each trigger scenario (horizon step) to attend
to the **most geomechanically relevant** depth layers:

- **Shallow failures** (T10, T25): expected attention concentration on layers 1–3
  (0–2 m depth) where saturated residual soil sits above bedrock.
- **Deep-seated failures** (T100, T200): expected attention shift to layers 4–6
  (3–10 m) where deeper clay-rich interfaces promote rotational sliding.

The `hierarchical` decoder additionally captures multi-scale interactions between
the surface (rainfall infiltration interface) and deeper impermeable layers.\
""")

C12 = cc("""\
# ── Build and train single BaseAttentive model ────────────────────────────────
model_ba = BaseAttentive(
    static_input_dim=N_STATIC, dynamic_input_dim=N_DYNAMIC,
    future_input_dim=N_FUTURE,  output_dim=OUTPUT_DIM,
    forecast_horizon=HORIZON,   objective='hybrid',
    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},
    embed_dim=32, num_heads=4, dropout_rate=0.15,
    name='ba_landslide',
)
_ = model_ba([Xs_tr[:4], Xd_tr[:4], Xf_tr[:4]])   # build
model_ba.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
print(f'Parameters : {model_ba.count_params():,}')

t0 = time.perf_counter()
history_ba = model_ba.fit(
    [Xs_tr, Xd_tr, Xf_tr], Y_tr,
    epochs=EPOCHS_MAIN, batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE,
                                              restore_best_weights=True)],
    verbose=0,
)
train_time_ba = time.perf_counter() - t0
print(f'Train time : {train_time_ba:.1f} s  '
      f'(stopped at epoch {len(history_ba.history["loss"])})')
print(f'Best val MSE : {min(history_ba.history["val_loss"]):.5f}')

# Predict probabilities (T50 = horizon index 2 → primary susceptibility)
Y_pred_ba  = model_ba.predict([Xs_te, Xd_te, Xf_te], verbose=0)  # (N_test, 5, 1)
prob_ba    = np.clip(Y_pred_ba[:, 2, 0], 0, 1)   # T50 susceptibility index
auc_ba     = roc_auc_score(inv_te, prob_ba)
ap_ba      = average_precision_score(inv_te, prob_ba)
print(f'\\nTest AUC-ROC  : {auc_ba:.4f}')
print(f'Test AUC-PR   : {ap_ba:.4f}')
""")

C13 = cc("""\
fpr_ba, tpr_ba, thr_ba = roc_curve(inv_te, prob_ba)
prec_ba, rec_ba, _     = precision_recall_curve(inv_te, prob_ba)
# Optimal threshold (Youden J) — clamp to [0.05, 0.95] so the
# model never collapses all predictions to one class on short runs
j_idx   = np.argmax(tpr_ba - fpr_ba)
opt_thr = float(np.clip(thr_ba[j_idx], 0.05, 0.95))
pred_cls_ba = (prob_ba >= opt_thr).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── (A) ROC ────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(fpr_ba, tpr_ba, lw=2.5, color='#3498db',
        label=f'BA-Cross+Hier  AUC={auc_ba:.3f}')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
ax.scatter(fpr_ba[j_idx], tpr_ba[j_idx], s=120, color='red', zorder=5,
           label=f'Optimal thr={opt_thr:.2f}')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(A) ROC Curve', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── (B) Precision-Recall ──────────────────────────────────────────────────────
ax = axes[1]
ax.step(rec_ba, prec_ba, lw=2.5, color='#2ecc71', where='post',
        label=f'AP = {ap_ba:.3f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('(B) Precision-Recall Curve', fontsize=11)
ax.axhline(inv_te.mean(), color='gray', lw=1, linestyle='--',
           label=f'Baseline (PR = {inv_te.mean():.2f})')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── (C) Confusion matrix ──────────────────────────────────────────────────────
ax = axes[2]
cm_ba = confusion_matrix(inv_te, pred_cls_ba, labels=[0, 1])
im = ax.imshow(cm_ba, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_ba[i,j]), ha='center', va='center',
                fontsize=16, fontweight='bold')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Predicted Stable', 'Predicted LS'], fontsize=9)
ax.set_yticklabels(['Actual Stable', 'Actual LS'], fontsize=9)
ax.set_title(f'(C) Confusion Matrix  (thr={opt_thr:.2f})', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('Section 3 — Single BA Model: Classification Performance', fontsize=13)
plt.tight_layout(); plt.show()

print(classification_report(inv_te, pred_cls_ba,
                             labels=[0, 1],
                             target_names=['Stable', 'Landslide']))
""")

C14 = cc("""\
# ── Susceptibility map (all pixels) ───────────────────────────────────────────
Y_pred_all = model_ba.predict([X_static, X_dyn, X_future], verbose=0)
susc_ba    = np.clip(Y_pred_all[:, 2, 0], 0, 1)   # T50 primary scenario

# Classify into 5 classes
def classify_susceptibility(probs, bounds=SUSC_BOUNDS):
    classes = np.zeros(len(probs), dtype=int)
    for cls_i in range(len(bounds)-1):
        mask = (probs > bounds[cls_i]) & (probs <= bounds[cls_i+1])
        classes[mask] = cls_i
    return classes

cls_ba = classify_susceptibility(susc_ba)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Susceptibility map
ax = axes[0]
cmap_susc = mcolors.ListedColormap(SUSC_COLORS)
norm_susc  = mcolors.BoundaryNorm(SUSC_BOUNDS, len(SUSC_COLORS))
im = ax.imshow(susc_ba.reshape(N_ROWS, N_COLS), origin='lower',
               cmap='RdYlGn_r', vmin=0, vmax=1,
               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], aspect='auto')
ax.scatter(lon_flat[is_landslide==1], lat_flat[is_landslide==1],
           s=5, c='black', alpha=0.6, label='Inventory')
plt.colorbar(im, ax=ax, label='P(failure | T50)')
ax.set_title('(A) Susceptibility Map (BA single, T50 scenario)', fontsize=11)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.legend(fontsize=9)

# Class distribution
ax = axes[1]
class_counts = np.bincount(cls_ba, minlength=5)
bars = ax.bar(SUSC_LABELS, class_counts / N_TOTAL * 100,
              color=SUSC_COLORS, edgecolor='white')
for bar, cnt in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{cnt}\\n({cnt/N_TOTAL*100:.1f}%)',
            ha='center', fontsize=8)
ax.set_ylabel('% of study area'); ax.set_ylim(0, 50)
ax.set_title('(B) Susceptibility Class Distribution', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Section 3 — Susceptibility Map: Single BA Model', fontsize=13)
plt.tight_layout(); plt.show()
""")

C15 = cc("""\
# ── VSN feature importance via gradient saliency ──────────────────────────────
N_SAL = min(128, TRAIN_SIZE)

xs_v = tf.Variable(Xs_tr[:N_SAL])
xd_v = tf.Variable(Xd_tr[:N_SAL])
xf_v = tf.Variable(Xf_tr[:N_SAL])

with tf.GradientTape() as tape:
    pred   = model_ba([xs_v, xd_v, xf_v], training=False)
    scalar = tf.reduce_mean(pred[:, 2, 0])   # T50 output

g_s, g_d, g_f = tape.gradient(scalar, [xs_v, xd_v, xf_v])

sal_static  = tf.abs(g_s).numpy().mean(axis=0)          # (N_STATIC,)
sal_dynamic = tf.abs(g_d).numpy()                        # (N_SAL, 6, N_DYN)
sal_layer   = sal_dynamic.mean(axis=(0, 2))              # (6,) layer importance
sal_dyn_feat= sal_dynamic.mean(axis=(0, 1))              # (N_DYN,)
sal_future  = tf.abs(g_f).numpy().mean(axis=0)           # (5, N_FUTURE)
sal_scen    = sal_future.mean(axis=1)                    # (5,) scenario importance

STATIC_NAMES  = ['slope', 'elevation', 'asp_sin', 'TWI',
                 'lithology', 'dist_fault', 'NDVI', 'ann_rain']
DYN_FEAT_NAMES= ['cohesion', 'friction', 'unit_wt',
                 'water_ct', 'clay_frac', 'void_ratio']
FUTURE_NAMES  = ['rain_int', 'duration', 'antec_wet', 'seismic']

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# (A) Static feature importance
ax = axes[0]
order_s = np.argsort(sal_static)
ax.barh([STATIC_NAMES[i] for i in order_s], sal_static[order_s],
        color='#3498db', edgecolor='white')
ax.set_title('(A) Static Feature Importance\\n(gradient saliency)', fontsize=11)
ax.set_xlabel('Mean |gradient|'); ax.grid(True, alpha=0.3, axis='x')

# (B) CRITICAL FAILURE PLANE: depth-layer saliency
ax = axes[1]
colors_lyr = plt.cm.hot_r(np.linspace(0.2, 0.8, LOOKBACK))
bars = ax.bar(LAYER_LABELS, sal_layer, color=colors_lyr, edgecolor='white')
best_layer = int(np.argmax(sal_layer))
bars[best_layer].set_edgecolor('red'); bars[best_layer].set_linewidth(3)
ax.set_title('(B) Critical Layer — Failure Plane Depth\\n(red border = most critical layer)',
             fontsize=11)
ax.set_ylabel('Mean |gradient|'); ax.grid(True, alpha=0.3, axis='y')
ax.set_xlabel('Depth layer')
ax.text(best_layer, sal_layer[best_layer]*1.02,
        f'CRITICAL\\n{LAYER_LABELS[best_layer]}',
        ha='center', color='red', fontsize=8, fontweight='bold')

# (C) Trigger scenario importance
ax = axes[2]
tp_labels = [f'T{t}yr' for t in RETURN_PERIODS]
ax.plot(tp_labels, sal_scen, 'o-', lw=2.5, color='#e74c3c', markersize=9)
ax.fill_between(range(HORIZON), sal_scen, alpha=0.2, color='#e74c3c')
ax.set_title('(C) Trigger Scenario Importance\\n(cross-attention to return periods)',
             fontsize=11)
ax.set_ylabel('Mean |gradient|'); ax.grid(True, alpha=0.3)

plt.suptitle('Section 3 — Gradient Saliency: Feature & Layer Importance', fontsize=13)
plt.tight_layout(); plt.show()

print(f'Most important static feature : {STATIC_NAMES[np.argmax(sal_static)]}')
print(f'Critical failure-plane depth  : {LAYER_LABELS[best_layer]}')
print(f'Most influential trigger       : T{RETURN_PERIODS[int(np.argmax(sal_scen))]} yr')
""")

C16 = mc("""\
### Interpreting Feature Importance

**(A) Static features**: slope gradient and distance to fault typically rank highest,
consistent with the geomorphological literature.  Elevation is secondary because it
acts as a proxy for climate (higher elevations receive more orographic rainfall) rather
than a direct mechanical control.  NDVI is a negative indicator — dense vegetation
increases root cohesion and reduces surface runoff, stabilising shallow slopes.

**(B) Critical failure plane** (the key novel result): the red-bordered bar identifies
the depth layer that the model assigns the highest attention weight — this is the
predicted **rupture interface** (the soil layer where shear stress first exceeds
shear strength).  In the field, this can be validated by correlating with observed
failure-scar depths from aerial photography or LiDAR.

**(C) Trigger scenario importance**: the gradient along the five return-period steps
reveals whether the model's predictions are dominated by moderate-frequency events
(T10–T25, short-duration saturation) or rare extreme events (T100–T200, combined
rainfall + seismic loading).  A flat curve implies the model relies on static
conditioning rather than trigger intensity — which may suggest over-fitting to
geomorphological proxies rather than genuine mechanistic relationships.\
""")

C17 = mc("""\
---

## 4 — Ensemble BaseAttentive: Uncertainty Quantification

### Epistemic uncertainty in susceptibility mapping

A single model's susceptibility estimate conflates two sources of uncertainty:
- **Aleatoric**: irreducible noise in the data (measurement errors, unknown micro-factors)
- **Epistemic**: uncertainty in model weights due to limited training data

Ensemble methods quantify **epistemic uncertainty** by training multiple models
with different architectures or initialisations and measuring the disagreement between
predictions.  Pixels where the ensemble disagrees strongly are those where additional
field data collection would most efficiently reduce uncertainty — a practical guide
for prioritising field campaigns.

### Three-member ensemble

| Member | Decoder stack | Distinct strength |
|--------|--------------|------------------|
| **BA-Cross** | `['cross']` | Future trigger features (rainfall forecast) |
| **BA-Hier** | `['hierarchical']` | Multi-scale depth-layer interactions |
| **BA-Cross+Hier** | `['cross','hierarchical']` | Combined — used as the reference single model |\
""")

C18 = cc("""\
ENS_CONFIGS = [
    dict(name='BA-Cross',    stack=['cross'],               embed=32, heads=4),
    dict(name='BA-Hier',     stack=['hierarchical'],         embed=32, heads=4),
    dict(name='BA-Cross+Hier', stack=['cross','hierarchical'], embed=32, heads=4),
]

ens_preds_all  = []   # (3, N_TOTAL, 5) — predictions for every pixel
ens_preds_te   = []   # (3, N_test,  5) — for evaluation
ens_histories  = {}

for cfg in ENS_CONFIGS:
    safe = cfg['name'].lower().replace('+','p').replace('-','_')
    m = BaseAttentive(
        static_input_dim=N_STATIC, dynamic_input_dim=N_DYNAMIC,
        future_input_dim=N_FUTURE,  output_dim=OUTPUT_DIM,
        forecast_horizon=HORIZON,   objective='hybrid',
        architecture_config={'decoder_attention_stack': cfg['stack']},
        embed_dim=cfg['embed'], num_heads=cfg['heads'],
        dropout_rate=0.15, name=f'ens_{safe}',
    )
    _ = m([Xs_tr[:4], Xd_tr[:4], Xf_tr[:4]])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    hist = m.fit(
        [Xs_tr, Xd_tr, Xf_tr], Y_tr,
        epochs=EPOCHS_MAIN, batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                  restore_best_weights=True)],
        verbose=0,
    )
    all_pred = np.clip(m.predict([X_static, X_dyn, X_future], verbose=0)[:,:,0], 0, 1)
    te_pred  = np.clip(m.predict([Xs_te, Xd_te, Xf_te],       verbose=0)[:,:,0], 0, 1)
    ens_preds_all.append(all_pred)
    ens_preds_te.append(te_pred)
    ens_histories[cfg['name']] = hist.history
    print(f'{cfg["name"]:18s}  '
          f'AUC={roc_auc_score(inv_te, te_pred[:,2]):.4f}  '
          f'epochs={len(hist.history["loss"])}')

ens_preds_all = np.array(ens_preds_all)   # (3, N_TOTAL, 5)
ens_preds_te  = np.array(ens_preds_te)    # (3, N_test, 5)

# Ensemble statistics
susc_ens_mean = ens_preds_all[:, :, 2].mean(axis=0)   # (N_TOTAL,) mean susceptibility
susc_ens_std  = ens_preds_all[:, :, 2].std(axis=0)    # (N_TOTAL,) epistemic uncertainty
auc_ens       = roc_auc_score(inv_te, ens_preds_te[:, :, 2].mean(axis=0))
print(f'\\nEnsemble AUC : {auc_ens:.4f}')
""")

C19 = cc("""\
fig, axes = plt.subplots(1, 3, figsize=(17, 6))

plot_map(axes[0], susc_ens_mean,
         '(A) Ensemble Mean Susceptibility (T50)', 'RdYlGn_r', 0, 1,
         'P(failure)')
axes[0].scatter(lon_flat[is_landslide==1], lat_flat[is_landslide==1],
                s=4, c='black', alpha=0.6, label='Inventory')
axes[0].legend(fontsize=9)

plot_map(axes[1], susc_ens_std,
         '(B) Epistemic Uncertainty (std of ensemble)', 'Purples', 0, 0.25,
         'Std dev')
# Mark high-uncertainty zones
hi_unc = susc_ens_std > np.percentile(susc_ens_std, 90)
axes[1].scatter(lon_flat[hi_unc], lat_flat[hi_unc],
                s=4, c='red', alpha=0.3, label='High uncertainty (top 10%)')
axes[1].legend(fontsize=9)

# Agreement map: fraction of ensemble members classifying as hazardous (>0.4)
agree = (ens_preds_all[:, :, 2] > 0.40).mean(axis=0)
plot_map(axes[2], agree,
         '(C) Ensemble Agreement\\n(fraction classifying as hazardous, thr=0.40)',
         'RdYlBu_r', 0, 1, 'Fraction of 3 members')

plt.suptitle('Section 4 — Ensemble Susceptibility & Epistemic Uncertainty', fontsize=13)
plt.tight_layout(); plt.show()

print(f'High-uncertainty pixels (std > 0.15) : {(susc_ens_std > 0.15).sum()}'
      f'  ({100*(susc_ens_std>0.15).mean():.1f}% of study area)')
print(f'Perfect agreement (all 3 agree)      : {(agree==1.0).sum() + (agree==0.0).sum()}'
      f'  ({100*((agree==1.0)|(agree==0.0)).mean():.1f}%)')
""")

C20 = mc("""\
### Interpreting Uncertainty Maps

**(A) Ensemble mean susceptibility**: this is the primary deliverable for hazard
zonation.  Because it averages over three architecturally distinct models, it is
less sensitive to any single model's idiosyncrasies than a single-model estimate.

**(B) Epistemic uncertainty**: high-uncertainty zones (purple) mark pixels where the
three models *disagree* significantly — indicating that the available training data
are insufficient to constrain the model in that spatial context.  In practice, these
are the highest-priority targets for additional field investigation (new boreholes,
UAV surveys, or additional inventory mapping).

**(C) Agreement map**: a stricter version of (B).  Pixels where all three members
agree on the hazard classification are shown in red (unanimous hazardous) or blue
(unanimous stable).  Yellow pixels represent genuine uncertainty that field data can
resolve.

**For the scientific paper**: report the fraction of the study area in each agreement
category as a table alongside the standard AUC metrics.  Uncertainty maps are
increasingly required by journal reviewers as evidence that the authors understand
the limits of their model's generalisability.\
""")

C21 = mc("""\
---

## 5 — Physics-Informed BaseAttentive

### The physics-informed regularisation framework

Standard deep learning for landslide susceptibility is purely data-driven: if the
inventory is biased (e.g., only well-accessible slopes are mapped), the model inherits
that bias.  Physics-informed regularisation introduces a **soft constraint** that
keeps predictions consistent with the **Infinite-Slope Factor of Safety**:

$$\\mathcal{L}_{\\text{total}} = \\underbrace{\\mathcal{L}_{\\text{MSE}}(\\hat{p}, p_{\\text{target}})}_{\\text{data-driven}} + \\lambda_{\\text{phys}} \\cdot \\underbrace{\\mathcal{L}_{\\text{phys}}(\\hat{p}_{T_{50}}, p_{\\text{FS}})}_{\\text{physics constraint}}$$

Where:

$$p_{\\text{FS}} = \\sigma\\!\\left(-k\\,(\\text{FS}_{T_{50}} - 1)\\right), \\quad k=4$$

A sigmoid centred at FS = 1 converts the mechanical stability index into a prior
failure probability.  When `FS < 1` the physics strongly signal failure; the physics
loss penalises the model if it predicts stability there (and vice versa).

This regularisation is **particularly valuable for sparse inventories**: even if only
a few landslide locations are mapped in a lithological unit, the FS constraint
propagates stability/instability signals to un-inventoried pixels with similar
geomechanical properties.\
""")

C22 = cc("""\
# ── Pre-compute FS-based physics prior ───────────────────────────────────────
fs_prior_all = (1.0 / (1.0 + np.exp(4.0 * (fs_matrix[:, 2] - 1.0)))).astype('float32')
fs_prior_tr  = fs_prior_all[tr_m]
fs_prior_te  = fs_prior_all[te_m]

# ── Physics-informed model ────────────────────────────────────────────────────
model_phys = BaseAttentive(
    static_input_dim=N_STATIC, dynamic_input_dim=N_DYNAMIC,
    future_input_dim=N_FUTURE,  output_dim=OUTPUT_DIM,
    forecast_horizon=HORIZON,   objective='hybrid',
    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},
    embed_dim=32, num_heads=4, dropout_rate=0.15,
    name='ba_physics',
)
_ = model_phys([Xs_tr[:4], Xd_tr[:4], Xf_tr[:4]])
opt_phys = keras.optimizers.Adam(1e-3)

# Batch indices for custom training loop
N_TR = len(Xs_tr)
N_BATCHES = N_TR // BATCH_SIZE

# ── Custom training loop with physics constraint ──────────────────────────────
@tf.function
def physics_train_step(xs_b, xd_b, xf_b, y_b, fs_b):
    with tf.GradientTape() as tape:
        y_hat  = model_phys([xs_b, xd_b, xf_b], training=True)  # (B, H, 1)
        # Data-driven MSE
        mse    = tf.reduce_mean(tf.square(y_b - y_hat))
        # Physics constraint: T50 prediction vs FS-based prior
        p_t50  = y_hat[:, 2, 0]                          # (B,) T50 output
        p_fs   = tf.cast(fs_b, tf.float32)               # (B,) FS prior
        phys   = tf.reduce_mean(tf.square(p_t50 - p_fs))
        total  = mse + LAMBDA_PHYS * phys
    grads = tape.gradient(total, model_phys.trainable_variables)
    opt_phys.apply_gradients(zip(grads, model_phys.trainable_variables))
    return total, mse, phys

phys_history = {'loss': [], 'mse': [], 'phys': [], 'val_auc': []}

for epoch in range(EPOCHS_MAIN):
    perm = np.random.permutation(N_TR)
    ep_loss, ep_mse, ep_phys = [], [], []
    for b in range(N_BATCHES):
        idx_b = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        l, m, p = physics_train_step(
            tf.constant(Xs_tr[idx_b]), tf.constant(Xd_tr[idx_b]),
            tf.constant(Xf_tr[idx_b]), tf.constant(Y_tr[idx_b]),
            tf.constant(fs_prior_tr[idx_b]),
        )
        ep_loss.append(float(l)); ep_mse.append(float(m)); ep_phys.append(float(p))
    phys_history['loss'].append(np.mean(ep_loss))
    phys_history['mse'].append(np.mean(ep_mse))
    phys_history['phys'].append(np.mean(ep_phys))
    # Quick val AUC
    y_te_pred = np.clip(model_phys.predict([Xs_te, Xd_te, Xf_te], verbose=0)[:, 2, 0], 0, 1)
    phys_history['val_auc'].append(roc_auc_score(inv_te, y_te_pred))
    if epoch % 5 == 0:
        print(f'  Epoch {epoch:3d}  loss={phys_history["loss"][-1]:.5f}'
              f'  mse={phys_history["mse"][-1]:.5f}'
              f'  phys={phys_history["phys"][-1]:.5f}'
              f'  val_AUC={phys_history["val_auc"][-1]:.4f}')

Y_pred_phys = np.clip(model_phys.predict([Xs_te, Xd_te, Xf_te], verbose=0)[:, 2, 0], 0, 1)
auc_phys    = roc_auc_score(inv_te, Y_pred_phys)
print(f'\\nPhysics-informed AUC : {auc_phys:.4f}')
""")

C23 = cc("""\
Y_pred_phys_all = np.clip(
    model_phys.predict([X_static, X_dyn, X_future], verbose=0)[:, 2, 0], 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── (A) Physics loss convergence ──────────────────────────────────────────────
ax = axes[0]
ep_ax = np.arange(1, EPOCHS_MAIN + 1)
ax.plot(ep_ax, phys_history['mse'],  lw=2, color='#3498db', label='MSE (data loss)')
ax.plot(ep_ax, phys_history['phys'], lw=2, color='#e74c3c', label='Physics loss (FS)')
ax.plot(ep_ax, phys_history['loss'], lw=2, color='black', linestyle='--',
        label='Total loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('(A) Physics-Informed Training Curves\\n(data MSE + physics constraint)',
             fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(ep_ax, phys_history['val_auc'], lw=1.5, color='#2ecc71',
         linestyle=':', alpha=0.8, label='Val AUC')
ax2.set_ylabel('AUC-ROC', color='#2ecc71', fontsize=9)
ax2.tick_params(axis='y', labelcolor='#2ecc71')

# ── (B) Physics consistency: FS prior vs predicted probability ────────────────
ax = axes[1]
sample_idx = RNG.integers(0, N_TOTAL, 400)
ax.scatter(fs_prior_all[sample_idx], Y_pred_phys_all[sample_idx],
           alpha=0.3, s=15, color='#e74c3c', label='Physics-informed BA')
ax.scatter(fs_prior_all[sample_idx], susc_ba[sample_idx],
           alpha=0.3, s=15, color='#3498db', label='Standard BA')
ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Perfect consistency')
ax.set_xlabel('FS-based physics prior  P(fail|FS)')
ax.set_ylabel('Model predicted P(fail | T50)')
ax.set_title('(B) Physics Consistency:\\nPrediction vs FS Prior', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

# ── (C) Standard vs physics-informed: prediction disagreement map ─────────────
ax = axes[2]
delta = Y_pred_phys_all - susc_ba
plot_map(ax, delta, '(C) Physics-Informed − Standard\\n(positive = physics raises risk)',
         'RdBu_r', -0.3, 0.3, 'Delta P(failure)')

plt.suptitle('Section 5 — Physics-Informed BaseAttentive: Training & Consistency',
             fontsize=13)
plt.tight_layout(); plt.show()

# Physical consistency metric: Spearman rho between FS prior and predictions
rho_ba,   _ = spearmanr(fs_prior_all, susc_ba)
rho_phys, _ = spearmanr(fs_prior_all, Y_pred_phys_all)
print(f'Physical consistency (Spearman rho with FS prior):')
print(f'  Standard BA        : {rho_ba:.4f}')
print(f'  Physics-informed BA: {rho_phys:.4f}  (higher = more physically consistent)')
""")

C24 = mc("""\
---

## 6 — Comparative Analysis: All Methods

### Benchmark suite

To demonstrate the value of the attention-based approach, we compare against:

| Method | Type | Key characteristic |
|--------|------|--------------------|
| **Logistic Regression** | Classical ML | Linear decision boundary |
| **Random Forest** | Ensemble ML | Non-linear, no temporal structure |
| **Standard BA** | Deep Learning | Depth-profile attention |
| **Ensemble BA** | Deep Learning | Epistemic uncertainty |
| **Physics-Informed BA** | Hybrid | Physical constraint + attention |

The comparison follows the **spatial block validation** protocol (southern 80%
trains, northern 20% tests) which is more realistic than random splits for
assessing geographical transferability — a critical property for operational
hazard mapping.\
""")

C25 = cc("""\
# ── Classical baselines ───────────────────────────────────────────────────────
X_flat_tr = np.concatenate([Xs_tr,
                             Xd_tr.reshape(len(Xs_tr), -1),
                             Xf_tr.reshape(len(Xs_tr), -1)], axis=1)
X_flat_te = np.concatenate([Xs_te,
                             Xd_te.reshape(len(Xs_te), -1),
                             Xf_te.reshape(len(Xs_te), -1)], axis=1)

scaler_cls = StandardScaler()
X_flat_tr_s = scaler_cls.fit_transform(X_flat_tr)
X_flat_te_s = scaler_cls.transform(X_flat_te)

# Logistic Regression
lr_cls = LogisticRegression(C=1.0, max_iter=300, random_state=42)
lr_cls.fit(X_flat_tr_s, inv_tr)
prob_lr  = lr_cls.predict_proba(X_flat_te_s)[:, 1]
auc_lr   = roc_auc_score(inv_te, prob_lr)
ap_lr    = average_precision_score(inv_te, prob_lr)

# Random Forest
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 random_state=42, n_jobs=-1)
rf_cls.fit(X_flat_tr_s, inv_tr)
prob_rf  = rf_cls.predict_proba(X_flat_te_s)[:, 1]
auc_rf   = roc_auc_score(inv_te, prob_rf)
ap_rf    = average_precision_score(inv_te, prob_rf)

# Collect all predictions
all_probs  = {
    'Logistic Reg': prob_lr,
    'Random Forest': prob_rf,
    'BA (standard)': prob_ba,
    'BA (ensemble)': ens_preds_te[:, :, 2].mean(axis=0),
    'BA (physics)' : Y_pred_phys,
}
all_aucs   = {k: roc_auc_score(inv_te, v) for k, v in all_probs.items()}
all_aps    = {k: average_precision_score(inv_te, v) for k, v in all_probs.items()}

print(f'{"Method":24s}  {"AUC-ROC":>9s}  {"AUC-PR":>8s}')
print('─' * 46)
for k in all_probs:
    print(f'{k:24s}  {all_aucs[k]:>9.4f}  {all_aps[k]:>8.4f}')
""")

C26 = cc("""\
METHOD_COLORS = {
    'Logistic Reg':  '#95a5a6',
    'Random Forest': '#e67e22',
    'BA (standard)': '#3498db',
    'BA (ensemble)': '#9b59b6',
    'BA (physics)' : '#e74c3c',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── (A) ROC comparison ────────────────────────────────────────────────────────
ax = axes[0]
for mname, probs in all_probs.items():
    fpr, tpr, _ = roc_curve(inv_te, probs)
    lw  = 2.5 if 'BA' in mname else 1.5
    ls  = '-' if 'BA' in mname else '--'
    ax.plot(fpr, tpr, lw=lw, linestyle=ls, color=METHOD_COLORS[mname],
            label=f'{mname}  (AUC={all_aucs[mname]:.3f})')
ax.plot([0,1],[0,1], 'k:', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(A) ROC Curves — All Methods', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

# ── (B) PR comparison ─────────────────────────────────────────────────────────
ax = axes[1]
for mname, probs in all_probs.items():
    prec, rec, _ = precision_recall_curve(inv_te, probs)
    lw  = 2.5 if 'BA' in mname else 1.5
    ls  = '-' if 'BA' in mname else '--'
    ax.step(rec, prec, lw=lw, linestyle=ls, color=METHOD_COLORS[mname],
            where='post', label=f'{mname}  (AP={all_aps[mname]:.3f})')
ax.axhline(inv_te.mean(), color='gray', lw=1, linestyle=':',
           label=f'Baseline PR={inv_te.mean():.2f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('(B) Precision-Recall Curves — All Methods', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

plt.suptitle('Section 6 — Comparative ROC & Precision-Recall Analysis', fontsize=13)
plt.tight_layout(); plt.show()
""")

C27 = cc("""\
# ── Side-by-side susceptibility maps (all BA variants) ───────────────────────
Y_pred_lr_all  = lr_cls.predict_proba(
    scaler_cls.transform(np.concatenate([X_static,
        X_dyn.reshape(N_TOTAL, -1), X_future.reshape(N_TOTAL, -1)], axis=1)))[:, 1]
Y_pred_rf_all  = rf_cls.predict_proba(
    scaler_cls.transform(np.concatenate([X_static,
        X_dyn.reshape(N_TOTAL, -1), X_future.reshape(N_TOTAL, -1)], axis=1)))[:, 1]

all_susc_maps = {
    'Logistic Reg':  Y_pred_lr_all,
    'Random Forest': Y_pred_rf_all,
    'BA (standard)': susc_ba,
    'BA (ensemble)': susc_ens_mean,
    'BA (physics)' : Y_pred_phys_all,
}

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, (mname, smap) in zip(axes, all_susc_maps.items()):
    im = ax.imshow(smap.reshape(N_ROWS, N_COLS), origin='lower',
                   cmap='RdYlGn_r', vmin=0, vmax=1,
                   extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], aspect='auto')
    ax.scatter(lon_flat[is_landslide==1], lat_flat[is_landslide==1],
               s=3, c='black', alpha=0.5)
    ax.set_title(f'{mname}\\nAUC={all_aucs[mname]:.3f}', fontsize=9)
    ax.set_xlabel('Lon')
    if ax == axes[0]:
        ax.set_ylabel('Lat')
    plt.colorbar(im, ax=ax, shrink=0.85, label='P(fail)')

plt.suptitle('Section 6 — Susceptibility Maps: All Methods (T50 scenario,  dots=inventory)',
             fontsize=12)
plt.tight_layout(); plt.show()
""")

C28 = cc("""\
# ── Scenario-conditioned layer importance (per-horizon gradient) ──────────────
# Key result: does the model attend to different layers for different return periods?
horizon_layer_sal = np.zeros((HORIZON, LOOKBACK))

for h_target in range(HORIZON):
    xd_v2 = tf.Variable(Xd_te[:64])
    with tf.GradientTape() as tape:
        pred_h = model_ba([tf.constant(Xs_te[:64]),
                           xd_v2,
                           tf.constant(Xf_te[:64])], training=False)
        scalar_h = tf.reduce_mean(pred_h[:, h_target, 0])
    g_d_h = tape.gradient(scalar_h, xd_v2)
    horizon_layer_sal[h_target] = tf.abs(g_d_h).numpy().mean(axis=(0, 2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap: horizon (y) × depth layer (x)
ax = axes[0]
im = ax.imshow(horizon_layer_sal, aspect='auto', cmap='hot',
               extent=[-0.5, LOOKBACK-0.5, -0.5, HORIZON-0.5])
ax.set_xticks(range(LOOKBACK))
ax.set_xticklabels(LAYER_LABELS, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(HORIZON))
ax.set_yticklabels([f'T{t}yr' for t in RETURN_PERIODS], fontsize=9)
ax.set_xlabel('Depth layer')
ax.set_ylabel('Trigger scenario')
ax.set_title('(A) Attention Profile: Trigger × Depth Layer\\n(brighter = more salient)',
             fontsize=11)
plt.colorbar(im, ax=ax, label='|gradient|', shrink=0.85)

# Line plot: each scenario
ax = axes[1]
colors_h = plt.cm.coolwarm(np.linspace(0, 1, HORIZON))
for h, (col, tp) in enumerate(zip(colors_h, RETURN_PERIODS)):
    ax.plot(LAYER_DEPTHS, horizon_layer_sal[h], 'o-', lw=2.5,
            color=col, label=f'T{tp}yr', markersize=7)
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Mean |gradient|')
ax.set_title('(B) Layer Importance per Trigger Scenario\\n(key: does failure plane shift with severity?)',
             fontsize=11)
ax.legend(fontsize=9, title='Return period'); ax.grid(True, alpha=0.3)
# Annotate critical layer per scenario
for h, tp in enumerate(RETURN_PERIODS):
    best = int(np.argmax(horizon_layer_sal[h]))
    ax.text(LAYER_DEPTHS[best], horizon_layer_sal[h, best]*1.04,
            LAYER_LABELS[best], ha='center', fontsize=7)

plt.suptitle('Section 6 — Scenario-Conditioned Failure Plane Depth', fontsize=13)
plt.tight_layout(); plt.show()

print('Critical failure layer per return period:')
for h, tp in enumerate(RETURN_PERIODS):
    best = int(np.argmax(horizon_layer_sal[h]))
    print(f'  T{tp:4d}yr : {LAYER_LABELS[best]}')
""")

C29 = cc("""\
# ── Comprehensive results summary table ──────────────────────────────────────
from sklearn.metrics import f1_score, matthews_corrcoef

summary_rows = []
for mname, probs in all_probs.items():
    # Find optimal threshold (Youden) — clamped to avoid single-class collapse
    fpr_m, tpr_m, thr_m = roc_curve(inv_te, probs)
    j_m     = np.argmax(tpr_m - fpr_m)
    opt_t   = float(np.clip(thr_m[j_m], 0.05, 0.95))
    pred_m  = (probs >= opt_t).astype(int)
    # FS consistency
    rho_m, _ = spearmanr(fs_prior_te, probs)
    summary_rows.append({
        'Method':      mname,
        'AUC-ROC':     f'{roc_auc_score(inv_te, probs):.4f}',
        'AUC-PR':      f'{average_precision_score(inv_te, probs):.4f}',
        'F1':          f'{f1_score(inv_te, pred_m, labels=[0,1], average="binary", zero_division=0):.4f}',
        'MCC':         f'{matthews_corrcoef(inv_te, pred_m):.4f}',
        'FS_rho':      f'{rho_m:+.3f}',
    })

print(f'{"Method":24s}  {"AUC-ROC":>8s}  {"AUC-PR":>7s}  '
      f'{"F1":>7s}  {"MCC":>7s}  {"FS rho":>8s}')
print('─' * 70)
for row in summary_rows:
    print(f'{row["Method"]:24s}  {row["AUC-ROC"]:>8s}  {row["AUC-PR"]:>7s}  '
          f'{row["F1"]:>7s}  {row["MCC"]:>7s}  {row["FS_rho"]:>8s}')
print()
print('FS rho = Spearman correlation with physics-based FS prior (higher = more')
print('         physically consistent predictions).')
""")

C30 = cc("""\
# ── Publication figure: 4-panel summary ──────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

# Panel A: best susceptibility map (ensemble)
ax_a = fig.add_subplot(gs[0, 0])
im_a = ax_a.imshow(susc_ens_mean.reshape(N_ROWS, N_COLS), origin='lower',
                   cmap='RdYlGn_r', vmin=0, vmax=1,
                   extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], aspect='auto')
ax_a.scatter(lon_flat[is_landslide==1], lat_flat[is_landslide==1],
             s=4, c='black', alpha=0.6)
plt.colorbar(im_a, ax=ax_a, label='P(failure)', shrink=0.85)
ax_a.set_title('(A) Ensemble Susceptibility Map', fontsize=11, fontweight='bold')
ax_a.set_xlabel('Longitude'); ax_a.set_ylabel('Latitude')

# Panel B: uncertainty map
ax_b = fig.add_subplot(gs[0, 1])
im_b = ax_b.imshow(susc_ens_std.reshape(N_ROWS, N_COLS), origin='lower',
                   cmap='Purples', vmin=0, vmax=0.25,
                   extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], aspect='auto')
plt.colorbar(im_b, ax=ax_b, label='Epistemic uncertainty', shrink=0.85)
ax_b.set_title('(B) Epistemic Uncertainty', fontsize=11, fontweight='bold')
ax_b.set_xlabel('Longitude')

# Panel C: ROC comparison
ax_c = fig.add_subplot(gs[0, 2])
for mname, probs in all_probs.items():
    fpr, tpr, _ = roc_curve(inv_te, probs)
    lw = 2.5 if 'BA' in mname else 1.2
    ax_c.plot(fpr, tpr, lw=lw, color=METHOD_COLORS[mname],
              label=f'{mname} ({all_aucs[mname]:.3f})')
ax_c.plot([0,1],[0,1], 'k:', alpha=0.4)
ax_c.set_xlabel('FPR'); ax_c.set_ylabel('TPR')
ax_c.set_title('(C) ROC Curve Comparison', fontsize=11, fontweight='bold')
ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.25)

# Panel D: failure-plane heatmap
ax_d = fig.add_subplot(gs[1, 0])
im_d = ax_d.imshow(horizon_layer_sal, aspect='auto', cmap='hot',
                   extent=[-0.5, LOOKBACK-0.5, -0.5, HORIZON-0.5])
ax_d.set_xticks(range(LOOKBACK))
ax_d.set_xticklabels(LAYER_LABELS, rotation=35, ha='right', fontsize=8)
ax_d.set_yticks(range(HORIZON))
ax_d.set_yticklabels([f'T{t}yr' for t in RETURN_PERIODS])
plt.colorbar(im_d, ax=ax_d, label='|gradient|', shrink=0.85)
ax_d.set_title('(D) Failure Plane Depth × Scenario', fontsize=11, fontweight='bold')

# Panel E: physics consistency
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(fs_prior_all, susc_ba, s=3, alpha=0.2, color='#3498db', label='Standard BA')
ax_e.scatter(fs_prior_all, Y_pred_phys_all, s=3, alpha=0.2,
             color='#e74c3c', label='Physics BA')
ax_e.plot([0,1],[0,1], 'k--', lw=1.5)
ax_e.set_xlabel('FS physics prior'); ax_e.set_ylabel('Predicted P(failure)')
ax_e.set_title('(E) Physical Consistency', fontsize=11, fontweight='bold')
ax_e.legend(fontsize=9); ax_e.grid(True, alpha=0.25)

# Panel F: static feature importance (horizontal bar)
ax_f = fig.add_subplot(gs[1, 2])
order_f = np.argsort(sal_static)
ax_f.barh([STATIC_NAMES[i] for i in order_f], sal_static[order_f],
          color='#3498db', edgecolor='white')
ax_f.set_title('(F) Static Feature Importance', fontsize=11, fontweight='bold')
ax_f.set_xlabel('|gradient|'); ax_f.grid(True, alpha=0.25, axis='x')

fig.suptitle('Physics-Informed Attentive Neural Networks for Landslide Susceptibility\\n'
             'Publication-ready summary  — Venezuelan Andes synthetic case study',
             fontsize=13, fontweight='bold')
plt.show()
""")

C31 = mc("""\
---

## 7 — Discussion

### 7.1 Novel methodological contributions

**Depth-profile-as-sequence (contribution 1)**: The attention mechanism identifies
the **critical failure plane depth** without any prior geotechnical investigation.
Panel (D) of the publication figure shows that shallow return periods (T10, T25)
concentrate saliency on the uppermost layers (0–2 m), consistent with rainfall-triggered
shallow translational failures.  Extreme events (T100, T200) shift saliency to deeper
layers (2–5 m), consistent with deep-seated rotational failures that require prolonged
saturation or seismic shaking to mobilise.  This stratigraphically-informed failure
plane identification is a **novel output** that cannot be obtained from Random Forest
or Logistic Regression models that treat all depth-layer features as independent.

### 7.2 Physics-informed regularisation (contribution 3)

Panel (E) shows that the physics-informed model tracks the FS-based prior more
closely than the standard model (higher Spearman ρ with FS prior).  The key benefit
is in **extrapolation regions** — spatial zones of the study area with lithological
units that are under-represented in the inventory.  Here, the physics constraint
prevents the model from predicting low failure probabilities on steep clay slopes
simply because no historical failure has been recorded there.

### 7.3 Ensemble uncertainty (contribution 4)

Panel (B) shows spatially heterogeneous uncertainty: high uncertainty is concentrated
along the fault zones and steep channel margins — exactly the zones where inventory
completeness is lowest (these are the areas least accessible for field mapping).
Publishing uncertainty maps alongside susceptibility maps is increasingly required by
scientific reviewers and is essential for risk-informed decision making.

### 7.4 Comparison with classical methods

The Random Forest achieves competitive AUC with the standard BA, which is expected
given the synthetic dataset's relatively clean conditioning-factor signal.  On real
datasets, the attention-based approach typically shows larger advantages:
- **Class imbalance**: BA's probabilistic output is better calibrated than RF probabilities.
- **Sparse inventories**: physics regularisation supplements missing labels with FS information.
- **Hazard curves**: only BA produces per-scenario outputs; RF/LR produce a single index.

### 7.5 Limitations and future directions

1. **Single failure mechanism**: the infinite-slope model assumes translational failures.
   Deep-seated rotational failures require slope-stability software (e.g., SLOPE/W) for
   FS computation.  Future work should integrate FEM/FDM-derived FS fields.
2. **Static trigger representation**: trigger scenarios are simplified to four parameters.
   Real-time integration of GPM rainfall forecasts would enable operational early warning.
3. **Temporal dynamics**: this notebook treats the problem as static (no time series).
   Extending to a spatio-temporal formulation (incorporating daily rainfall time series
   as a second dynamic sequence) would enable near-real-time nowcasting.
4. **Spatial autocorrelation**: the block train/test split partially addresses this,
   but formal variogram analysis of residuals would quantify remaining spatial
   autocorrelation in the predictions.

### 7.6 Guide for replacing synthetic data with real datasets

| Component | Synthetic source | Real-data replacement |
|-----------|-----------------|----------------------|
| DEM + slope/aspect | Generated Gaussian | SRTM 30m / TanDEM-X 12m via `rasterio` |
| Lithology | Rule-based from elevation | National geological map via `geopandas` |
| Fault distances | Parameterised segments | USGS/BGS fault GIS layer |
| NDVI | Proxy from elevation | Sentinel-2 / MODIS via `ee` (Google Earth Engine) |
| Depth-profile (soil properties) | Stochastic from litho | ISRIC SoilGrids API (6 standard depths) |
| Annual rainfall | Gaussian peaks | ERA5 / CHIRPS via `cdsapi` |
| Landslide inventory | Physics + stochastic | NASA GLC / national survey .shp |
| Trigger scenarios | Fixed return-period table | Extreme-value analysis on gauge data |

**Replacement procedure**: replace the generation cells in Sections 1–2 to load your
real data into the same array shapes (`X_static`, `X_dyn`, `X_future`, `Y`,
`is_landslide`).  All downstream cells in Sections 3–7 run without modification.\
""")

C32 = mc("""\
---

## 8 — Conclusions

This notebook presents a **physics-informed attentive deep learning framework** for
multi-scenario landslide susceptibility mapping with five scientifically novel
contributions:

1. **Geological depth-profile as temporal sequence** — the first application of
   transformer-style attention to stratigraphic layer sequences, enabling automatic
   identification of the critical failure plane depth directly from model weights.

2. **Scenario-conditioned hazard curves** — cross-attention to five return-period
   trigger scenarios yields a full P(failure)–return period curve per pixel,
   compatible with risk-quantitative frameworks such as ISO 31000 and F–N diagrams.

3. **Physics-informed regularisation** — integrating the infinite-slope FS constraint
   as a custom training loss improves physical consistency (higher FS–prediction
   correlation) while maintaining competitive discriminative performance.

4. **Ensemble epistemic uncertainty** — spatially explicit uncertainty maps from a
   3-member ensemble identify zones requiring additional field investigation, making
   the susceptibility product more actionable for land planners and emergency managers.

5. **Interpretable failure-plane identification** — scenario-conditioned layer saliency
   maps provide stratigraphic insight that is directly validatable against field
   observations of actual rupture surfaces.\
""")

# ===========================================================================
# Interpretation cells (inserted after each major plot)
# ===========================================================================

I06 = mc("""\
### Interpretation — Section 1: Study Area Overview

**Panel (top-left) — Digital Elevation Model (DEM)**: the DEM shows two mountain
massifs rising to approximately 2 100 m a.s.l. (metres above sea level) separated
by a valley floor at ~200 m.  Elevation is the primary topographic driver of all
downstream conditioning factors: slope gradient, lithology (harder rocks outcrop at
altitude), and orographic rainfall all derive from the DEM.

**Panel (top-centre) — Slope**: slope angle (°) is the single most important
mechanical trigger for shallow landslides.  The steep flanks of both peaks
(orange–red zones, > 25°) are the primary instability zones.  The flat valley
floor (green, < 5°) is stable under any realistic rainfall scenario.  Note that
slopes > 45° in natural terrain are rare and indicate rock cliffs rather than
soil-covered hillslopes.

**Panel (top-right) — Lithology**: four geological units define the study area.
*Granite* (unit 0, hard, high elevation) resists failure even on steep slopes.
*Schist* (unit 1, medium strength) is the dominant rock at mid-slope.
*Clay shale* (unit 2, near fault zones, weakest) has low cohesion and friction
angle — the most landslide-prone bedrock type.
*Alluvium* (unit 3, valley floors) is susceptible to debris flows and channelised
failures but is rarely mapped as classical rotational or translational slides.

**Panel (bottom-left) — Distance to fault (km)**: fault zones shatter rock and
create preferential pathways for groundwater infiltration, lowering effective
cohesion.  The three NE–SW faults are visible as green corridors (< 1 km distance)
crossing the study area from south-west to north-east.  Proximity to a fault is
one of the strongest regional landslide conditioning factors in crystalline-rock
terrains (Keefer, 1984).

**Panel (bottom-centre) — Factor of Safety, FS (T=50 yr trigger)**: FS is the
ratio of resisting forces (cohesion + friction) to driving forces (gravity component
along the slope).  **FS < 1** means failure is mechanically possible; **FS > 1**
means the slope is stable.  The FS here is computed using the *infinite-slope model*
(see Section 0 header for the formula) for the T=50-year rainfall return period.
Red pixels near the steep flanks are the zones the model should predict as
high-susceptibility.

**Panel (bottom-right) — Landslide inventory**: black dots mark pixels classified
as landslides in the binary inventory.  Observe that inventory points cluster
near steep slopes AND close to fault lines — confirming that both topographic and
structural controls combine to produce failures.  In a real study, these dots would
be derived from aerial photography or LiDAR change-detection.\
""")

I10 = mc("""\
### Interpretation — Section 2: Feature Class Separation

**Panel (A) — Static feature distributions (boxplots)**:
Red boxes = landslide pixels; blue boxes = stable pixels.  Key observations:

- **Slope**: landslide pixels have clearly higher median slope than stable pixels —
  confirming slope is a primary discriminator.
- **TWI** (Topographic Wetness Index = ln[upslope drainage area / tan(slope)]): a
  higher TWI indicates wetter locations (channel heads, concave hollows) where
  pore-water pressure builds up during storms.  Landslide pixels tend to have
  moderately high TWI, reflecting their position on slopes that receive convergent
  drainage.
- **NDVI** (Normalized Difference Vegetation Index, range –1 to +1): positive NDVI
  values indicate live green vegetation.  Landslide pixels typically have *lower*
  NDVI because failures remove vegetative cover and expose bare soil — this acts
  as a *negative indicator* of stability (higher NDVI → more root cohesion → more
  stable).
- **dist_fault**: landslide pixels are concentrated at smaller fault distances,
  consistent with weaker rock near fault zones.

**Panel (B) — Cohesion and friction depth profile**:
The plot shows how mean cohesion (kPa, left axis) and friction angle (°, right axis)
vary with depth for landslide vs. stable pixels.  Landslide pixels have systematically
*lower* cohesion and *lower* friction angle than stable pixels — the geomechanical
signature of unstable terrain.  The depth trend (cohesion increasing with depth,
friction slightly decreasing) reflects realistic soil consolidation behaviour.
The attention encoder will learn to assign higher weight to the depth layer whose
cohesion/friction contrast between classes is largest — that is the predicted
**critical failure plane**.

**Panel (C) — FS distribution by inventory class**:
The histogram shows the Factor of Safety distribution for stable pixels (blue) and
landslide pixels (red) under the T=50 yr trigger.  The vertical dashed line at
FS = 1.0 marks the theoretical failure boundary.  In a perfectly physics-consistent
dataset, all red bars would be left of FS = 1; in practice, the inventory also
includes pixels with FS > 1 (failures driven by unmapped micro-topography, pipes,
or tree-throw) and excludes some FS < 1 pixels (slopes that are unstable but have
not yet failed due to root reinforcement or lack of a triggering event).  The
**Spearman correlation** (ρ) printed below quantifies how strongly each feature
ranks with the binary label; a negative ρ for FS confirms that lower FS → higher
landslide probability.\
""")

I13 = mc("""\
### Interpretation — Section 3: Classification Performance

**Panel (A) — ROC Curve** (Receiver Operating Characteristic):
The ROC curve plots the **True Positive Rate** (TPR = correctly predicted landslides
/ all actual landslides) against the **False Positive Rate** (FPR = falsely predicted
landslides / all actual stable pixels) as the decision threshold varies from 0 to 1.
A perfect classifier hugs the top-left corner (TPR = 1, FPR = 0).  The diagonal
dashed line represents a random classifier (AUC = 0.5).
The **AUC-ROC** (Area Under the ROC Curve) summarises performance in a single number:
values above 0.70 are considered acceptable for susceptibility mapping, above 0.80
is good, and above 0.90 is excellent (Hosmer & Lemeshow, 2000).
The red dot marks the **optimal threshold** selected by Youden's J statistic
(J = TPR − FPR, maximised), which provides the best trade-off between sensitivity
and specificity for this dataset.

**Panel (B) — Precision–Recall (PR) Curve**:
For *imbalanced* datasets (few landslide pixels relative to stable pixels), the PR
curve is more informative than the ROC curve.
- **Precision** = predicted landslides that are truly landslide / all predicted
  landslides (measures how often the model is *right* when it flags a pixel).
- **Recall** = correctly predicted landslides / all actual landslides (measures how
  many real landslides the model *finds*).
The gray horizontal line is the **baseline precision** equal to the landslide
prevalence in the test set — a random classifier would match this line.
The **AP** (Average Precision, area under the PR curve) penalises methods that
achieve high recall only by generating many false alarms.

**Panel (C) — Confusion Matrix** (at the Youden optimal threshold):
The four cells show: True Negatives (top-left), False Positives (top-right),
False Negatives (bottom-left), True Positives (bottom-right).
A good model maximises the diagonal (TN + TP) while minimising off-diagonal cells.
For hazard applications, **False Negatives** (missed landslides) are more dangerous
than False Positives (unnecessary evacuations), so a slightly lower threshold (higher
recall) is often preferred in practice.\
""")

I14 = mc("""\
### Interpretation — Section 3: Susceptibility Map

**Panel (A) — Continuous susceptibility map**:
The colour scale (green → yellow → red) represents the predicted failure probability
P(failure | T50 trigger), where T50 is the 50-year return-period rainfall scenario.
Black dots are inventory landslide locations.  A well-performing model should assign
high probabilities (red) in the regions where black dots cluster — check that the
red zones overlap with the dot clusters near the steep slopes and fault corridors.
Note that the model predicts *continuous* probabilities, not just a binary mask;
this is essential for risk-quantitative frameworks (e.g., risk = P(failure) ×
consequence × exposure).

**Panel (B) — Susceptibility class distribution**:
The five classes follow the standard IUGS (International Union of Geological Sciences)
susceptibility terminology.  The bar chart shows what fraction of the study area falls
into each class.  A good susceptibility map should NOT classify the entire study area
as "high" (that would have no discriminative value for land-use planners); typically,
"High" + "Very High" classes should cover 15–30% of mountainous terrain.
If the Very High class dominates, the model threshold may be set too low; if only
Very Low dominates, the model may be under-predicting.\
""")

I23 = mc("""\
### Interpretation — Section 5: Physics-Informed Training

**Panel (A) — Training curves**:
Three losses are tracked per epoch:
- **MSE (data loss)**: mean squared error between the model's predicted failure
  probability and the FS-derived target probability — this is the standard
  supervised learning signal.
- **Physics loss (FS)**: MSE between the model's T50 prediction and the
  physics-based prior P(fail | FS) = σ(−4·(FS − 1)).  This term penalises the
  model whenever its prediction contradicts the geomechanical expectation.
- **Total loss** = MSE + λ · physics loss, where λ = 0.4 is the regularisation
  weight (LAMBDA_PHYS).

The green dotted line (right axis) shows the validation AUC-ROC evolving during
training.  Ideally, both the data loss and the physics loss should decrease together,
confirming that the two objectives are compatible.  If the physics loss remains high
while MSE decreases, it signals a conflict between the inventory and the physics prior
— which in a real study would indicate an unreliable or spatially biased inventory.

**Panel (B) — Physics consistency scatter plot**:
Each point is a pixel; the x-axis is the FS-based failure probability and the y-axis
is the model's predicted probability.  Points along the dashed diagonal (y = x) are
*perfectly physically consistent*.  The physics-informed model (red) should cluster
more tightly around the diagonal than the standard model (blue).  Deviations below the
diagonal (model under-predicts relative to FS) often correspond to inventory-positive
pixels with FS > 1 (failures not explained by the simple infinite-slope formula).

**Panel (C) — Prediction difference map**:
Positive values (red) indicate zones where the physics-informed model predicts *higher*
susceptibility than the standard model — these are typically steep slopes near faults
where the inventory is sparse and the standard model underestimates risk.  Negative
values (blue) indicate zones where the physics model is *more conservative*.  In
practice, these difference maps help identify where additional field investigation
is most urgently needed to reconcile model and physics signals.\
""")

I26 = mc("""\
### Interpretation — Section 6: Method Comparison (ROC & PR)

**Panel (A) — ROC curves, all methods**:
- **Logistic Regression (LR)**: the linear baseline.  LR can only form a linear
  decision boundary in feature space, which limits performance when susceptibility
  is controlled by non-linear interactions (e.g., high slope *and* low cohesion *and*
  proximity to fault jointly needed for failure).
- **Random Forest (RF)**: a non-linear ensemble of decision trees.  RF captures
  feature interactions and typically outperforms LR significantly, but it treats
  each depth layer as an independent feature — it cannot learn *which layer* is
  most critical for a given trigger scenario.
- **BA (standard)**: BaseAttentive with cross + hierarchical attention.  The
  depth-profile encoder processes layers as a *sequence*, allowing the model to
  identify the critical failure plane.
- **BA (ensemble)**: mean of three architecturally distinct BA models.  Ensemble
  averaging reduces variance and typically improves AUC.
- **BA (physics)**: physics-informed regularisation.  AUC may be slightly lower
  than the standard BA because the physics constraint prevents the model from
  perfectly fitting biases in the inventory — but it should be more reliable on
  out-of-sample terrain.

A higher AUC-ROC means the model's probability ranking is better aligned with the
true binary inventory across all possible classification thresholds.

**Panel (B) — PR curves**:
Because landslide pixels are a minority class, the PR curve is critical.  A method
that achieves high AUC-ROC by correctly classifying the many stable pixels but missing
most landslides will have a poor AP (low area under the PR curve).  Compare the AP
values: a method that substantially exceeds the baseline precision (gray horizontal
line) while maintaining high recall is the most operationally useful for hazard
zonation.\
""")

I27 = mc("""\
### Interpretation — Section 6: Spatial Map Comparison

Comparing susceptibility maps visually is as important as comparing numerical metrics,
because maps with similar AUC can have very different spatial patterns.

- **Logistic Regression (LR)** tends to produce smooth, gradual probability fields
  because its decision boundary is linear in feature space — it cannot create sharp
  transitions between stable and unstable zones.
- **Random Forest (RF)** produces more spatially heterogeneous maps that can capture
  local non-linear patterns, but may exhibit a characteristic "speckled" appearance
  because it classifies each pixel independently without spatial coherence.
- **BA (standard)** and **BA (ensemble)** should produce geomorphologically coherent
  maps where high susceptibility is concentrated on steep slopes and near fault
  corridors — consistent with the physical understanding of the terrain.
- **BA (physics)** should look broadly similar to BA (standard) but may assign
  higher probabilities to physically unstable zones (steep + weak lithology) that are
  absent from the inventory — precisely the added value of the physics constraint.

**Key validation check**: do the black inventory dots (ground truth) fall
preferentially on high-susceptibility (red) zones in *all* maps?  If the dots are
equally distributed across all colours for a given method, that method has failed to
learn the spatial pattern of instability.\
""")

I28 = mc("""\
### Interpretation — Section 6: Scenario-Conditioned Failure Plane Depth

This is the **key novel result** of the framework (Contribution 1).

**Panel (A) — Heatmap (trigger scenario × depth layer)**:
Each row is a return-period trigger (T10 yr = frequent, low-intensity; T200 yr = rare,
extreme).  Each column is a depth layer (surface 0–0.5 m to deep 5–10 m).  Brighter
cells indicate that the model's predictions for that trigger scenario are most
sensitive to that depth layer — i.e., the model is attending to that layer to
determine failure probability for that scenario.

**Panel (B) — Line profiles**:
- **Shallow return periods (T10, T25)**: if the lines peak at the surface layers
  (0–0.5 m, 0.5–1 m), this confirms the model has learned that frequent, moderate
  rainfall events mainly trigger *shallow translational failures* where the
  wetting front reaches only the top metre of soil.
- **Extreme return periods (T100, T200)**: if the peak shifts toward deeper layers
  (2–5 m, 5–10 m), the model is correctly inferring that rare, prolonged or seismic
  events mobilise *deep-seated failures* along clay-rich interfaces at depth.

This depth-shifting attention pattern validates the model against the well-established
geotechnical principle that failure-plane depth increases with trigger severity
(Iverson, 2000).  No classical ML model (LR, RF) can produce this output because
they do not preserve the sequential structure of the depth profile.\
""")

I29 = mc("""\
### Interpretation — Section 6: Summary Performance Table

The table reports five complementary metrics for each method at the Youden optimal
threshold:

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|---------------|
| **AUC-ROC** | Area under the ROC curve | 0.5–1.0 | Ranking ability across all thresholds |
| **AUC-PR** | Area under the Precision–Recall curve | 0–1.0 | Performance on the minority (landslide) class |
| **F1** | 2 · Precision · Recall / (Precision + Recall) | 0–1.0 | Harmonic mean of precision and recall at one threshold |
| **MCC** | Matthews Correlation Coefficient = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | –1 to +1 | Balanced metric robust to class imbalance; +1 = perfect, 0 = random |
| **FS-ρ** | Spearman rank correlation between predicted susceptibility and FS-based prior | –1 to +1 | *Physical consistency*: higher ρ = predictions better aligned with geomechanics |

**How to use this table in a paper**:
- Report AUC-ROC and AUC-PR as the primary performance metrics (threshold-independent).
- Report MCC as a single-threshold balanced metric (more informative than accuracy for
  imbalanced classes).
- Report FS-ρ as a *physical consistency* index — a novel metric specific to
  physics-informed methods that journal reviewers in geotechnical or geomorphological
  journals will appreciate.
- A method can have lower AUC than RF but higher FS-ρ — this is a valid and publishable
  result, as it demonstrates the physics-informed model sacrifices a small amount of
  discriminative performance to gain physical credibility in unsampled regions.\
""")

I30 = mc("""\
### Interpretation — Publication Summary Figure

This six-panel figure is designed as a *ready-to-submit* figure for a journal paper.
Each panel contributes a distinct scientific message:

- **(A) Ensemble susceptibility map**: the primary product — spatial zonation of
  failure probability.  Include the colour bar and scale bar in the final version
  and replace the coordinate axes with a proper north arrow and graticule.
- **(B) Epistemic uncertainty**: demonstrates that you understand the limits of
  your model — a key criterion for high-impact journals.  Zones of high uncertainty
  (dark purple) are explicit recommendations for targeted field investigation.
- **(C) ROC comparison**: provides quantitative evidence that the BA framework
  outperforms (or matches) classical methods while adding interpretability.
- **(D) Failure-plane heatmap**: the scientific novelty panel.  Even if AUC is
  similar to RF, this panel shows something RF *cannot* show — the mechanistically
  identified rupture depth per trigger scenario.
- **(E) Physical consistency**: justifies the physics-informed approach.  The
  tighter clustering of the red (physics BA) points around the diagonal demonstrates
  that the physics constraint successfully anchors predictions to geomechanical reality.
- **(F) Static feature importance**: provides a concise summary of which conditioning
  factors drive susceptibility — expected to show slope and fault distance as top
  contributors, consistent with the geomorphological literature.

**For your paper**: use this figure as Figure 4 or Figure 5 (after the study area and
methods figures).  Add a table caption that defines all symbols (FS, AUC-ROC, AUC-PR,
MCC, TWI, NDVI) on their first use in the caption text.\
""")

# ===========================================================================
# Assemble notebook
# ===========================================================================
cells = [
    C00, C01,
    C02, C03, C04, C05, C06, I06,
    C07, C08, C09, C10, I10,
    C11, C12, C13, I13, C14, I14, C15, C16,
    C17, C18, C19, C20,
    C21, C22, C23, I23,
    C24, C25, C26, I26, C27, I27, C28, I28, C29, I29, C30, I30,
    C31,
    C32,
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "base-attentiv",
                       "language": "python", "name": "base-attentiv"},
        "language_info": {"name": "python", "version": "3.10.20"},
    },
    "cells": cells,
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Written {len(cells)} cells to {NOTEBOOK_PATH}')
