"""
Deep update for 06_crps_probabilistic_forecasting.ipynb.

Changes:
- Replace random y_true with structured sine-wave + noise (meaningful patterns)
- Increase quantile training to 15 epochs
- Add rich visualizations:
  1. Fan chart (quantile intervals) over forecast horizon
  2. Gaussian mean +/- sigma bands
  3. Mixture density curve per forecast step
  4. CRPS comparison bar chart across all 3 modes
  5. Calibration (coverage) plot for quantile mode
"""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('examples/06_crps_probabilistic_forecasting.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def new_id(): return uuid.uuid4().hex[:8]
def code_cell(src):
    return {"cell_type":"code","execution_count":None,"id":new_id(),"metadata":{},"outputs":[],"source":src}
def md_cell(src):
    return {"cell_type":"markdown","id":new_id(),"metadata":{},"source":src}

def find(keyword, ctype=None, start=0):
    for i in range(start, len(cells)):
        src = ''.join(cells[i]['source']) if isinstance(cells[i]['source'],list) else cells[i]['source']
        if keyword in src and (ctype is None or cells[i]['cell_type']==ctype):
            return i
    return -1

# ============================================================
# Replace random data with structured patterns
# ============================================================
DATA_CELL = (
    "# Synthetic dataset: structured daily-cycle signals with controlled noise\n"
    "# Using patterns makes probabilistic forecasting results interpretable.\n"
    "BATCH = 128\n"
    "LOOKBACK = 24\n"
    "HORIZON  = 12\n"
    "STATIC_DIM  = 4\n"
    "DYN_DIM     = 8\n"
    "FUT_DIM     = 4\n"
    "OUTPUT_DIM  = 2\n\n"
    "rng = np.random.default_rng(42)\n\n"
    "# Daily sine-wave base pattern\n"
    "t_hist = np.linspace(0, 4*np.pi, LOOKBACK)\n"
    "t_fut  = np.linspace(4*np.pi, 6*np.pi, HORIZON)\n"
    "base_past   = np.sin(t_hist)\n"
    "base_future = np.sin(t_fut)\n\n"
    "# Static + dynamic + future inputs\n"
    "x_static  = rng.standard_normal((BATCH, STATIC_DIM)).astype('float32')\n"
    "x_dynamic = np.zeros((BATCH, LOOKBACK, DYN_DIM), dtype='float32')\n"
    "for d in range(DYN_DIM):\n"
    "    x_dynamic[:,:,d] = (np.tile(np.sin(t_hist*(1+d*0.1)), (BATCH,1))\n"
    "                        + 0.15*rng.standard_normal((BATCH,LOOKBACK)))\n"
    "x_future  = rng.standard_normal((BATCH, HORIZON, FUT_DIM)).astype('float32')\n\n"
    "# Target: sine wave + moderate noise (unimodal -- good for quantile/gaussian)\n"
    "y_true = np.stack([\n"
    "    np.tile(base_future, (BATCH,1)) + 0.25*rng.standard_normal((BATCH,HORIZON)),\n"
    "    np.tile(base_future*(1.2), (BATCH,1)) + 0.3*rng.standard_normal((BATCH,HORIZON)),\n"
    "], axis=-1).astype('float32')\n\n"
    "# Bimodal target for mixture mode: two regimes separated by +/-0.6\n"
    "regime = rng.integers(0, 2, size=(BATCH, HORIZON))\n"
    "shift  = np.where(regime, +0.6, -0.6)\n"
    "y_bimodal = np.stack([\n"
    "    np.tile(base_future, (BATCH,1)) + shift + 0.1*rng.standard_normal((BATCH,HORIZON)),\n"
    "    np.tile(base_future, (BATCH,1)) + shift + 0.1*rng.standard_normal((BATCH,HORIZON)),\n"
    "], axis=-1).astype('float32')\n\n"
    "print('x_static :', x_static.shape)\n"
    "print('x_dynamic:', x_dynamic.shape)\n"
    "print('x_future :', x_future.shape)\n"
    "print('y_true   :', y_true.shape,  ' (unimodal, for quantile/gaussian)')\n"
    "print('y_bimodal:', y_bimodal.shape,' (bimodal regime, for mixture)')\n"
)

# ============================================================
# Updated quantile training cell (15 epochs + better output)
# ============================================================
Q_TRAIN = (
    "import keras\n\n"
    "QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]\n\n"
    "model_q = BaseAttentive(\n"
    "    static_input_dim=STATIC_DIM,\n"
    "    dynamic_input_dim=DYN_DIM,\n"
    "    future_input_dim=FUT_DIM,\n"
    "    output_dim=OUTPUT_DIM,\n"
    "    forecast_horizon=HORIZON,\n"
    "    quantiles=QUANTILES,\n"
    "    embed_dim=32,\n"
    "    num_heads=4,\n"
    "    dropout_rate=0.1,\n"
    "    name='QuantileModel',\n"
    ")\n"
    "preds_q = model_q([x_static, x_dynamic, x_future])\n"
    "print('Quantile output shape:', preds_q.shape)\n"
    "# Expected: (128, 12, 5, 2) -- batch, horizon, quantiles, output_dim\n\n"
    "crps_q = CRPSLoss(mode='quantile', quantiles=QUANTILES)\n"
    "model_q.compile(optimizer=keras.optimizers.Adam(1e-3), loss=crps_q, metrics=['mae'])\n"
    "print('Training quantile model (15 epochs)...')\n"
    "history_q = model_q.fit(\n"
    "    x=[x_static, x_dynamic, x_future], y=y_true,\n"
    "    epochs=15, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n"
    "print(f'  Final train CRPS: {history_q.history[\"loss\"][-1]:.4f}  '\n"
    "      f'val CRPS: {history_q.history[\"val_loss\"][-1]:.4f}')\n"
)

Q_READ = (
    "import numpy as np\n\n"
    "preds_np = np.array(model_q([x_static, x_dynamic, x_future]))\n"
    "# preds_np shape: (BATCH, HORIZON, N_QUANTILES, OUTPUT_DIM)\n\n"
    "median_idx = QUANTILES.index(0.5)\n"
    "lower_idx  = QUANTILES.index(0.1)\n"
    "q25_idx    = QUANTILES.index(0.25)\n"
    "q75_idx    = QUANTILES.index(0.75)\n"
    "upper_idx  = QUANTILES.index(0.9)\n\n"
    "median_pred  = preds_np[:, :, median_idx, :]   # (128, 12, 2)\n"
    "lower_bound  = preds_np[:, :, lower_idx,  :]   # (128, 12, 2)\n"
    "q25_bound    = preds_np[:, :, q25_idx,    :]\n"
    "q75_bound    = preds_np[:, :, q75_idx,    :]\n"
    "upper_bound  = preds_np[:, :, upper_idx,  :]   # (128, 12, 2)\n\n"
    "interval_width_80 = upper_bound - lower_bound\n"
    "print('Median prediction shape:', median_pred.shape)\n"
    "print('80% PI mean width:', interval_width_80.mean().round(4))\n"
)

Q_VIZ = (
    "import matplotlib.pyplot as plt\n\n"
    "steps = np.arange(1, HORIZON + 1)\n\n"
    "# ---- Fan chart for output dim 0 ----\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
    "for d, ax in enumerate(axes):\n"
    "    s = d   # show sample 0 for dim 0, sample 1 for dim 1\n"
    "    ax.fill_between(steps,\n"
    "                    lower_bound[s,:,d], upper_bound[s,:,d],\n"
    "                    alpha=0.20, color='steelblue', label='80% PI (q10-q90)')\n"
    "    ax.fill_between(steps,\n"
    "                    q25_bound[s,:,d],   q75_bound[s,:,d],\n"
    "                    alpha=0.40, color='steelblue', label='50% PI (q25-q75)')\n"
    "    ax.plot(steps, median_pred[s,:,d],\n"
    "            color='steelblue',  lw=2.5,  label='Median (q50)')\n"
    "    ax.plot(steps, y_true[s,:,d],\n"
    "            color='black',      lw=2,    linestyle='--', label='Actual', zorder=5)\n"
    "    ax.set_title(f'Output dim {d+1} -- Quantile Fan Chart', fontsize=12)\n"
    "    ax.set_xlabel('Forecast step')\n"
    "    ax.set_ylabel('Value')\n"
    "    ax.legend(fontsize=9)\n"
    "    ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Mode 1 -- Quantile CRPS: Probabilistic Forecast Fan Charts', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n"
)

Q_CALIB = (
    "# Calibration plot: empirical coverage vs nominal quantile level\n"
    "# A well-calibrated model should have coverage ≈ nominal level.\n"
    "nominal_levels = np.array(QUANTILES)\n"
    "emp_coverage   = np.array([\n"
    "    float(np.mean(y_true[:,:,0] <= preds_np[:,:,q,0]))\n"
    "    for q in range(len(QUANTILES))\n"
    "])\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n\n"
    "ax = axes[0]\n"
    "ax.plot([0,1], [0,1], 'k--', lw=1.5, label='Perfect calibration')\n"
    "ax.plot(nominal_levels, emp_coverage,\n"
    "        'o-', color='steelblue', lw=2, markersize=7, label='Empirical coverage')\n"
    "ax.set_xlabel('Nominal quantile level'); ax.set_ylabel('Empirical coverage')\n"
    "ax.set_title('Quantile Calibration Plot', fontsize=12)\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "ax = axes[1]\n"
    "ax.plot(history_q.history['loss'],     color='steelblue', lw=2, label='Train CRPS')\n"
    "ax.plot(history_q.history['val_loss'], color='tomato',    lw=2, linestyle='--', label='Val CRPS')\n"
    "ax.set_title('Quantile Model Training History', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('CRPS Loss')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Mode 1 -- Calibration & Convergence', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# Gaussian mode - richer visualization
# ============================================================
GAUSS_LATENT = (
    "# Build a small Keras model to produce latent features from our inputs\n"
    "FEATURE_DIM = 16\n"
    "# Use a simple dense projection to get latent features\n"
    "latent_inp  = keras.Input(shape=(HORIZON, DYN_DIM), name='dyn_inp')\n"
    "latent_proj = keras.layers.Dense(FEATURE_DIM, activation='relu')(latent_inp)\n"
    "latent_model = keras.Model(latent_inp, latent_proj, name='latent_encoder')\n"
    "# Extract latent features from dynamic input (first HORIZON steps)\n"
    "latent_features = np.array(\n"
    "    latent_model(x_dynamic[:, :HORIZON, :DYN_DIM].astype('float32'))\n"
    ").astype('float32')\n\n"
    "gaussian_head = GaussianHead(output_dim=OUTPUT_DIM)\n"
    "gaussian_raw  = gaussian_head(latent_features)\n"
    "y_pred_g = {'loc': gaussian_raw['mean'], 'scale': gaussian_raw['scale']}\n\n"
    "print('Gaussian loc   shape:', y_pred_g['loc'].shape)\n"
    "print('Gaussian scale shape:', y_pred_g['scale'].shape)\n"
)

GAUSS_CRPS = (
    "crps_g = CRPSLoss(mode='gaussian')\n"
    "loss_g = crps_g(y_true, y_pred_g)\n"
    "crps_g_val = float(np.array(loss_g))\n"
    "print(f'Gaussian CRPS: {crps_g_val:.4f}')\n\n"
    "loc_np   = np.array(y_pred_g['loc'])\n"
    "sigma_np = np.array(y_pred_g['scale'])\n"
    "print('Mean  shape:', loc_np.shape)\n"
    "print('Sigma shape:', sigma_np.shape)\n"
    "print('Mean sigma (sample 0, step 0):', sigma_np[0, 0].round(4))\n"
)

GAUSS_VIZ = (
    "# Gaussian uncertainty bands: mean +/- 1sigma and +/- 2sigma\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
    "for d, ax in enumerate(axes):\n"
    "    s = d\n"
    "    mu  = loc_np[s, :, d]\n"
    "    sig = sigma_np[s, :, d]\n\n"
    "    ax.fill_between(steps, mu-2*sig, mu+2*sig,\n"
    "                    alpha=0.15, color='darkorange', label=r'Mean +/- 2sigma (95%)')\n"
    "    ax.fill_between(steps, mu-sig,   mu+sig,\n"
    "                    alpha=0.35, color='darkorange', label=r'Mean +/- 1sigma (68%)')\n"
    "    ax.plot(steps, mu,              color='darkorange', lw=2.5, label='Gaussian mean')\n"
    "    ax.plot(steps, y_true[s,:,d],   color='black',      lw=2, linestyle='--',\n"
    "            label='Actual', zorder=5)\n"
    "    ax.set_title(f'Output dim {d+1} -- Gaussian Uncertainty', fontsize=12)\n"
    "    ax.set_xlabel('Forecast step'); ax.set_ylabel('Value')\n"
    "    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Mode 2 -- Gaussian CRPS: Mean +/- Sigma Uncertainty', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "# Sigma growth over horizon\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "for samp in range(min(5, BATCH)):\n"
    "    ax.plot(steps, sigma_np[samp,:,0], alpha=0.6, lw=1.5,\n"
    "            label=f'Sample {samp+1}' if samp < 3 else '')\n"
    "ax.plot(steps, sigma_np[:,:,0].mean(0), color='black', lw=2.5, label='Mean sigma')\n"
    "ax.set_title('Gaussian: Predictive Sigma over Forecast Horizon', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Sigma (uncertainty)')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# Mixture mode - bimodal visualization
# ============================================================
MIX_PRED = (
    "N_COMPONENTS = 3\n\n"
    "mixture_head = MixtureDensityHead(output_dim=OUTPUT_DIM, num_components=N_COMPONENTS)\n"
    "mixture_raw  = mixture_head(latent_features)\n"
    "y_pred_m = {\n"
    "    'loc':     mixture_raw['means'],\n"
    "    'scale':   mixture_raw['scales'],\n"
    "    'weights': mixture_raw['weights'],\n"
    "}\n"
    "print('Mixture loc     shape:', y_pred_m['loc'].shape)\n"
    "print('Mixture scale   shape:', y_pred_m['scale'].shape)\n"
    "print('Mixture weights shape:', y_pred_m['weights'].shape)\n"
)

MIX_CRPS = (
    "crps_m     = CRPSLoss(mode='mixture', mc_samples=128)\n"
    "loss_m     = crps_m(y_bimodal, y_pred_m)\n"
    "crps_m_val = float(np.array(loss_m))\n"
    "print(f'Mixture CRPS (128 samples): {crps_m_val:.4f}')\n"
)

MIX_VIZ = (
    "# Mixture density visualization: plot the GMM density at each step\n"
    "import matplotlib.cm as cm\n\n"
    "means_np   = np.array(y_pred_m['loc'])     # (B, H, K, D)\n"
    "scales_np  = np.array(y_pred_m['scale'])   # (B, H, K, D)\n"
    "weights_np = np.array(y_pred_m['weights']) # (B, H, K, D)\n\n"
    "fig, axes = plt.subplots(2, 3, figsize=(15, 7))\n"
    "z_grid = np.linspace(-2.5, 2.5, 200)\n"
    "cmap = cm.get_cmap('tab10')\n\n"
    "for step_i, ax in enumerate(axes.ravel()):\n"
    "    if step_i >= HORIZON: ax.axis('off'); continue\n"
    "    s, d = 0, 0  # sample 0, output dim 0\n"
    "    # Sum of weighted Gaussian densities\n"
    "    density = np.zeros_like(z_grid)\n"
    "    for k in range(N_COMPONENTS):\n"
    "        mu_k  = means_np[s, step_i, k, d]\n"
    "        sig_k = scales_np[s, step_i, k, d]\n"
    "        w_k   = weights_np[s, step_i, k, d]\n"
    "        from scipy.stats import norm as scipy_norm\n"
    "        density += w_k * scipy_norm.pdf(z_grid, mu_k, sig_k)\n"
    "    ax.fill_between(z_grid, density, alpha=0.4, color=cmap(step_i % 10))\n"
    "    ax.plot(z_grid, density, color=cmap(step_i % 10), lw=1.5)\n"
    "    ax.axvline(y_bimodal[s, step_i, d], color='black', lw=2, linestyle='--',\n"
    "               label='Actual' if step_i==0 else '')\n"
    "    ax.set_title(f'Step {step_i+1}', fontsize=10)\n"
    "    ax.set_xlabel('Value') if step_i >= 3 else None\n"
    "    ax.set_yticks([]); ax.grid(True, alpha=0.2)\n"
    "    if step_i == 0: ax.legend(fontsize=8)\n\n"
    "plt.suptitle('Mode 3 -- Mixture Density: Predictive Distribution per Forecast Step',\n"
    "             fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# Final comparison + high-sample eval
# ============================================================
HIGH_EVAL = (
    "crps_eval = CRPSLoss(mode='mixture', mc_samples=512)\n"
    "eval_loss  = crps_eval(y_bimodal, y_pred_m)\n"
    "crps_m_512 = float(np.array(eval_loss))\n"
    "print(f'Mixture CRPS (512 samples): {crps_m_512:.6f}')\n"
    "print(f'Mixture CRPS (128 samples): {crps_m_val:.6f}')\n"
    "print(f'Estimation variance: {abs(crps_m_val - crps_m_512):.6f}')\n"
)

COMPARE_VIZ = (
    "# Overlay all 3 mode predictions on the same sample\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
    "for d, ax in enumerate(axes):\n"
    "    s = 0\n"
    "    # Quantile fan\n"
    "    ax.fill_between(steps,\n"
    "                    lower_bound[s,:,d], upper_bound[s,:,d],\n"
    "                    alpha=0.15, color='steelblue', label='Quantile 80% PI')\n"
    "    ax.plot(steps, median_pred[s,:,d],\n"
    "            color='steelblue',  lw=2, label='Quantile median')\n"
    "    # Gaussian bands\n"
    "    ax.fill_between(steps,\n"
    "                    loc_np[s,:,d]-sigma_np[s,:,d],\n"
    "                    loc_np[s,:,d]+sigma_np[s,:,d],\n"
    "                    alpha=0.15, color='darkorange', label='Gaussian +/-1sigma')\n"
    "    ax.plot(steps, loc_np[s,:,d],\n"
    "            color='darkorange',  lw=2, label='Gaussian mean')\n"
    "    # Mixture component means\n"
    "    for k in range(N_COMPONENTS):\n"
    "        ax.plot(steps, means_np[s,:,k,d],\n"
    "                color='mediumseagreen', lw=1.2, linestyle=':',\n"
    "                alpha=0.8, label=f'Mixture comp {k+1}' if d==0 and k==0 else '')\n"
    "    # Actual\n"
    "    ax.plot(steps, y_true[s,:,d],\n"
    "            color='black', lw=2.5, linestyle='--', label='Actual', zorder=6)\n"
    "    ax.set_title(f'Output dim {d+1} -- All 3 CRPS Modes Compared', fontsize=12)\n"
    "    ax.set_xlabel('Forecast step'); ax.set_ylabel('Value')\n"
    "    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('CRPS Modes Comparison: Quantile vs Gaussian vs Mixture', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "# CRPS summary bar chart\n"
    "crps_quantile_eval = float(np.array(CRPSLoss(mode='quantile', quantiles=QUANTILES)(y_true, np.array(model_q([x_static, x_dynamic, x_future])))))\n"
    "crps_gauss_eval    = float(np.array(crps_g))\n"
    "crps_mix_eval      = crps_m_val\n\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "labels = ['Quantile\\n(pinball)', 'Gaussian\\n(closed-form)', 'Mixture\\n(MC-128)']\n"
    "values = [crps_quantile_eval, crps_gauss_eval, crps_mix_eval]\n"
    "colors = ['steelblue', 'darkorange', 'mediumseagreen']\n"
    "bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white', lw=1.5)\n"
    "for bar, v in zip(bars, values):\n"
    "    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.4f}',\n"
    "            ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
    "ax.set_title('CRPS by Mode', fontsize=13)\n"
    "ax.set_ylabel('CRPS Loss'); ax.grid(True, axis='y', alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# Apply changes
# ============================================================

# 1. Replace data cell
i = find('BATCH = 64', 'code')
if i >= 0:
    cells[i] = code_cell(DATA_CELL)
    print(f"Replaced data cell at [{i}]")

# 2. Replace quantile model build + train cell (combined)
i = find('QUANTILES = [0.1', 'code')
if i >= 0:
    cells[i] = code_cell(Q_TRAIN)
    print(f"Replaced quantile build/train cell at [{i}]")

# 3. Update quantile train step
i = find('CRPSLoss(mode="quantile"', 'code')
if i >= 0:
    del cells[i]
    print(f"Removed old crps_q compile cell (merged into Q_TRAIN)")

# 4. Replace reading quantile predictions cell
i = find('Reading quantile', None)
if i < 0: i = find('import numpy as np', 'code', 5)
if i >= 0:
    cells[i+1] = code_cell(Q_READ)
    print(f"Replaced quantile read cell at [{i+1}]")

# Insert fan chart + calibration after Q_READ
q_read_i = find('80 %', 'code')
if q_read_i >= 0:
    cells[q_read_i+1:q_read_i+1] = [
        md_cell("### Fan Chart — Quantile Intervals"),
        code_cell(Q_VIZ),
        md_cell("### Calibration Plot — Coverage vs Nominal Level"),
        code_cell(Q_CALIB),
    ]
    print(f"Inserted quantile viz after [{q_read_i}]")

# 5. Replace Gaussian latent cell
i = find('FEATURE_DIM = 16', 'code')
if i >= 0:
    cells[i] = code_cell(GAUSS_LATENT)
    print(f"Replaced gaussian latent cell at [{i}]")

# 6. Replace Gaussian CRPS cell
i = find('crps_g = CRPSLoss(mode="gaussian")', 'code')
if i >= 0:
    cells[i] = code_cell(GAUSS_CRPS)
    print(f"Replaced gaussian crps cell at [{i}]")

# 7. Replace gaussian inspect cell + insert viz
i = find('loc_pred = np.array', 'code')
if i >= 0:
    cells[i] = code_cell(GAUSS_VIZ)
    print(f"Replaced gaussian inspect/viz cell at [{i}]")

# 8. Replace mixture predict cell
i = find('N_COMPONENTS = 3', 'code')
if i >= 0:
    cells[i] = code_cell(MIX_PRED)
    print(f"Replaced mixture pred cell at [{i}]")

# 9. Replace mixture CRPS cell
i = find('crps_m = CRPSLoss(mode="mixture"', 'code')
if i >= 0:
    cells[i] = code_cell(MIX_CRPS)
    print(f"Replaced mixture crps cell at [{i}]")

# Insert mixture density viz after crps_m cell
i = find('crps_m_val', 'code')
if i >= 0:
    cells[i+1:i+1] = [
        md_cell("### Mixture Density — Distribution per Forecast Step"),
        code_cell(MIX_VIZ),
    ]
    print(f"Inserted mixture viz after [{i}]")

# 10. Replace high mc_samples eval cell
i = find('crps_eval = CRPSLoss(mode="mixture"', 'code')
if i >= 0:
    cells[i] = code_cell(HIGH_EVAL)
    print(f"Replaced high-sample eval cell at [{i}]")

# 11. Replace the custom training loop comment cell + insert comparison viz
i = find('Custom Training Loop', None)
if i >= 0:
    i = find('See comment above', 'code', i)
    if i >= 0:
        cells[i:i+1] = [
            md_cell("## All-Modes Comparison & CRPS Summary"),
            code_cell(COMPARE_VIZ),
        ]
        print(f"Inserted comparison viz at [{i}]")

print(f"\nFinal cells: {len(nb['cells'])}")
for j, c in enumerate(cells):
    src = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"  [{j:02d}] {c['cell_type'][:4]} | {src[:65].replace(chr(10),' ')}")

with open('examples/06_crps_probabilistic_forecasting.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nDone.")
