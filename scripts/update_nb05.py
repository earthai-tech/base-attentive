"""
Update 05_kernel_robust_networks.ipynb:
- Replace random targets with structured sine-wave patterns
- Increase training epochs
- Add architecture-specific visualization for each of the 4 patterns:
  1. Ensemble: individual kernels + ensemble + uncertainty band
  2. Physics-Guided: constrained vs unconstrained predictions
  3. Transfer Learning: convergence curves pre-train vs fine-tune vs scratch
  4. Multi-Task: 3-panel outputs
"""
import json, uuid, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('examples/05_kernel_robust_networks.ipynb', encoding='utf-8') as f:
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
# 1. ENSEMBLE - Replace sample data with structured patterns
# ============================================================

# Replace random data with pattern-based data for ensemble section
ENS_DATA = (
    "# Structured sine-wave data for ensemble demonstration\n"
    "import numpy as np\n"
    "np.random.seed(42)\n"
    "batch_size = 64\n"
    "LOOKBACK, HORIZON = 48, 24\n\n"
    "# Structured pattern: sine wave + harmonic\n"
    "t_past   = np.linspace(0, 4*np.pi, LOOKBACK)\n"
    "t_future = np.linspace(4*np.pi, 6*np.pi, HORIZON)\n\n"
    "static_features = np.random.randn(batch_size, 4).astype('float32')\n"
    "dynamic_past = (np.tile(np.sin(t_past), (batch_size,1))[:,:,None]\n"
    "                + 0.3*np.sin(2*t_past)[None,:,None]\n"
    "                + 0.1*np.random.randn(batch_size,LOOKBACK,5)).astype('float32')[:,:,:5]\n"
    "# Fix: broadcast properly\n"
    "dynamic_past = np.zeros((batch_size, LOOKBACK, 5), dtype='float32')\n"
    "for d in range(5):\n"
    "    dynamic_past[:,:,d] = (np.tile(np.sin(t_past*(1+d*0.1)), (batch_size,1))\n"
    "                           + 0.1*np.random.randn(batch_size,LOOKBACK))\n\n"
    "known_future = np.zeros((batch_size, HORIZON, 2), dtype='float32')\n"
    "for d in range(2):\n"
    "    known_future[:,:,d] = (np.tile(np.cos(t_future*(1+d*0.2)), (batch_size,1))\n"
    "                           + 0.1*np.random.randn(batch_size,HORIZON))\n\n"
    "target = (np.tile(np.sin(t_future), (batch_size,1))[:,:,None]\n"
    "          + 0.1*np.random.randn(batch_size,HORIZON,1)).astype('float32')\n\n"
    "print(f'Ensemble data -- static:{static_features.shape}  '\n"
    "      f'dynamic:{dynamic_past.shape}  future:{known_future.shape}  target:{target.shape}')\n"
)

# Update ensemble training to more epochs
ENS_TRAIN = (
    "import keras\n\n"
    "kernel_1 = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    forecast_horizon=HORIZON, objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},\n"
    "    embed_dim=32, num_heads=4, name='kernel_hybrid',\n"
    ")\n"
    "kernel_2 = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    forecast_horizon=HORIZON, objective='transformer',\n"
    "    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},\n"
    "    embed_dim=32, num_heads=4, name='kernel_transformer',\n"
    ")\n"
    "kernel_3 = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    forecast_horizon=HORIZON, objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['memory', 'cross']},\n"
    "    memory_size=32, embed_dim=32, num_heads=4, name='kernel_memory',\n"
    ")\n\n"
    "print('Created 3 BaseAttentive kernels for ensemble')\n"
)

ENS_BUILD = (
    "# Build ensemble model using Keras functional API\n"
    "static_input  = keras.Input(shape=(4,),          name='static')\n"
    "dynamic_input = keras.Input(shape=(LOOKBACK, 5), name='dynamic_past')\n"
    "future_input  = keras.Input(shape=(HORIZON, 2),  name='known_future')\n\n"
    "pred_1 = kernel_1([static_input, dynamic_input, future_input])\n"
    "pred_2 = kernel_2([static_input, dynamic_input, future_input])\n"
    "pred_3 = kernel_3([static_input, dynamic_input, future_input])\n\n"
    "ensemble_concat   = keras.layers.Concatenate(axis=-1)([pred_1, pred_2, pred_3])\n"
    "ensemble_combined = keras.layers.Dense(64, activation='relu')(ensemble_concat)\n"
    "ensemble_output   = keras.layers.Dense(24, activation='linear')(ensemble_combined)\n"
    "ensemble_output   = keras.layers.Reshape((HORIZON, 1))(ensemble_output)\n\n"
    "ensemble_model = keras.Model(\n"
    "    inputs=[static_input, dynamic_input, future_input],\n"
    "    outputs=ensemble_output,\n"
    "    name='BaseAttentive_Ensemble',\n"
    ")\n"
    "ensemble_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])\n"
    "print('Ensemble model built.')\n"
)

ENS_FIT = (
    "print('Training ensemble model (15 epochs)...')\n"
    "history_ensemble = ensemble_model.fit(\n"
    "    [static_features, dynamic_past, known_future], target,\n"
    "    epochs=15, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n\n"
    "# Get predictions from individual kernels + ensemble\n"
    "pred_k1 = kernel_1.predict([static_features, dynamic_past, known_future], verbose=0)\n"
    "pred_k2 = kernel_2.predict([static_features, dynamic_past, known_future], verbose=0)\n"
    "pred_k3 = kernel_3.predict([static_features, dynamic_past, known_future], verbose=0)\n"
    "ensemble_predictions = ensemble_model.predict(\n"
    "    [static_features, dynamic_past, known_future], verbose=0)\n\n"
    "print(f'Ensemble MSE={history_ensemble.history[\"loss\"][-1]:.4f}  '\n"
    "      f'val={history_ensemble.history[\"val_loss\"][-1]:.4f}')\n"
    "print(f'Prediction shape: {ensemble_predictions.shape}')\n"
)

ENS_PLOT = (
    "import matplotlib.pyplot as plt\n\n"
    "steps = np.arange(1, HORIZON + 1)\n"
    "s = 0\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n\n"
    "# Left: individual kernels + ensemble + actual\n"
    "ax = axes[0]\n"
    "ax.plot(steps, target[s,:,0],               color='black',       lw=2.5,  label='Actual',          zorder=5)\n"
    "ax.plot(steps, pred_k1[s,:,0],              color='steelblue',   lw=1.2,  label='Kernel 1 (Hybrid)',   alpha=0.7)\n"
    "ax.plot(steps, pred_k2[s,:,0],              color='darkorange',  lw=1.2,  label='Kernel 2 (Transformer)', alpha=0.7)\n"
    "ax.plot(steps, pred_k3[s,:,0],              color='mediumseagreen', lw=1.2, label='Kernel 3 (Memory)', alpha=0.7)\n"
    "ax.plot(steps, ensemble_predictions[s,:,0], color='crimson',     lw=2.5,  label='Ensemble',         zorder=4)\n"
    "# Uncertainty band: std across kernels\n"
    "stack = np.stack([pred_k1[:,:,0], pred_k2[:,:,0], pred_k3[:,:,0]], axis=0)\n"
    "band_lo = stack.min(axis=0)[s]\n"
    "band_hi = stack.max(axis=0)[s]\n"
    "ax.fill_between(steps, band_lo, band_hi, alpha=0.15, color='crimson', label='Kernel range')\n"
    "ax.set_title('Ensemble vs Individual Kernels', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Value')\n"
    "ax.legend(fontsize=8); ax.grid(True, alpha=0.3)\n\n"
    "# Right: training loss\n"
    "ax = axes[1]\n"
    "ax.plot(history_ensemble.history['loss'],     color='crimson',  lw=2, label='Train MSE')\n"
    "ax.plot(history_ensemble.history['val_loss'], color='tomato',   lw=2, linestyle='--', label='Val MSE')\n"
    "ax.set_title('Ensemble Training History', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Architecture 1 -- Ensemble: Variance Reduction', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "# MAE comparison\n"
    "maes = {}\n"
    "for name, pred in [('K1 Hybrid', pred_k1), ('K2 Trans.', pred_k2),\n"
    "                   ('K3 Memory', pred_k3), ('Ensemble',  ensemble_predictions)]:\n"
    "    maes[name] = float(np.mean(np.abs(pred[:,:,0] - target[:,:,0])))\n"
    "    print(f'  {name:12s} MAE: {maes[name]:.4f}')\n\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "palette = ['steelblue','darkorange','mediumseagreen','crimson']\n"
    "bars = ax.bar(list(maes.keys()), list(maes.values()), color=palette, width=0.5,\n"
    "              edgecolor='white', lw=1.5)\n"
    "for bar, v in zip(bars, maes.values()):\n"
    "    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.4f}',\n"
    "            ha='center', va='bottom', fontsize=10, fontweight='bold')\n"
    "ax.set_title('Ensemble vs Individual Kernels -- MAE', fontsize=12)\n"
    "ax.set_ylabel('Mean Absolute Error'); ax.grid(True, axis='y', alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# 2. PHYSICS-GUIDED - Better comparison
# ============================================================

PG_TRAIN = (
    "# Physics-guided model created!\n"
    "print('Training physics-guided model (10 epochs)...')\n"
    "pg_model = PhysicsGuidedEnsemble(ensemble_model, physics_weight=0.05)\n"
    "pg_model.compile(optimizer=keras.optimizers.Adam(5e-4))\n\n"
    "# Train standard ensemble (no physics) for comparison\n"
    "ensemble_model.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse')\n"
    "history_std = ensemble_model.fit(\n"
    "    [static_features, dynamic_past, known_future], target,\n"
    "    epochs=10, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n\n"
    "history_pg = pg_model.fit(\n"
    "    ([static_features, dynamic_past, known_future], target),\n"
    "    epochs=10, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n\n"
    "pg_predictions = pg_model.predict(\n"
    "    [static_features, dynamic_past, known_future], verbose=0)\n"
    "std_predictions = ensemble_model.predict(\n"
    "    [static_features, dynamic_past, known_future], verbose=0)\n\n"
    "print('Predictions computed.')\n"
    "print(f'  Physics-guided shape: {pg_predictions.shape}')\n"
)

PG_PLOT = (
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n\n"
    "# Left: prediction comparison\n"
    "ax = axes[0]\n"
    "s = 1\n"
    "ax.plot(steps, target[s,:,0],       color='black',       lw=2.5, label='Actual',           zorder=5)\n"
    "ax.plot(steps, std_predictions[s,:,0], color='steelblue',  lw=2,   linestyle='--', label='Standard Ensemble')\n"
    "ax.plot(steps, pg_predictions[s,:,0],  color='darkorange', lw=2,   linestyle='-',  label='Physics-Guided')\n"
    "ax.set_title('Physics-Guided vs Standard Predictions', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Value')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "# Right: training loss comparison\n"
    "ax = axes[1]\n"
    "ax.plot(history_std.history['loss'], color='steelblue',  lw=2, label='Standard train loss')\n"
    "ax.plot(history_pg.history['loss'],  color='darkorange', lw=2, linestyle='--', label='Physics-guided loss')\n"
    "ax.set_title('Training Loss: Standard vs Physics-Guided', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Architecture 2 -- Physics-Guided Constraints', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# 3. TRANSFER LEARNING - Show convergence advantage
# ============================================================

TL_TRAIN = (
    "# Create pre-trained BaseAttentive model\n"
    "pretrained_model = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    forecast_horizon=HORIZON, objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},\n"
    "    embed_dim=64, num_heads=8, name='pretrained_encoder',\n"
    ")\n"
    "pretrained_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')\n\n"
    "# Pre-train on large dataset\n"
    "N_large = 256\n"
    "t_l = np.linspace(0, 8*np.pi, LOOKBACK + HORIZON + 10)\n"
    "static_large   = np.random.randn(N_large, 4).astype('float32')\n"
    "dynamic_large  = np.zeros((N_large, LOOKBACK, 5), dtype='float32')\n"
    "for d in range(5):\n"
    "    dynamic_large[:,:,d] = np.tile(np.sin(t_l[:LOOKBACK]*(1+d*0.1)),\n"
    "                                    (N_large,1)) + 0.1*np.random.randn(N_large,LOOKBACK)\n"
    "future_large   = np.random.randn(N_large, HORIZON, 2).astype('float32')\n"
    "target_large   = (np.tile(np.sin(t_l[LOOKBACK:LOOKBACK+HORIZON]),\n"
    "                           (N_large,1))[:,:,None]\n"
    "                  + 0.1*np.random.randn(N_large,HORIZON,1)).astype('float32')\n\n"
    "print('Pre-training on large dataset (10 epochs)...')\n"
    "history_pretrain = pretrained_model.fit(\n"
    "    [static_large, dynamic_large, future_large], target_large,\n"
    "    epochs=10, batch_size=32, verbose=0,\n"
    ")\n"
    "print('Pre-training complete.')\n"
)

TL_FINETUNE = (
    "# Fine-tune on small target dataset (16 samples)\n"
    "target_static  = np.random.randn(16, 4).astype('float32')\n"
    "target_dynamic = np.zeros((16, LOOKBACK, 5), dtype='float32')\n"
    "for d in range(5):\n"
    "    target_dynamic[:,:,d] = (np.tile(np.sin(t_past*(1.1+d*0.1)), (16,1))\n"
    "                             + 0.05*np.random.randn(16,LOOKBACK))\n"
    "target_future  = np.random.randn(16, HORIZON, 2).astype('float32')\n"
    "target_y       = (np.tile(np.sin(t_future*1.1), (16,1))[:,:,None]\n"
    "                  + 0.05*np.random.randn(16,HORIZON,1)).astype('float32')\n\n"
    "# Model 1: Fine-tune from pretrained\n"
    "transfer_model = keras.models.clone_model(pretrained_model)\n"
    "transfer_model.set_weights(pretrained_model.get_weights())\n"
    "for layer in transfer_model.layers[:-4]:\n"
    "    layer.trainable = False\n"
    "transfer_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='mse')\n\n"
    "# Model 2: Train from scratch on small dataset\n"
    "scratch_model = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    forecast_horizon=HORIZON, objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},\n"
    "    embed_dim=64, num_heads=8, name='from_scratch',\n"
    ")\n"
    "scratch_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')\n\n"
    "print('Fine-tuning from pretrained (15 epochs)...')\n"
    "history_finetune = transfer_model.fit(\n"
    "    [target_static, target_dynamic, target_future], target_y,\n"
    "    epochs=15, batch_size=8, verbose=0,\n"
    ")\n\n"
    "print('Training from scratch (15 epochs)...')\n"
    "history_scratch = scratch_model.fit(\n"
    "    [target_static, target_dynamic, target_future], target_y,\n"
    "    epochs=15, batch_size=8, verbose=0,\n"
    ")\n\n"
    "transfer_pred = transfer_model.predict(\n"
    "    [target_static, target_dynamic, target_future], verbose=0)\n"
    "scratch_pred  = scratch_model.predict(\n"
    "    [target_static, target_dynamic, target_future], verbose=0)\n"
    "print(f'Transfer learning prediction shape: {transfer_pred.shape}')\n"
)

TL_PLOT = (
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n\n"
    "# Left: convergence comparison\n"
    "ax = axes[0]\n"
    "ax.plot(history_finetune.history['loss'], color='steelblue',  lw=2.5, label='Fine-tune (pretrained)')\n"
    "ax.plot(history_scratch.history['loss'],  color='darkorange', lw=2.5, label='Train from scratch')\n"
    "ax.set_title('Transfer Learning: Convergence Speed', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "# Right: final predictions on target domain\n"
    "ax = axes[1]\n"
    "s = 0\n"
    "ax.plot(steps, target_y[s,:,0],       color='black',      lw=2.5, label='Actual')\n"
    "ax.plot(steps, transfer_pred[s,:,0],  color='steelblue',  lw=2,   linestyle='--', label='Fine-tuned')\n"
    "ax.plot(steps, scratch_pred[s,:,0],   color='darkorange', lw=2,   linestyle=':',  label='From scratch')\n"
    "ax.set_title('Forecast on Target Domain', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Value')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Architecture 3 -- Transfer Learning Advantage', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "mae_ft  = float(np.mean(np.abs(transfer_pred[:,:,0] - target_y[:,:,0])))\n"
    "mae_sc  = float(np.mean(np.abs(scratch_pred[:,:,0]  - target_y[:,:,0])))\n"
    "print(f'  Fine-tuned  MAE: {mae_ft:.4f}')\n"
    "print(f'  From scratch MAE: {mae_sc:.4f}')\n"
    "print(f'  Improvement: {(mae_sc-mae_ft)/mae_sc*100:.1f}%')\n"
)

# ============================================================
# 4. MULTI-TASK - Rich 3-panel output visualization
# ============================================================

MT_DATA = (
    "# Multi-task targets: structured + correlated\n"
    "np.random.seed(77)\n"
    "t_mt   = np.linspace(0, 4*np.pi, HORIZON)\n"
    "demand = np.tile(np.sin(t_mt)+1, (batch_size,1))       # energy demand (regression)\n"
    "peak_h = np.argmax(demand, axis=1)                      # peak hour (classification)\n"
    "anom   = (np.abs(demand - demand.mean(1,keepdims=True)) > 0.8).astype('float32')  # anomaly\n\n"
    "task1_target = demand.astype('float32')\n"
    "task2_target = keras.utils.to_categorical(peak_h % HORIZON, num_classes=HORIZON).astype('float32')\n"
    "task3_target = np.clip(anom, 0, 1)\n\n"
    "print(f'Multi-task targets -- energy:{task1_target.shape}  '\n"
    "      f'peak_hour:{task2_target.shape}  anomaly:{task3_target.shape}')\n"
)

MT_TRAIN = (
    "print('Training multi-task model (12 epochs)...')\n"
    "history_multitask = multitask_model.fit(\n"
    "    [static_features, dynamic_past, known_future],\n"
    "    [task1_target, task2_target, task3_target],\n"
    "    epochs=12, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n\n"
    "mt_pred_energy, mt_pred_peak, mt_pred_anom = multitask_model.predict(\n"
    "    [static_features, dynamic_past, known_future], verbose=0)\n\n"
    "print('Training complete.')\n"
    "print(f'  Energy demand shape:  {mt_pred_energy.shape}')\n"
    "print(f'  Peak hour shape:      {mt_pred_peak.shape}')\n"
    "print(f'  Anomaly score shape:  {mt_pred_anom.shape}')\n"
)

MT_PLOT = (
    "s = 0\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n\n"
    "# Task 1: Energy demand\n"
    "ax = axes[0]\n"
    "ax.plot(steps, task1_target[s], color='steelblue', lw=2.5, label='Actual demand')\n"
    "ax.plot(steps, mt_pred_energy[s], color='tomato', lw=2, linestyle='--', label='Predicted demand')\n"
    "ax.set_title('Task 1: Energy Demand', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Demand (norm.)')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "# Task 2: Peak hour probability\n"
    "ax = axes[1]\n"
    "ax.bar(steps, mt_pred_peak[s],     color='steelblue', alpha=0.5, label='Predicted prob.')\n"
    "ax.axvline(peak_h[s]+1, color='red', lw=2, linestyle='--', label=f'True peak h={peak_h[s]+1}')\n"
    "ax.set_title('Task 2: Peak Hour Probability', fontsize=12)\n"
    "ax.set_xlabel('Hour'); ax.set_ylabel('Probability')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n\n"
    "# Task 3: Anomaly score\n"
    "ax = axes[2]\n"
    "ax.fill_between(steps, mt_pred_anom[s],  alpha=0.4, color='darkorange', label='Anomaly score')\n"
    "ax.plot(steps,        task3_target[s],   color='black', lw=1.5, label='True anomaly')\n"
    "ax.axhline(0.5, color='red', linestyle=':', lw=1.5, label='Detection threshold')\n"
    "ax.set_title('Task 3: Anomaly Detection', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Score')\n"
    "ax.set_ylim(-0.1, 1.1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Architecture 4 -- Multi-Task Learning: All 3 Outputs', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "# Training history\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "for key, color, label in [\n"
    "    ('energy_demand_loss',  'steelblue',  'Energy demand'),\n"
    "    ('peak_hour_loss',      'darkorange', 'Peak hour'),\n"
    "    ('anomaly_score_loss',  'mediumseagreen', 'Anomaly detection'),\n"
    "    ('loss',                'black',      'Total loss'),\n"
    "]:\n"
    "    if key in history_multitask.history:\n"
    "        ax.plot(history_multitask.history[key], color=color, lw=2, label=label)\n"
    "ax.set_title('Multi-Task Training -- Loss per Task', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# Apply changes
# ============================================================

# Section 1: Ensemble
i = find('Create sample data', 'code')
if i < 0: i = find('batch_size = 32', 'code')
print(f"Ensemble data cell at: {i}")
cells[i] = code_cell(ENS_DATA)

i = find('Create individual BaseAttentive kernels', 'code')
print(f"Ensemble kernels cell at: {i}")
cells[i] = code_cell(ENS_TRAIN)

i = find('Build ensemble model', 'code')
print(f"Ensemble build cell at: {i}")
cells[i] = code_cell(ENS_BUILD)

i = find('Train ensemble', 'code')
print(f"Ensemble fit cell at: {i}")
cells[i] = code_cell(ENS_FIT)

# Insert ensemble plot after the fit cell
cells[i+1:i+1] = [
    md_cell("### Visualization -- Ensemble Architecture"),
    code_cell(ENS_PLOT),
]

# Section 2: Physics-Guided
i = find('Train physics-guided model', 'code')
print(f"PG train cell at: {i}")
if i >= 0:
    cells[i] = code_cell(PG_TRAIN)
    cells[i+1:i+1] = [
        md_cell("### Visualization -- Physics-Guided vs Standard"),
        code_cell(PG_PLOT),
    ]

# Section 3: Transfer Learning
i = find('Pre-train on large dataset', 'code')
print(f"TL pretrain cell at: {i}")
if i >= 0: cells[i] = code_cell(TL_TRAIN)

i = find('Fine-tune on target location', 'code')
print(f"TL finetune cell at: {i}")
if i >= 0:
    cells[i] = code_cell(TL_FINETUNE)
    cells[i+1:i+1] = [
        md_cell("### Visualization -- Transfer Learning: Convergence vs Scratch"),
        code_cell(TL_PLOT),
    ]

# Section 4: Multi-Task
i = find('Create target data for all 3 tasks', 'code')
print(f"MT data cell at: {i}")
if i >= 0: cells[i] = code_cell(MT_DATA)

i = find('Train multi-task model', 'code')
print(f"MT train cell at: {i}")
if i >= 0:
    cells[i] = code_cell(MT_TRAIN)
    cells[i+1:i+1] = [
        md_cell("### Visualization -- Multi-Task: All 3 Task Outputs"),
        code_cell(MT_PLOT),
    ]

print(f"\nFinal cell count: {len(nb['cells'])}")
for j, c in enumerate(cells):
    src = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"  [{j:02d}] {c['cell_type'][:4]} | {src[:65].replace(chr(10),' ')}")

with open('examples/05_kernel_robust_networks.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nDone.")
