"""
Fix the partial update of 04_standalone_applications.ipynb:
- Cell [13] still has old energy predictions content → replace with EN_PRED
- Cell [21] has EN_PRED instead of traffic model → replace with TR_MODEL
- EN viz cells [22-24] are misplaced (after traffic section) → remove + re-insert after [13]
- Add TR viz cells after TR_MODEL
"""
import json, uuid

with open('examples/04_standalone_applications.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def new_id(): return uuid.uuid4().hex[:8]
def code_cell(src):
    return {"cell_type":"code","execution_count":None,"id":new_id(),"metadata":{},"outputs":[],"source":src}
def md_cell(src):
    return {"cell_type":"markdown","id":new_id(),"metadata":{},"source":src}

# Print current state
print("Current cells:")
for i, c in enumerate(cells):
    src = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"  [{i:02d}] {c['cell_type'][:4]} | {src[:70].replace(chr(10),' ')}")

EN_PRED = (
    "energy_predictions = energy_model.predict(\n"
    "    [static_energy, dynamic_energy, known_future_energy], verbose=0)\n"
    "print(f'Prediction shape: {energy_predictions.shape}')\n"
    "mae_en = float(np.mean(np.abs(energy_predictions[:,:,0] - target_energy[:,:,0])))\n"
    "print(f'Overall MAE: {mae_en:.1f} kW')\n"
)

EN_PLOT1 = (
    "hours_en = np.arange(1, HORIZON_EN + 1)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "ax = axes[0]\n"
    "ax.plot(history_energy.history['loss'],     color='darkorange', lw=2, label='Train MSE')\n"
    "ax.plot(history_energy.history['val_loss'], color='crimson',    lw=2, linestyle='--', label='Val MSE')\n"
    "ax.set_title('Training History -- Energy Demand', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "ax = axes[1]\n"
    "s = 0\n"
    "ax.plot(hours_en, target_energy[s,:,0],      color='steelblue',  lw=2.5, label='Actual demand')\n"
    "ax.plot(hours_en, energy_predictions[s,:,0], color='darkorange', lw=2,   linestyle='--', label='Forecast')\n"
    "ax.fill_between(hours_en,\n"
    "    energy_predictions[s,:,0]*0.95, energy_predictions[s,:,0]*1.05,\n"
    "    alpha=0.2, color='darkorange', label='+-5% band')\n"
    "for day in range(2):\n"
    "    bh = day * 24\n"
    "    ax.axvspan(bh+1, bh+7,   alpha=0.07, color='navy', label='Night' if day==0 else '')\n"
    "    ax.axvspan(bh+19, bh+24, alpha=0.07, color='navy')\n"
    "ax.set_title('48-Hour Energy Demand Forecast', fontsize=12)\n"
    "ax.set_xlabel('Hour ahead'); ax.set_ylabel('Demand (kW)')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.suptitle('Application 2 -- Energy Demand Forecasting', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n"
)

EN_PLOT2 = (
    "fig, ax = plt.subplots(figsize=(13, 4))\n"
    "for s, c in enumerate(['steelblue','darkorange','mediumseagreen']):\n"
    "    ax.plot(hours_en, target_energy[s,:,0],      color=c, lw=2,   label=f'Actual Bldg {s+1}')\n"
    "    ax.plot(hours_en, energy_predictions[s,:,0], color=c, lw=1.5, linestyle=':',\n"
    "            alpha=0.9, label=f'Forecast Bldg {s+1}')\n"
    "for day in range(2):\n"
    "    bh = day * 24\n"
    "    ax.axvspan(bh+1, bh+7,   alpha=0.06, color='navy')\n"
    "    ax.axvspan(bh+19, bh+24, alpha=0.06, color='navy')\n"
    "ax.set_title('3 Buildings -- 48h Energy Demand Forecast', fontsize=13)\n"
    "ax.set_xlabel('Hour ahead'); ax.set_ylabel('Demand (kW)')\n"
    "ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

TR_MODEL = (
    "traffic_model = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=4, future_input_dim=2,\n"
    "    output_dim=2, forecast_horizon=HORIZON_TR,\n"
    "    objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['cross','hierarchical']},\n"
    "    embed_dim=48, num_heads=4, dropout_rate=0.1, name='traffic_model',\n"
    ")\n"
    "traffic_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])\n"
    "print('Training traffic model (12 epochs)...')\n"
    "history_traffic = traffic_model.fit(\n"
    "    [static_traffic, dynamic_traffic, known_future_traffic], target_traffic,\n"
    "    epochs=12, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n"
    "print(f'  train_MSE={history_traffic.history[\"loss\"][-1]:.1f}  '\n"
    "      f'val_MSE={history_traffic.history[\"val_loss\"][-1]:.1f}')\n\n"
    "traffic_pred = traffic_model.predict(\n"
    "    [static_traffic, dynamic_traffic, known_future_traffic], verbose=0)\n"
    "mae_vol = float(np.mean(np.abs(traffic_pred[:,:,0] - target_traffic[:,:,0])))\n"
    "mae_spd = float(np.mean(np.abs(traffic_pred[:,:,1] - target_traffic[:,:,1])))\n"
    "print(f'  Volume MAE: {mae_vol:.0f} veh/h    Speed MAE: {mae_spd:.1f} km/h')\n"
)

TR_PLOT1 = (
    "steps_tr = np.arange(1, HORIZON_TR + 1) * 5\n"
    "s_tr = 3\n\n"
    "fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)\n"
    "ax = axes[0]\n"
    "ax.plot(steps_tr, target_traffic[s_tr,:,0], color='steelblue', lw=2.5, label='Actual volume')\n"
    "ax.plot(steps_tr, traffic_pred[s_tr,:,0],   color='tomato',    lw=2,   linestyle='--', label='Forecast volume')\n"
    "ax.fill_between(steps_tr,\n"
    "    traffic_pred[s_tr,:,0]*0.92, traffic_pred[s_tr,:,0]*1.08,\n"
    "    alpha=0.15, color='tomato', label='+-8% band')\n"
    "ax.set_ylabel('Volume (veh/h)')\n"
    "ax.set_title('Traffic Volume Forecast', fontsize=12)\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n\n"
    "ax = axes[1]\n"
    "ax.plot(steps_tr, target_traffic[s_tr,:,1], color='darkorange',     lw=2.5, label='Actual speed')\n"
    "ax.plot(steps_tr, traffic_pred[s_tr,:,1],   color='mediumseagreen', lw=2,   linestyle='--', label='Forecast speed')\n"
    "ax.axhline(50, color='red', linestyle=':', lw=1.5, label='Congestion threshold 50 km/h')\n"
    "ax.set_xlabel('Minutes ahead'); ax.set_ylabel('Speed (km/h)')\n"
    "ax.set_title('Traffic Speed Forecast', fontsize=12)\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.suptitle('Application 4 -- Traffic Flow Prediction', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n"
)

TR_PLOT2 = (
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "ax = axes[0]\n"
    "ax.scatter(target_traffic[:8,:,0].ravel(), target_traffic[:8,:,1].ravel(),\n"
    "           s=8, alpha=0.4, color='steelblue', label='Actual')\n"
    "ax.scatter(traffic_pred[:8,:,0].ravel(),   traffic_pred[:8,:,1].ravel(),\n"
    "           s=8, alpha=0.4, color='tomato',    label='Forecast')\n"
    "ax.set_xlabel('Volume (veh/h)'); ax.set_ylabel('Speed (km/h)')\n"
    "ax.set_title('Volume-Speed Relationship', fontsize=12)\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "ax = axes[1]\n"
    "ax.plot(history_traffic.history['loss'],     color='steelblue', lw=2, label='Train MSE')\n"
    "ax.plot(history_traffic.history['val_loss'], color='tomato',    lw=2, linestyle='--', label='Val MSE')\n"
    "ax.set_title('Traffic Model -- Training History', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True, alpha=0.3)\n"
    "plt.suptitle('Application 4 -- Traffic Analysis', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ---- Targeted fix ----
# Step 1: Remove misplaced EN viz cells [22, 23, 24] (they are EN viz header, EN_PLOT1, EN_PLOT2)
# These are currently after cell [21] (which is the wrong EN_PRED at traffic position)
del cells[22:25]   # remove 3 cells → now 25 cells total

# Step 2: Replace cell [13] with EN_PRED
cells[13] = code_cell(EN_PRED)

# Step 3: Insert EN viz cells after [13]
cells[14:14] = [
    md_cell("### Visualization -- Energy Demand Forecasts"),
    code_cell(EN_PLOT1),
    code_cell(EN_PLOT2),
]
# Now: +3 cells → 28 total. Old [21] (EN_PRED wrong) is now at [24].

# Step 4: Replace cell [24] (was [21] = wrong EN_PRED) with TR_MODEL
cells[24] = code_cell(TR_MODEL)

# Step 5: Insert TR viz after cell [24]
cells[25:25] = [
    md_cell("### Visualization -- Traffic Flow Forecasts"),
    code_cell(TR_PLOT1),
    code_cell(TR_PLOT2),
]

# Verify
print("\nFixed cells:")
for i, c in enumerate(cells):
    src = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"  [{i:02d}] {c['cell_type'][:4]} | {src[:70].replace(chr(10),' ')}")

with open('examples/04_standalone_applications.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nDone. Total cells: {len(nb['cells'])}")
