"""
Fix 05_kernel_robust_networks.ipynb:
- ENS_TRAIN ended up at [22] (last cell, Python -1 indexing) → move to after [04]
- Missing: MT_TRAIN cell, MT viz cells, summary cell
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

ENS_TRAIN = (
    "# Create 3 individual BaseAttentive kernels\n"
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
    ")\n"
    "print('Created 3 BaseAttentive kernels for ensemble')\n"
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
    "# Task 1: Energy demand regression\n"
    "ax = axes[0]\n"
    "ax.plot(steps, task1_target[s], color='steelblue', lw=2.5, label='Actual demand')\n"
    "ax.plot(steps, mt_pred_energy[s], color='tomato', lw=2, linestyle='--', label='Predicted demand')\n"
    "ax.set_title('Task 1: Energy Demand', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Demand (norm.)')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "# Task 2: Peak hour classification\n"
    "ax = axes[1]\n"
    "ax.bar(steps, mt_pred_peak[s], color='steelblue', alpha=0.5, label='Predicted prob.')\n"
    "ax.axvline(float(peak_h[s])+1, color='red', lw=2, linestyle='--',\n"
    "           label=f'True peak h={int(peak_h[s])+1}')\n"
    "ax.set_title('Task 2: Peak Hour Probability', fontsize=12)\n"
    "ax.set_xlabel('Hour'); ax.set_ylabel('Probability')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n\n"
    "# Task 3: Anomaly score\n"
    "ax = axes[2]\n"
    "ax.fill_between(steps, mt_pred_anom[s], alpha=0.4, color='darkorange', label='Anomaly score')\n"
    "ax.plot(steps, task3_target[s], color='black', lw=1.5, label='True anomaly')\n"
    "ax.axhline(0.5, color='red', linestyle=':', lw=1.5, label='Detection threshold')\n"
    "ax.set_title('Task 3: Anomaly Detection', fontsize=12)\n"
    "ax.set_xlabel('Forecast step'); ax.set_ylabel('Score')\n"
    "ax.set_ylim(-0.1, 1.1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Architecture 4 -- Multi-Task Learning: All 3 Outputs', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "# Training loss curves\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "for key, color, label in [\n"
    "    ('loss',               'black',         'Total loss'),\n"
    "    ('energy_demand_loss', 'steelblue',      'Energy demand'),\n"
    "    ('peak_hour_loss',     'darkorange',     'Peak hour'),\n"
    "    ('anomaly_score_loss', 'mediumseagreen', 'Anomaly'),\n"
    "]:\n"
    "    if key in history_multitask.history:\n"
    "        ax.plot(history_multitask.history[key], color=color, lw=2, label=label)\n"
    "ax.set_title('Multi-Task Training -- Loss per Task', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

SUMMARY = (
    "## Summary: Kernel Architectures\n\n"
    "BaseAttentive works effectively as a neural network kernel:\n\n"
    "| Architecture | Purpose | Key Advantage |\n"
    "|---|---|---|\n"
    "| **Ensemble** | Robust predictions | Variance reduction, uncertainty band |\n"
    "| **Physics-Guided** | Respect domain constraints | Physically plausible outputs |\n"
    "| **Transfer Learning** | Few-shot adaptation | Faster convergence on small datasets |\n"
    "| **Multi-Task** | Correlated predictions | Shared representations, regularization |\n\n"
    "### Key Insights\n"
    "- BaseAttentive as core component maintains flexibility\n"
    "- Combine with standard Keras layers for custom architectures\n"
    "- Each kernel can be independently trained or frozen\n"
    "- GPU acceleration through the selected Keras backend\n"
)

# ---- Fix 1: Remove misplaced ENS_TRAIN from position [22] ----
del cells[22]   # remove wrong last cell

# ---- Fix 2: Insert ENS_TRAIN after ENS_DATA [04] ----
cells[5:5] = [code_cell(ENS_TRAIN)]

# Now the structure is:
# [04] ENS_DATA, [05] ENS_TRAIN (new), [06] ENS_BUILD, [07] ENS_FIT, [08] viz, [09] ENS_PLOT
# ... (physics/TL sections shifted by +1)
# [21] MT model build, [22] MT_DATA (shifted from 21 → 22)

# ---- Fix 3: Insert MT_TRAIN and MT viz after MT_DATA ----
# Find MT_DATA cell (contains 'Multi-task targets')
mt_data_i = next(i for i, c in enumerate(cells)
                 if 'Multi-task targets' in ''.join(c['source'] if isinstance(c['source'],list) else c['source']))
print(f"MT_DATA at: {mt_data_i}")

cells[mt_data_i+1:mt_data_i+1] = [
    code_cell(MT_TRAIN),
    md_cell("### Visualization -- Multi-Task: All 3 Task Outputs"),
    code_cell(MT_PLOT),
]

# ---- Fix 4: Add summary at end ----
cells.append(md_cell(SUMMARY))

# Final check
print(f"\nFinal cells ({len(cells)} total):")
for i, c in enumerate(cells):
    src = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"  [{i:02d}] {c['cell_type'][:4]} | {src[:70].replace(chr(10),' ')}")

with open('examples/05_kernel_robust_networks.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nDone.")
