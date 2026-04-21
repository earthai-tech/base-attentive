"""
Update 07_v2_spec_registry.ipynb:
- Add training demo (fit + predict with a spec-built model)
- Add spec-vs-keyword comparison section
- Add registry visualization (list registered keys, show component graph)
"""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('examples/07_v2_spec_registry.ipynb', encoding='utf-8') as f:
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

# ── New section content ──────────────────────────────────────────────────────

TRAIN_DEMO_MD = """\
---

## 9 — Training a Spec-Built Model

A `BaseAttentiveV2` model compiles and trains exactly like any Keras model.
Here we train the spec-built model on a small synthetic dataset.\
"""

TRAIN_DEMO_CODE = (
    "import keras\n\n"
    "# Synthetic training data\n"
    "rng = np.random.default_rng(0)\n"
    "B, T, H = 64, 24, 24\n"
    "x_s_train = rng.standard_normal((B, 4)).astype('float32')\n"
    "x_d_train = rng.standard_normal((B, T, 8)).astype('float32')\n"
    "x_f_train = rng.standard_normal((B, H, 6)).astype('float32')\n"
    "y_train   = rng.standard_normal((B, H, 2)).astype('float32')\n\n"
    "# Build a fresh spec-based model for training\n"
    "train_spec = replace(\n"
    "    BASE_SPEC,\n"
    "    embed_dim=32, attention_heads=4,\n"
    ")\n"
    "model_train = BaseAttentive(\n"
    "    static_input_dim=train_spec.static_input_dim,\n"
    "    dynamic_input_dim=train_spec.dynamic_input_dim,\n"
    "    future_input_dim=train_spec.future_input_dim,\n"
    "    output_dim=train_spec.output_dim,\n"
    "    forecast_horizon=train_spec.forecast_horizon,\n"
    "    embed_dim=train_spec.embed_dim,\n"
    "    num_heads=train_spec.attention_heads,\n"
    "    dropout_rate=train_spec.dropout_rate,\n"
    "    name='spec_trained',\n"
    ")\n"
    "model_train.compile(\n"
    "    optimizer=keras.optimizers.Adam(1e-3),\n"
    "    loss='mse', metrics=['mae'],\n"
    ")\n"
    "history = model_train.fit(\n"
    "    [x_s_train, x_d_train, x_f_train], y_train,\n"
    "    epochs=10, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n"
    "print(f'Final train MSE : {history.history[\"loss\"][-1]:.4f}')\n"
    "print(f'Final val   MSE : {history.history[\"val_loss\"][-1]:.4f}')\n\n"
    "y_pred = model_train.predict(\n"
    "    [x_s_train, x_d_train, x_f_train], verbose=0)\n"
    "print(f'Prediction shape: {y_pred.shape}')\n"
)

TRAIN_VIZ_CODE = (
    "import matplotlib.pyplot as plt\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n\n"
    "# Learning curves\n"
    "ax = axes[0]\n"
    "ax.plot(history.history['loss'],     color='steelblue',  lw=2, label='Train MSE')\n"
    "ax.plot(history.history['val_loss'], color='darkorange', lw=2, linestyle='--', label='Val MSE')\n"
    "ax.set_title('Spec-Built Model: Training Curves', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "# Forecast vs actual (sample 0, output dim 0)\n"
    "ax = axes[1]\n"
    "steps = np.arange(1, H + 1)\n"
    "ax.plot(steps, y_train[0, :, 0],  color='steelblue',  lw=2.5, label='Actual')\n"
    "ax.plot(steps, y_pred[0, :, 0],   color='darkorange', lw=2,   linestyle='--', label='Predicted')\n"
    "ax.set_title('Sample Forecast (Output Dim 0)', fontsize=12)\n"
    "ax.set_xlabel('Horizon step'); ax.set_ylabel('Value')\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Section 9 -- Training a Spec-Built Model', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n"
)

COMPARE_MD = """\
---

## 10 — Spec vs Keyword: Side-by-Side Comparison

Both `BaseAttentive` (keyword) and `BaseAttentiveV2` (spec) produce
equivalent models. The spec approach adds reproducibility and
configurability via `dataclasses.replace`.\
"""

COMPARE_CODE = (
    "# Build keyword model\n"
    "kw_params = dict(\n"
    "    static_input_dim=4, dynamic_input_dim=8,\n"
    "    future_input_dim=6, output_dim=2,\n"
    "    forecast_horizon=24, embed_dim=32, num_heads=4,\n"
    "    dropout_rate=0.1,\n"
    ")\n"
    "model_kw2 = BaseAttentive(**kw_params, name='keyword_v')\n\n"
    "# Build spec model\n"
    "cmp_spec = replace(BASE_SPEC, embed_dim=32, attention_heads=4)\n"
    "model_sp2 = BaseAttentive(\n"
    "    static_input_dim=cmp_spec.static_input_dim,\n"
    "    dynamic_input_dim=cmp_spec.dynamic_input_dim,\n"
    "    future_input_dim=cmp_spec.future_input_dim,\n"
    "    output_dim=cmp_spec.output_dim,\n"
    "    forecast_horizon=cmp_spec.forecast_horizon,\n"
    "    embed_dim=cmp_spec.embed_dim,\n"
    "    num_heads=cmp_spec.attention_heads,\n"
    "    dropout_rate=cmp_spec.dropout_rate,\n"
    "    name='spec_v',\n"
    ")\n\n"
    "# Compare parameter counts\n"
    "x_s2 = rng.standard_normal((4, 4)).astype('float32')\n"
    "x_d2 = rng.standard_normal((4, 24, 8)).astype('float32')\n"
    "x_f2 = rng.standard_normal((4, 24, 6)).astype('float32')\n"
    "_ = model_kw2([x_s2, x_d2, x_f2])\n"
    "_ = model_sp2([x_s2, x_d2, x_f2])\n\n"
    "kw_params_n  = model_kw2.count_params()\n"
    "sp_params_n  = model_sp2.count_params()\n"
    "print(f'Keyword model params : {kw_params_n:,}')\n"
    "print(f'Spec    model params : {sp_params_n:,}')\n"
    "print(f'Match: {kw_params_n == sp_params_n}')\n"
)

COMPARE_VIZ_CODE = (
    "# Visual comparison: keyword dict vs spec fields\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n\n"
    "# Left: keyword dict display\n"
    "ax = axes[0]\n"
    "ax.axis('off')\n"
    "kw_text = '\\n'.join(\n"
    "    [f'{k} = {v}' for k, v in kw_params.items()]\n"
    ")\n"
    "ax.text(0.05, 0.95, 'BaseAttentive(\\n  ' + kw_text.replace('\\n','\\n  ') + '\\n)',\n"
    "        transform=ax.transAxes, fontsize=9, verticalalignment='top',\n"
    "        fontfamily='monospace',\n"
    "        bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))\n"
    "ax.set_title('Keyword Approach', fontsize=12, pad=10)\n\n"
    "# Right: spec fields display\n"
    "ax = axes[1]\n"
    "ax.axis('off')\n"
    "sp_text = (\n"
    "    f'embed_dim        = {cmp_spec.embed_dim}\\n'\n"
    "    f'attention_heads  = {cmp_spec.attention_heads}\\n'\n"
    "    f'dropout_rate     = {cmp_spec.dropout_rate}\\n'\n"
    "    f'forecast_horizon = {cmp_spec.forecast_horizon}\\n'\n"
    "    f'static_input_dim = {cmp_spec.static_input_dim}\\n'\n"
    "    f'dynamic_input_dim= {cmp_spec.dynamic_input_dim}\\n'\n"
    "    f'future_input_dim = {cmp_spec.future_input_dim}\\n'\n"
    "    f'output_dim       = {cmp_spec.output_dim}'\n"
    ")\n"
    "ax.text(0.05, 0.95, 'BaseAttentiveSpec(\\n  ' + sp_text.replace('\\n','\\n  ') + '\\n)',\n"
    "        transform=ax.transAxes, fontsize=9, verticalalignment='top',\n"
    "        fontfamily='monospace',\n"
    "        bbox=dict(boxstyle='round', facecolor='#f0f8e8', alpha=0.8))\n"
    "ax.set_title('Spec Approach', fontsize=12, pad=10)\n\n"
    "plt.suptitle('Section 10 -- Keyword vs Spec: Equivalent Models', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n"
    "print('\\nAdvantages of Spec approach:')\n"
    "print('  * Immutable: prevents accidental mutation')\n"
    "print('  * JSON-serialisable: saved/reloaded exactly')\n"
    "print('  * Composable: derive variants with replace()')\n"
    "print('  * Documented: typed fields with defaults')\n"
)

REGISTRY_MD = """\
---

## 11 — Registry Inspection & Visualization

`ComponentRegistry` stores all registered builder functions by string key.
Inspecting the registry helps discover available components.\
"""

REGISTRY_CODE = (
    "from base_attentive.registry import component_registry\n\n"
    "# List all registered keys\n"
    "all_keys = list(component_registry._registry.keys())\n"
    "print(f'Total registered components: {len(all_keys)}')\n"
    "print()\n"
    "for key in sorted(all_keys):\n"
    "    print(f'  {key}')\n"
)

REGISTRY_VIZ_CODE = (
    "# Group keys by prefix (encoder, decoder, etc.)\n"
    "from collections import defaultdict\n\n"
    "groups = defaultdict(list)\n"
    "for key in sorted(all_keys):\n"
    "    prefix = key.split('.')[0] if '.' in key else 'other'\n"
    "    groups[prefix].append(key)\n\n"
    "fig, ax = plt.subplots(figsize=(12, max(4, len(all_keys)*0.35 + 1)))\n"
    "ax.axis('off')\n\n"
    "y = 0.98\n"
    "colors = ['#3498db','#e67e22','#2ecc71','#9b59b6','#e74c3c','#1abc9c','#f39c12']\n"
    "for gi, (grp, keys) in enumerate(sorted(groups.items())):\n"
    "    color = colors[gi % len(colors)]\n"
    "    ax.text(0.01, y, grp.upper(), transform=ax.transAxes,\n"
    "            fontsize=10, fontweight='bold', color=color)\n"
    "    y -= 0.05\n"
    "    for key in keys:\n"
    "        ax.text(0.04, y, f'  {key}', transform=ax.transAxes,\n"
    "                fontsize=9, fontfamily='monospace', color='#333333')\n"
    "        y -= 0.04\n"
    "    y -= 0.02\n\n"
    "ax.set_title('ComponentRegistry: All Registered Keys', fontsize=13, pad=15)\n"
    "plt.tight_layout(); plt.show()\n"
)

UPDATED_SUMMARY = """\
---

## Summary

| Concept | Purpose |
|---------|--------|
| `BaseAttentiveSpec` | Frozen, JSON-serialisable model blueprint |
| `BaseAttentiveComponentSpec` | Declarative selection of component keys |
| `ComponentRegistry.register()` | Plug in a custom builder by string key |
| `assemble_model()` | Resolve a `BaseAttentiveV2Assembly` from a spec |
| `BaseAttentiveV2` | Trainable resolver-driven model scaffold |
| `serialize_base_attentive_spec()` | Stable JSON export for saved experiments |
| `dataclasses.replace()` | Derive spec variants without mutation |
| `component_registry._registry` | Inspect all available component keys |

### Key Takeaways

- **Spec = reproducibility**: a spec file fully defines the model — no code needed
- **Keyword approach** remains valid for quick prototyping
- **Spec approach** shines for experiment tracking, hyperparameter sweeps, and deployment
- The registry is extensible: register any builder function by a string key\
"""

# ── Apply changes ─────────────────────────────────────────────────────────────

# Find the summary cell (last cell [31])
summary_i = find('## Summary', 'markdown')
print(f"Summary at: [{summary_i}]")

# Insert new sections before the summary
insert_cells = [
    md_cell(TRAIN_DEMO_MD),
    code_cell(TRAIN_DEMO_CODE),
    code_cell(TRAIN_VIZ_CODE),
    md_cell(COMPARE_MD),
    code_cell(COMPARE_CODE),
    code_cell(COMPARE_VIZ_CODE),
    md_cell(REGISTRY_MD),
    code_cell(REGISTRY_CODE),
    code_cell(REGISTRY_VIZ_CODE),
]

cells[summary_i:summary_i] = insert_cells
print(f"Inserted {len(insert_cells)} cells before summary")

# Replace the summary cell with updated version
new_summary_i = summary_i + len(insert_cells)
cells[new_summary_i] = md_cell(UPDATED_SUMMARY)
print(f"Replaced summary at [{new_summary_i}]")

print(f"\nFinal cells: {len(cells)}")
for j, c in enumerate(cells):
    src = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
    line = src[:70].replace(chr(10),' ')
    print(f"  [{j:02d}] {c['cell_type'][:4]} | {line}")

with open('examples/07_v2_spec_registry.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nDone.")
