import json, uuid, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('examples/04_standalone_applications.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def new_id(): return uuid.uuid4().hex[:8]
def code_cell(src):
    return {"cell_type":"code","execution_count":None,"id":new_id(),"metadata":{},"outputs":[],"source":src}
def md_cell(src):
    return {"cell_type":"markdown","id":new_id(),"metadata":{},"source":src}

def do_find(keyword, ctype=None, start=0):
    for i in range(start, len(cells)):
        src = ''.join(cells[i]['source']) if isinstance(cells[i]['source'],list) else cells[i]['source']
        if keyword in src and (ctype is None or cells[i]['cell_type']==ctype):
            return i
    return -1

offset = 0

def replace(keyword, ctype, new_src, start=0):
    i = do_find(keyword, ctype, start + offset)
    if i < 0: print(f"WARNING: not found '{keyword}'"); return -1
    cells[i] = code_cell(new_src) if ctype == 'code' else md_cell(new_src)
    return i

def insert_after(after_keyword, after_ctype, new_cells_list, start=0):
    global offset
    i = do_find(after_keyword, after_ctype, start + offset)
    if i < 0: print(f"WARNING: not found '{after_keyword}'"); return
    cells[i+1:i+1] = new_cells_list
    offset += len(new_cells_list)

# ============================================================
# AIR QUALITY
# ============================================================
AQ_DATA = (
    "# Air Quality -- realistic PM2.5 with daily rush-hour pattern\n"
    "np.random.seed(42)\n"
    "N_AIR, LOOKBACK_AIR, HORIZON_AIR = 128, 72, 24\n\n"
    "def pm25_profile(t):\n"
    "    morning = 22 * np.exp(-0.5 * ((t % 24 - 8)  / 1.5) ** 2)\n"
    "    evening = 18 * np.exp(-0.5 * ((t % 24 - 18) / 2.0) ** 2)\n"
    "    return (20 + morning + evening).astype('float32')\n\n"
    "t_buf = np.arange(LOOKBACK_AIR + HORIZON_AIR + 24)\n"
    "base  = pm25_profile(t_buf)\n"
    "offsets = np.random.randint(0, 24, N_AIR)\n\n"
    "pm25_past   = np.array([base[o : o+LOOKBACK_AIR]                           for o in offsets])\n"
    "pm25_future = np.array([base[o+LOOKBACK_AIR : o+LOOKBACK_AIR+HORIZON_AIR] for o in offsets])\n\n"
    "static_features = np.random.randn(N_AIR, 4).astype('float32')\n"
    "static_features[:, 2] = np.abs(static_features[:, 2]) * 1000\n"
    "static_features[:, 3] = (static_features[:, 3] > 0).astype('float32')\n\n"
    "dynamic_past = np.zeros((N_AIR, LOOKBACK_AIR, 5), dtype='float32')\n"
    "dynamic_past[:,:,0] = pm25_past + np.random.randn(N_AIR, LOOKBACK_AIR).astype('float32') * 3\n"
    "dynamic_past[:,:,1] = np.abs(np.random.randn(N_AIR, LOOKBACK_AIR)).astype('float32') * 40\n"
    "dynamic_past[:,:,2] = np.abs(np.random.randn(N_AIR, LOOKBACK_AIR)).astype('float32') * 25\n"
    "dynamic_past[:,:,3] = 15 + np.random.randn(N_AIR, LOOKBACK_AIR).astype('float32') * 5\n"
    "dynamic_past[:,:,4] = 55 + np.abs(np.random.randn(N_AIR, LOOKBACK_AIR)).astype('float32') * 18\n\n"
    "known_future = np.zeros((N_AIR, HORIZON_AIR, 2), dtype='float32')\n"
    "known_future[:,:,0] = np.abs(np.random.randn(N_AIR, HORIZON_AIR)).astype('float32') * 3\n"
    "known_future[:,:,1] = 15 + np.random.randn(N_AIR, HORIZON_AIR).astype('float32') * 4\n\n"
    "target = np.clip(\n"
    "    pm25_future + np.random.randn(N_AIR, HORIZON_AIR).astype('float32') * 3, 0, 150\n"
    ")[:, :, None]\n\n"
    "print('Air Quality Data Shapes:')\n"
    "for nm, a in [('Static',static_features),('Dynamic',dynamic_past),\n"
    "              ('Future',known_future),('Target PM2.5',target)]:\n"
    "    print(f'  {nm:14s}: {a.shape}')\n"
    "print(f'  PM2.5 past [{dynamic_past[:,:,0].min():.1f}, {dynamic_past[:,:,0].max():.1f}]  '\n"
    "      f'target [{target.min():.1f}, {target.max():.1f}] ug/m3')\n"
)

AQ_MODEL = (
    "import keras\n\n"
    "air_quality_model = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    output_dim=1, forecast_horizon=HORIZON_AIR,\n"
    "    objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},\n"
    "    embed_dim=32, num_heads=4, dropout_rate=0.1, name='air_quality_model',\n"
    ")\n"
    "air_quality_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])\n"
    "print('Training air quality model (15 epochs)...')\n"
    "history_air = air_quality_model.fit(\n"
    "    [static_features, dynamic_past, known_future], target,\n"
    "    epochs=15, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n"
    "print(f'  train_MSE={history_air.history[\"loss\"][-1]:.3f}  '\n"
    "      f'val_MSE={history_air.history[\"val_loss\"][-1]:.3f}  '\n"
    "      f'val_MAE={history_air.history[\"val_mae\"][-1]:.2f} ug/m3')\n"
)

AQ_PRED = (
    "pred_air = air_quality_model.predict([static_features, dynamic_past, known_future], verbose=0)\n"
    "print(f'Prediction shape: {pred_air.shape}')\n"
    "mae_air = float(np.mean(np.abs(pred_air[:,:,0] - target[:,:,0])))\n"
    "print(f'Overall MAE: {mae_air:.2f} ug/m3')\n"
)

AQ_PLOT1 = (
    "import matplotlib.pyplot as plt\n\n"
    "WHO = 25.0\n"
    "hours = np.arange(1, HORIZON_AIR + 1)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "ax = axes[0]\n"
    "ax.plot(history_air.history['loss'],     color='steelblue', lw=2, label='Train MSE')\n"
    "ax.plot(history_air.history['val_loss'], color='tomato',    lw=2, linestyle='--', label='Val MSE')\n"
    "ax.set_title('Training History -- Air Quality', fontsize=12)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "ax = axes[1]\n"
    "palette = ['steelblue', 'darkorange', 'mediumseagreen']\n"
    "for s, c in enumerate(palette):\n"
    "    ax.plot(hours, target[s,:,0],   color=c, lw=2,   label=f'Actual Stn {s+1}')\n"
    "    ax.plot(hours, pred_air[s,:,0], color=c, lw=1.5, linestyle='--', alpha=0.85,\n"
    "            label=f'Forecast Stn {s+1}')\n"
    "ax.axhline(WHO, color='red', linestyle=':', lw=1.5, label=f'WHO limit ({WHO:.0f} ug/m3)')\n"
    "ax.set_title('PM2.5 24-Hour Forecast vs Actual', fontsize=12)\n"
    "ax.set_xlabel('Hour ahead'); ax.set_ylabel('PM2.5 (ug/m3)')\n"
    "ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)\n"
    "plt.suptitle('Application 1 -- Air Quality Forecasting', fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n"
)

AQ_PLOT2 = (
    "fig, ax = plt.subplots(figsize=(13, 4))\n"
    "hist_h  = np.arange(-LOOKBACK_AIR, 0)\n"
    "fcast_h = np.arange(0, HORIZON_AIR)\n"
    "s = 2\n"
    "ax.plot(hist_h,  dynamic_past[s,:,0], color='gray',      lw=1.5, label='Historical PM2.5')\n"
    "ax.plot(fcast_h, target[s,:,0],       color='steelblue', lw=2.5, label='Actual (24h)')\n"
    "ax.plot(fcast_h, pred_air[s,:,0],     color='tomato',    lw=2,   linestyle='--', label='Forecast (24h)')\n"
    "ax.fill_between(fcast_h, pred_air[s,:,0]-5, pred_air[s,:,0]+5,\n"
    "                alpha=0.15, color='tomato', label='+-5 ug/m3')\n"
    "ax.axvline(0, color='black', alpha=0.4, lw=1)\n"
    "ax.axhline(WHO, color='red', linestyle=':', lw=1.5, label='WHO limit')\n"
    "ax.set_title('Station 3 -- History + 24h PM2.5 Forecast', fontsize=13)\n"
    "ax.set_xlabel('Hours relative to forecast start')\n"
    "ax.set_ylabel('PM2.5 (ug/m3)')\n"
    "ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# ENERGY
# ============================================================
EN_DATA = (
    "# Energy Demand -- daily load profile with morning/evening peaks\n"
    "np.random.seed(7)\n"
    "N_EN, LOOKBACK_EN, HORIZON_EN = 128, 96, 48\n\n"
    "def load_profile(t):\n"
    "    base  = 300\n"
    "    peak1 = 350 * np.maximum(0, np.sin(np.pi * (t % 24 - 6) / 14))\n"
    "    peak2 = 180 * np.exp(-0.5 * ((t % 24 - 19) / 2) ** 2)\n"
    "    return (base + peak1 + peak2).astype('float32')\n\n"
    "t_e    = np.arange(LOOKBACK_EN + HORIZON_EN + 24)\n"
    "base_e = load_profile(t_e)\n"
    "offsets_e   = np.random.randint(0, 24, N_EN)\n"
    "load_past   = np.array([base_e[o : o+LOOKBACK_EN]                         for o in offsets_e])\n"
    "load_future = np.array([base_e[o+LOOKBACK_EN : o+LOOKBACK_EN+HORIZON_EN] for o in offsets_e])\n\n"
    "static_energy = np.random.randn(N_EN, 4).astype('float32')\n"
    "static_energy[:,1] = np.abs(static_energy[:,1]) * 10000\n"
    "static_energy[:,3] = np.abs(static_energy[:,3]) * 100\n\n"
    "t_dyn = np.tile(np.arange(LOOKBACK_EN), (N_EN, 1))\n"
    "dynamic_energy = np.zeros((N_EN, LOOKBACK_EN, 5), dtype='float32')\n"
    "dynamic_energy[:,:,0] = load_past + np.random.randn(N_EN,LOOKBACK_EN).astype('float32') * 20\n"
    "dynamic_energy[:,:,1] = 15 + np.random.randn(N_EN,LOOKBACK_EN).astype('float32') * 8\n"
    "dynamic_energy[:,:,2] = np.abs(np.random.randn(N_EN,LOOKBACK_EN)).astype('float32') * 500\n"
    "dynamic_energy[:,:,3] = np.sin(2*np.pi*t_dyn/24)\n"
    "dynamic_energy[:,:,4] = np.cos(2*np.pi*t_dyn/24)\n\n"
    "t_fut_e = np.tile(np.arange(LOOKBACK_EN, LOOKBACK_EN+HORIZON_EN), (N_EN, 1))\n"
    "known_future_energy = np.zeros((N_EN, HORIZON_EN, 2), dtype='float32')\n"
    "known_future_energy[:,:,0] = 15 + np.random.randn(N_EN,HORIZON_EN).astype('float32') * 6\n"
    "known_future_energy[:,:,1] = (t_fut_e % 7 < 5).astype('float32')\n\n"
    "target_energy = np.clip(\n"
    "    load_future + np.random.randn(N_EN, HORIZON_EN).astype('float32') * 15, 100, 900\n"
    ")[:, :, None]\n\n"
    "print('Energy Demand Data Shapes:')\n"
    "for nm, a in [('Static',static_energy),('Dynamic',dynamic_energy),\n"
    "              ('Future',known_future_energy),('Target load',target_energy)]:\n"
    "    print(f'  {nm:14s}: {a.shape}')\n"
    "print(f'  Load range  past [{dynamic_energy[:,:,0].min():.0f}, {dynamic_energy[:,:,0].max():.0f}]  '\n"
    "      f'target [{target_energy.min():.0f}, {target_energy.max():.0f}] kW')\n"
)

EN_MODEL = (
    "energy_model = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    output_dim=1, forecast_horizon=HORIZON_EN,\n"
    "    objective='hybrid',\n"
    "    architecture_config={'decoder_attention_stack': ['cross', 'hierarchical']},\n"
    "    embed_dim=48, num_heads=8, dropout_rate=0.1, name='energy_model',\n"
    ")\n"
    "energy_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])\n"
    "print('Training energy demand model (15 epochs)...')\n"
    "history_energy = energy_model.fit(\n"
    "    [static_energy, dynamic_energy, known_future_energy], target_energy,\n"
    "    epochs=15, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n"
    "print(f'  train_MSE={history_energy.history[\"loss\"][-1]:.1f}  '\n"
    "      f'val_MSE={history_energy.history[\"val_loss\"][-1]:.1f}  '\n"
    "      f'val_MAE={history_energy.history[\"val_mae\"][-1]:.1f} kW')\n"
)

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

# ============================================================
# WEATHER
# ============================================================
WX_DATA = (
    "# Weather -- daily temperature cycle + pressure anti-correlation\n"
    "np.random.seed(13)\n"
    "N_WX, LOOKBACK_WX, HORIZON_WX = 128, 60, 30\n\n"
    "t_wx = np.arange(LOOKBACK_WX + HORIZON_WX + 24)\n"
    "temp_base  = (15 + 7  * np.sin(2*np.pi*t_wx/12 - np.pi/2)).astype('float32')\n"
    "press_base = (1013 - 4 * np.sin(2*np.pi*t_wx/12 - np.pi/2)).astype('float32')\n\n"
    "offsets_wx = np.random.randint(0, 12, N_WX)\n"
    "temp_past   = np.array([temp_base[o : o+LOOKBACK_WX]              for o in offsets_wx])\n"
    "temp_fut    = np.array([temp_base[o+LOOKBACK_WX : o+LOOKBACK_WX+HORIZON_WX]  for o in offsets_wx])\n"
    "press_past  = np.array([press_base[o : o+LOOKBACK_WX]             for o in offsets_wx])\n"
    "press_fut   = np.array([press_base[o+LOOKBACK_WX : o+LOOKBACK_WX+HORIZON_WX] for o in offsets_wx])\n\n"
    "static_weather = np.random.randn(N_WX, 4).astype('float32')\n"
    "static_weather[:,0] = 40 + static_weather[:,0]*10\n"
    "static_weather[:,1] = -100 + static_weather[:,1]*30\n"
    "static_weather[:,2] = np.abs(static_weather[:,2])*2000\n\n"
    "dynamic_weather = np.zeros((N_WX, LOOKBACK_WX, 5), dtype='float32')\n"
    "dynamic_weather[:,:,0] = temp_past  + np.random.randn(N_WX,LOOKBACK_WX).astype('float32')*0.8\n"
    "dynamic_weather[:,:,1] = press_past + np.random.randn(N_WX,LOOKBACK_WX).astype('float32')*2\n"
    "dynamic_weather[:,:,2] = 60 + np.abs(np.random.randn(N_WX,LOOKBACK_WX)).astype('float32')*15\n"
    "dynamic_weather[:,:,3] = np.abs(np.random.randn(N_WX,LOOKBACK_WX)).astype('float32')*4\n"
    "dynamic_weather[:,:,4] = np.abs(np.random.randn(N_WX,LOOKBACK_WX)).astype('float32')*120\n\n"
    "known_future_weather = np.zeros((N_WX, HORIZON_WX, 2), dtype='float32')\n"
    "known_future_weather[:,:,0] = 1\n"
    "known_future_weather[:,:,1] = 4\n\n"
    "target_weather = np.zeros((N_WX, HORIZON_WX, 2), dtype='float32')\n"
    "target_weather[:,:,0] = temp_fut  + np.random.randn(N_WX,HORIZON_WX).astype('float32')*0.5\n"
    "target_weather[:,:,1] = press_fut + np.random.randn(N_WX,HORIZON_WX).astype('float32')*1.5\n\n"
    "print('Weather Data Shapes:')\n"
    "for nm, a in [('Static',static_weather),('Dynamic',dynamic_weather),\n"
    "              ('Future',known_future_weather),('Target (T,P)',target_weather)]:\n"
    "    print(f'  {nm:14s}: {a.shape}')\n"
    "print(f'  Temp range  past [{dynamic_weather[:,:,0].min():.1f}, {dynamic_weather[:,:,0].max():.1f}]C  '\n"
    "      f'target [{target_weather[:,:,0].min():.1f}, {target_weather[:,:,0].max():.1f}]C')\n"
)

WX_MODEL = (
    "weather_model = BaseAttentive(\n"
    "    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,\n"
    "    output_dim=2, forecast_horizon=HORIZON_WX,\n"
    "    objective='transformer',\n"
    "    architecture_config={'decoder_attention_stack': ['cross','hierarchical','memory']},\n"
    "    memory_size=32, embed_dim=64, num_heads=8, dropout_rate=0.1, name='weather_model',\n"
    ")\n"
    "weather_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])\n"
    "print('Training weather model (12 epochs)...')\n"
    "history_weather = weather_model.fit(\n"
    "    [static_weather, dynamic_weather, known_future_weather], target_weather,\n"
    "    epochs=12, batch_size=16, validation_split=0.2, verbose=0,\n"
    ")\n"
    "print(f'  train_MSE={history_weather.history[\"loss\"][-1]:.3f}  '\n"
    "      f'val_MSE={history_weather.history[\"val_loss\"][-1]:.3f}')\n\n"
    "pred_weather = weather_model.predict(\n"
    "    [static_weather, dynamic_weather, known_future_weather], verbose=0)\n"
    "mae_temp  = float(np.mean(np.abs(pred_weather[:,:,0] - target_weather[:,:,0])))\n"
    "mae_press = float(np.mean(np.abs(pred_weather[:,:,1] - target_weather[:,:,1])))\n"
    "print(f'  Temp MAE: {mae_temp:.2f}C    Pressure MAE: {mae_press:.2f} hPa')\n"
)

WX_PLOT = (
    "steps_wx = np.arange(1, HORIZON_WX + 1)\n"
    "s_wx = 1\n\n"
    "fig, axes = plt.subplots(2, 2, figsize=(14, 7))\n\n"
    "ax = axes[0][1]\n"
    "ax.plot(history_weather.history['loss'],     color='royalblue', lw=2, label='Train MSE')\n"
    "ax.plot(history_weather.history['val_loss'], color='tomato',    lw=2, linestyle='--', label='Val MSE')\n"
    "ax.set_title('Training History', fontsize=11)\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "for d, (ca, cp, title, ylab) in enumerate([\n"
    "    ('royalblue', 'tomato',     'Temperature Forecast (C)',  'Temp (C)'),\n"
    "    ('darkgreen', 'darkorange', 'Pressure Forecast (hPa)',   'Pressure (hPa)'),\n"
    "]):\n"
    "    ax = axes[d][0]\n"
    "    ax.plot(steps_wx, target_weather[s_wx,:,d], color=ca, lw=2.5, label='Actual')\n"
    "    ax.plot(steps_wx, pred_weather[s_wx,:,d],   color=cp, lw=2,   linestyle='--', label='Forecast')\n"
    "    ax.fill_between(steps_wx,\n"
    "                    pred_weather[s_wx,:,d]-0.5, pred_weather[s_wx,:,d]+0.5,\n"
    "                    alpha=0.15, color=cp)\n"
    "    ax.set_title(title, fontsize=11)\n"
    "    ax.set_xlabel('2-hour step'); ax.set_ylabel(ylab)\n"
    "    ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "ax = axes[1][1]\n"
    "ax.scatter(target_weather[:5,:,0].ravel(), target_weather[:5,:,1].ravel(),\n"
    "           s=10, alpha=0.4, color='royalblue', label='Actual')\n"
    "ax.scatter(pred_weather[:5,:,0].ravel(),   pred_weather[:5,:,1].ravel(),\n"
    "           s=10, alpha=0.4, color='tomato',    label='Forecast')\n"
    "ax.set_xlabel('Temperature (C)'); ax.set_ylabel('Pressure (hPa)')\n"
    "ax.set_title('Temp-Pressure Relationship', fontsize=11)\n"
    "ax.legend(); ax.grid(True, alpha=0.3)\n\n"
    "plt.suptitle('Application 3 -- Weather Forecasting', fontsize=13)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ============================================================
# TRAFFIC
# ============================================================
TR_DATA = (
    "# Traffic -- double-peak rush-hour volume with inverse speed relationship\n"
    "np.random.seed(99)\n"
    "N_TR, LOOKBACK_TR, HORIZON_TR = 128, 96, 48\n\n"
    "def rush_hour_volume(t):\n"
    "    morning = 1800 * np.exp(-0.5 * ((t % 24 - 8)    / 1.0) ** 2)\n"
    "    evening = 1500 * np.exp(-0.5 * ((t % 24 - 17.5) / 1.5) ** 2)\n"
    "    return (800 + morning + evening).astype('float32')\n\n"
    "def speed_from_volume(vol):\n"
    "    congestion = np.clip(vol / 3500, 0, 1)\n"
    "    return (100 * (1 - 0.65 * congestion)).astype('float32')\n\n"
    "t_tr     = np.arange(LOOKBACK_TR + HORIZON_TR + 24)\n"
    "vol_base = rush_hour_volume(t_tr * 5 / 60)\n"
    "spd_base = speed_from_volume(vol_base)\n\n"
    "offsets_tr = np.random.randint(0, 24, N_TR)\n"
    "vol_past   = np.array([vol_base[o : o+LOOKBACK_TR]                         for o in offsets_tr])\n"
    "vol_future = np.array([vol_base[o+LOOKBACK_TR : o+LOOKBACK_TR+HORIZON_TR] for o in offsets_tr])\n"
    "spd_past   = np.array([spd_base[o : o+LOOKBACK_TR]                         for o in offsets_tr])\n"
    "spd_future = np.array([spd_base[o+LOOKBACK_TR : o+LOOKBACK_TR+HORIZON_TR] for o in offsets_tr])\n\n"
    "static_traffic = np.random.randn(N_TR, 4).astype('float32')\n"
    "static_traffic[:,0] = (np.random.rand(N_TR) > 0.5).astype('float32')\n"
    "static_traffic[:,1] = 70 + np.abs(static_traffic[:,1]) * 30\n"
    "static_traffic[:,2] = np.abs(static_traffic[:,2]) * 3 + 2\n\n"
    "dynamic_traffic = np.zeros((N_TR, LOOKBACK_TR, 4), dtype='float32')\n"
    "dynamic_traffic[:,:,0] = vol_past + np.random.randn(N_TR,LOOKBACK_TR).astype('float32') * 80\n"
    "dynamic_traffic[:,:,1] = spd_past + np.random.randn(N_TR,LOOKBACK_TR).astype('float32') * 3\n"
    "dynamic_traffic[:,:,2] = np.clip(dynamic_traffic[:,:,0]/3500, 0, 1)\n"
    "dynamic_traffic[:,:,3] = (np.random.rand(N_TR,LOOKBACK_TR)>0.97).astype('float32')\n\n"
    "t_fut_tr = np.tile(np.arange(LOOKBACK_TR, LOOKBACK_TR+HORIZON_TR)*5/60, (N_TR,1))\n"
    "known_future_traffic = np.zeros((N_TR, HORIZON_TR, 2), dtype='float32')\n"
    "known_future_traffic[:,:,0] = t_fut_tr % 24\n"
    "known_future_traffic[:,:,1] = (t_fut_tr // 24) % 7 >= 5\n\n"
    "target_traffic = np.zeros((N_TR, HORIZON_TR, 2), dtype='float32')\n"
    "target_traffic[:,:,0] = np.clip(\n"
    "    vol_future + np.random.randn(N_TR,HORIZON_TR).astype('float32')*60, 0, 4500)\n"
    "target_traffic[:,:,1] = np.clip(\n"
    "    spd_future + np.random.randn(N_TR,HORIZON_TR).astype('float32')*2,  20, 110)\n\n"
    "print('Traffic Data Shapes:')\n"
    "for nm, a in [('Static',static_traffic),('Dynamic',dynamic_traffic),\n"
    "              ('Future',known_future_traffic),('Target (vol,spd)',target_traffic)]:\n"
    "    print(f'  {nm:14s}: {a.shape}')\n"
    "print(f'  Volume range  past [{dynamic_traffic[:,:,0].min():.0f}, {dynamic_traffic[:,:,0].max():.0f}]  '\n"
    "      f'target [{target_traffic[:,:,0].min():.0f}, {target_traffic[:,:,0].max():.0f}] veh/h')\n"
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

SUMMARY_PLOT = (
    "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n"
    "app_data = [\n"
    "    ('Air Quality',  hours,    target[0,:,0],          pred_air[0,:,0],          'PM2.5 (ug/m3)',  'steelblue'),\n"
    "    ('Energy',       hours_en, target_energy[0,:,0],   energy_predictions[0,:,0], 'Demand (kW)',    'darkorange'),\n"
    "    ('Weather Temp', steps_wx, target_weather[0,:,0],  pred_weather[0,:,0],      'Temp (C)',       'royalblue'),\n"
    "    ('Traffic Vol',  steps_tr, target_traffic[0,:,0],  traffic_pred[0,:,0],      'Vol (veh/h)',    'mediumseagreen'),\n"
    "]\n"
    "for ax, (title, x, actual, forecast, ylabel, color) in zip(axes.ravel(), app_data):\n"
    "    ax.plot(x, actual,   color=color,    lw=2.5, label='Actual')\n"
    "    ax.plot(x, forecast, color='tomato', lw=2,   linestyle='--', label='Forecast')\n"
    "    ax.set_title(title, fontsize=12); ax.set_ylabel(ylabel)\n"
    "    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)\n"
    "plt.suptitle('BaseAttentive -- Standalone Applications Summary', fontsize=14)\n"
    "plt.tight_layout(); plt.show()\n\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "labels_bar = ['Air Quality\\n(ug/m3)', 'Energy\\n(kW)', 'Weather-T\\n(C)', 'Traffic Vol\\n(veh/h)']\n"
    "maes_bar   = [mae_air, mae_en, mae_temp, mae_vol]\n"
    "colors_bar = ['steelblue', 'darkorange', 'royalblue', 'mediumseagreen']\n"
    "bars = ax.bar(labels_bar, maes_bar, color=colors_bar, width=0.5, edgecolor='white', lw=1.5)\n"
    "for bar, v in zip(bars, maes_bar):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, v*1.02, f'{v:.2f}',\n"
    "            ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
    "ax.set_title('Mean Absolute Error by Application', fontsize=13)\n"
    "ax.set_ylabel('MAE'); ax.grid(True, axis='y', alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
)

# ---- Apply changes ----
replace('Create synthetic air quality data', 'code', AQ_DATA)
replace('Configure BaseAttentive for air quality', 'code', AQ_MODEL)
replace('Make predictions', 'code', AQ_PRED)
insert_after('mae_air', 'code', [
    md_cell("### Visualization -- Air Quality Forecasts"),
    code_cell(AQ_PLOT1),
    code_cell(AQ_PLOT2),
])

replace('Create synthetic energy demand data', 'code', EN_DATA)
replace('Configure BaseAttentive for energy', 'code', EN_MODEL)
en_pred_start = do_find('energy_predictions', None) + offset
replace('Make predictions', 'code', EN_PRED, start=en_pred_start)
insert_after('mae_en', 'code', [
    md_cell("### Visualization -- Energy Demand Forecasts"),
    code_cell(EN_PLOT1),
    code_cell(EN_PLOT2),
])

replace('Create synthetic weather data', 'code', WX_DATA)
replace('Configure and train weather model', 'code', WX_MODEL)
insert_after('mae_press', 'code', [
    md_cell("### Visualization -- Weather Forecasts"),
    code_cell(WX_PLOT),
])

replace('Create synthetic traffic data', 'code', TR_DATA)
replace('Configure and train traffic model', 'code', TR_MODEL)
insert_after('mae_spd', 'code', [
    md_cell("### Visualization -- Traffic Flow Forecasts"),
    code_cell(TR_PLOT1),
    code_cell(TR_PLOT2),
])

# Summary dashboard
summary_i = do_find('## Summary', 'markdown')
cells[summary_i:summary_i] = [
    md_cell("## Summary Dashboard -- All Applications"),
    code_cell(SUMMARY_PLOT),
]

with open('examples/04_standalone_applications.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Done. Total cells: {len(nb['cells'])}")
