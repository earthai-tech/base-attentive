Applications and Use Cases
===========================

This page walks through real-world applications of BaseAttentive with
complete v2 configuration examples.  Each section covers the input
structure, recommended v2 configuration, backend choice, and common
extension patterns.

.. contents:: On this page
   :local:
   :depth: 2

----

V2 Configuration Patterns
--------------------------

Before diving into domain examples, here is a summary of the v2 patterns
used throughout this page.

Keyword-argument style (quick)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most applications you can configure everything through keyword arguments:

.. code-block:: python

   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       mode="tft",
       embed_dim=64,
       num_heads=8,
       quantiles=[0.1, 0.5, 0.9],
   )

Spec-based style (reproducible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you need reproducibility, hyperparameter search, or config-file driven
workflows, use ``BaseAttentiveSpec``:

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       hidden_units=128,
       attention_heads=8,
       dropout_rate=0.1,
       backend_name="torch",
       head_type="quantile",
       quantiles=(0.1, 0.5, 0.9),
       components=BaseAttentiveComponentSpec(
           sequence_pooling="pool.last",
       ),
   )
   model = BaseAttentiveV2Assembly().build(spec)

Choosing a backend per application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Backend
     - Typical use
   * - TensorFlow
     - Deployment (TF Serving, TFLite, SavedModel), CI/CD pipelines
   * - Torch
     - Research iteration, CUDA/MPS GPU acceleration, custom autograd
   * - JAX
     - Batch parallelism, TPU training, functional/stateless workflows

----

Standalone Forecasting Applications
====================================

Air Quality Forecasting
-----------------------

**Challenge:** Air pollution varies with meteorology, human activity, and
geography.  Cities need real-time forecasts for health alerts across
multiple monitoring stations.

**Input structure:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Stream
     - Features
     - Example values
   * - Static (4)
     - Latitude, longitude, elevation, urban index
     - ``[48.85, 2.35, 35.0, 0.82]``
   * - Dynamic (5, T=168)
     - PM2.5, NO₂, O₃, temperature, relative humidity
     - 7 days × hourly
   * - Future (2, H=24)
     - Wind speed forecast, temperature forecast
     - Next 24 hours from NWP model

**V2 configuration:**

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive
   from base_attentive.components import CRPSLoss

   air_model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=5,
       future_input_dim=2,
       output_dim=1,           # PM2.5
       forecast_horizon=24,
       # ── architecture ────────────────────────────────
       mode="tft",             # VSN + gated residuals + cross attention
       embed_dim=64,
       num_heads=8,
       scales=[1, 2, 4],       # capture hourly / 2h / 4h patterns
       multi_scale_agg="average",
       # ── output ──────────────────────────────────────
       quantiles=[0.1, 0.5, 0.9],
   )

   air_model.compile(
       optimizer="adam",
       loss=CRPSLoss(mode="quantile", quantiles=[0.1, 0.5, 0.9]),
   )
   air_model.fit(
       [static_features, historical_obs, weather_forecast],
       targets,
       epochs=50,
       batch_size=64,
   )

   predictions = air_model.predict([test_static, test_dynamic, test_future])
   # shape: (batch, 24, 3, 1) — horizon × quantiles × output_dim

**Spec-based version (for experiment tracking):**

.. code-block:: python

   import json
   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   air_spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=5,
       future_input_dim=2,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       hidden_units=128,
       attention_heads=8,
       dropout_rate=0.1,
       backend_name="tensorflow",
       head_type="quantile",
       quantiles=(0.1, 0.5, 0.9),
   )

   # Save spec to JSON for reproducibility
   with open("air_quality_spec.json", "w") as f:
       json.dump(air_spec.__dict__, f, indent=2)

   air_model = BaseAttentiveV2Assembly().build(air_spec)

**Use cases:** Health index alerts, school/event planning, industrial
emission monitoring, vulnerable population notifications.

----

Energy Demand Forecasting
--------------------------

**Challenge:** Electric grids must balance supply and demand in real-time.
Peak demand prediction enables optimal resource allocation and demand-response
activation.

**Input structure:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Stream
     - Features
     - Example values
   * - Static (5)
     - Building type, floor area, insulation, HVAC capacity, solar flag
     - One-hot encoded + continuous
   * - Dynamic (6, T=336)
     - Hourly load, temperature, solar irradiance, hour_sin, hour_cos, dow_sin
     - 2 weeks × hourly
   * - Future (3, H=48)
     - Temperature forecast, day type, planned events
     - Deterministic calendar + NWP

**V2 configuration:**

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   energy_model = BaseAttentive(
       static_input_dim=5,
       dynamic_input_dim=6,
       future_input_dim=3,
       output_dim=1,
       forecast_horizon=48,      # 2-day forecast
       # ── architecture ────────────────────────────────
       objective="hybrid",
       scales=[1, 2, 4, 8],      # 1h / 2h / 4h / 8h patterns
       multi_scale_agg="average",
       attention_levels=["cross", "hierarchical"],
       embed_dim=64,
       num_heads=8,
       # ── regularisation ──────────────────────────────
       dropout_rate=0.1,
       use_vsn=True,
       # ── output ──────────────────────────────────────
       quantiles=[0.1, 0.5, 0.9],
   )
   energy_model.compile(optimizer="adam", loss="mse")

**Multi-building portfolio pattern:**

When deploying across many buildings, keep one spec and swap only the
data — this ensures identical architecture across instances:

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   base_spec = BaseAttentiveSpec(
       static_input_dim=5,
       dynamic_input_dim=6,
       future_input_dim=3,
       output_dim=1,
       forecast_horizon=48,
       embed_dim=64,
       attention_heads=8,
       backend_name="tensorflow",
       head_type="point",
   )

   building_models = {}
   for building_id, data in buildings.items():
       model = BaseAttentiveV2Assembly().build(base_spec)
       model.compile(optimizer="adam", loss="mse")
       model.fit(data["x_train"], data["y_train"], epochs=30, verbose=0)
       building_models[building_id] = model

**Use cases:** Grid balancing, demand response, renewable integration,
peak shaving, smart buildings.

----

Weather Prediction
------------------

**Challenge:** Weather systems exhibit multi-scale dynamics — synoptic
patterns (days), mesoscale events (hours), and local effects (minutes).
A model must capture all simultaneously.

**Input structure:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Stream
     - Features
     - Example values
   * - Static (4)
     - Latitude, longitude, elevation, terrain type
     - Continuous + categorical
   * - Dynamic (7, T=120)
     - Temperature, pressure, RH, wind_u, wind_v, cloud cover, precip
     - 10 days × 2-hourly
   * - Future (4, H=30)
     - Seasonal sin/cos, jet stream index, El Niño index, forecast hour
     - Deterministic

**V2 configuration:**

.. code-block:: python

   from base_attentive import BaseAttentive
   from base_attentive.components import CRPSLoss

   weather_model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=7,
       future_input_dim=4,
       output_dim=3,              # temperature, pressure, precipitation
       forecast_horizon=30,
       # ── architecture ────────────────────────────────
       mode="pihal",              # multi-scale LSTM + memory + hierarchical
       scales=[1, 2, 4],
       multi_scale_agg="average",
       attention_levels=["cross", "hierarchical", "memory"],
       memory_size=64,
       embed_dim=64,
       num_heads=8,
       num_encoder_layers=4,
       # ── output ──────────────────────────────────────
       quantiles=[0.1, 0.5, 0.9],
   )

   weather_model.compile(
       optimizer="adam",
       loss=CRPSLoss(mode="quantile", quantiles=[0.1, 0.5, 0.9]),
   )

**Using JAX backend for TPU training:**

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "jax"

   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   weather_spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=7,
       future_input_dim=4,
       output_dim=3,
       forecast_horizon=30,
       embed_dim=128,
       attention_heads=16,
       dropout_rate=0.1,
       backend_name="jax",          # JAX for TPU
       head_type="quantile",
       quantiles=(0.1, 0.5, 0.9),
   )
   weather_model = BaseAttentiveV2Assembly().build(weather_spec)

**Use cases:** NWP post-processing, agricultural planning, renewable
energy siting, disaster early warning.

----

Traffic Flow Prediction
-----------------------

**Challenge:** Traffic patterns have strong periodic structure (rush hours,
weekdays vs weekends) but also exhibit abrupt changes (incidents, events,
weather).

**Input structure:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Stream
     - Features
     - Example values
   * - Static (4)
     - Road type, lanes, speed limit, urban flag
     - One-hot + continuous
   * - Dynamic (5, T=288)
     - Volume, speed, occupancy, incident flag, weather effect
     - 24 h × 5-minute
   * - Future (5, H=48)
     - Hour_sin, hour_cos, day_of_week, known events, weather score
     - Deterministic calendar + forecast

**V2 configuration:**

.. code-block:: python

   from base_attentive import BaseAttentive

   traffic_model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=5,
       future_input_dim=5,
       output_dim=2,              # volume, speed
       forecast_horizon=48,       # 4 h at 5-minute resolution
       # ── architecture ────────────────────────────────
       mode="tft",
       attention_levels=["cross", "hierarchical"],
       scales=[1, 3, 6, 12],      # 5 / 15 / 30 / 60-min patterns
       multi_scale_agg="average",
       embed_dim=64,
       num_heads=8,
       # ── output ──────────────────────────────────────
       quantiles=[0.1, 0.5, 0.9],
   )
   traffic_model.compile(optimizer="adam", loss="mse")

**Using Torch backend with CUDA:**

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"

   from base_attentive.backend import TorchDeviceManager
   dm = TorchDeviceManager(prefer="cuda")
   print(dm.device)   # "cuda:0" or "cpu"

   from base_attentive import BaseAttentive
   traffic_model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=5,
       future_input_dim=5,
       output_dim=2,
       forecast_horizon=48,
       mode="tft",
       embed_dim=64,
   )

**Use cases:** Navigation systems, congestion pricing, signal control,
public-transit prioritisation, emergency routing.

----

BaseAttentive as a Kernel in Larger Models
==========================================

V2 makes ``BaseAttentive`` particularly well suited as a reusable kernel.
Because every component is registered and resolved at build time, you can
share a single spec across multiple wrapper models while swapping the outer
logic for each application.

Wrapper pattern — shared spec, different heads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import keras
   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   # One spec, two applications
   shared_spec = BaseAttentiveSpec(
       static_input_dim=5,
       dynamic_input_dim=8,
       future_input_dim=4,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       attention_heads=8,
       backend_name="tensorflow",
       head_type="point",
   )

   class DemandForecaster(keras.Model):
       def __init__(self):
           super().__init__()
           self.kernel    = BaseAttentiveV2Assembly().build(shared_spec)
           self.bias_head = keras.layers.Dense(1)

       def call(self, inputs, training=False):
           base = self.kernel(inputs, training=training)
           ctx  = keras.ops.mean(inputs[1], axis=1)   # dynamic mean
           bias = keras.ops.expand_dims(self.bias_head(ctx), axis=1)
           return base + bias

   class AnomalyForecaster(keras.Model):
       def __init__(self):
           super().__init__()
           self.kernel      = BaseAttentiveV2Assembly().build(shared_spec)
           self.anomaly_out = keras.layers.Dense(1, activation="sigmoid")

       def call(self, inputs, training=False):
           base = self.kernel(inputs, training=training)
           ctx  = keras.ops.mean(base, axis=1)
           return {
               "forecast": base,
               "anomaly_score": self.anomaly_out(ctx),
           }

   demand_model  = DemandForecaster()
   anomaly_model = AnomalyForecaster()

Kernel with custom registered encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register a domain-specific encoder once, then use it across any number of
specs:

.. code-block:: python

   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY

   def seismic_encoder(*, context, units, hidden_units, **kw):
       """Short-window STA/LTA-inspired encoder for seismic signals."""
       import keras
       inp = keras.Input(shape=(None, units))
       # Short-term average
       sta = keras.layers.Conv1D(hidden_units, kernel_size=10,
                                  padding="causal", activation="relu")(inp)
       # Long-term average
       lta = keras.layers.Conv1D(hidden_units, kernel_size=50,
                                  padding="causal", activation="relu")(inp)
       x   = keras.layers.Concatenate()([sta, lta])
       x   = keras.layers.Dense(hidden_units, activation="relu")(x)
       return keras.Model(inp, x, name="seismic_encoder")

   DEFAULT_COMPONENT_REGISTRY.register(
       "encoder.seismic_stalta",
       seismic_encoder,
       backend="generic",
       description="STA/LTA-inspired encoder for seismic time series.",
   )

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   seismic_spec = BaseAttentiveSpec(
       static_input_dim=3,       # lat, lon, depth
       dynamic_input_dim=3,      # Z, N, E components
       future_input_dim=0,
       output_dim=1,
       forecast_horizon=12,
       embed_dim=64,
       hidden_units=128,
       components=BaseAttentiveComponentSpec(
           temporal_encoder="encoder.seismic_stalta",
       ),
   )
   seismic_model = BaseAttentiveV2Assembly().build(seismic_spec)

----

Ensemble Methods
----------------

Combine multiple ``BaseAttentiveSpec`` configs that differ in architecture
while sharing the same outer training loop:

.. code-block:: python

   import numpy as np
   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   base_dims = dict(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=4,
       output_dim=1,
       forecast_horizon=24,
       backend_name="tensorflow",
       head_type="point",
   )

   specs = [
       BaseAttentiveSpec(**base_dims, embed_dim=32,  attention_heads=4,
                         dropout_rate=0.1),   # lightweight
       BaseAttentiveSpec(**base_dims, embed_dim=64,  attention_heads=8,
                         dropout_rate=0.1),   # medium
       BaseAttentiveSpec(**base_dims, embed_dim=128, attention_heads=16,
                         dropout_rate=0.2),   # large + more regularisation
   ]

   members = []
   for spec in specs:
       m = BaseAttentiveV2Assembly().build(spec)
       m.compile(optimizer="adam", loss="mse")
       m.fit(train_x, train_y, epochs=30, verbose=0)
       members.append(m)

   preds   = np.array([m.predict(test_x) for m in members])  # (3, B, H, O)
   mean_pred = preds.mean(axis=0)
   std_pred  = preds.std(axis=0)    # epistemic uncertainty estimate

----

Physics-Guided Networks
------------------------

Use the Keras ``GradientTape`` custom training loop to combine a data loss
with a physics constraint.  The spec-based build gives you an easily
serialisable configuration:

.. code-block:: python

   import keras
   import numpy as np
   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   spec = BaseAttentiveSpec(
       static_input_dim=5,
       dynamic_input_dim=6,
       future_input_dim=4,
       output_dim=1,
       forecast_horizon=48,
       embed_dim=64,
       attention_heads=8,
       backend_name="tensorflow",
       head_type="point",
   )

   class PhysicsGuidedForecaster(keras.Model):
       def __init__(self, spec, physics_weight=0.1):
           super().__init__()
           self.kernel         = BaseAttentiveV2Assembly().build(spec)
           self.physics_weight = physics_weight

       def call(self, inputs, training=False):
           return self.kernel(inputs, training=training)

       def _physics_residual(self, inputs, preds):
           """Energy-balance penalty: prediction should not deviate from
           last-observed value by more than a physically plausible amount."""
           _, dynamic_x, _ = inputs
           last_obs = dynamic_x[:, -1:, :1]   # (B, 1, 1)
           return keras.ops.mean(keras.ops.abs(preds - last_obs))

       def train_step(self, data):
           x, y = data
           with keras.GradientTape() as tape:
               preds       = self(x, training=True)
               data_loss   = keras.losses.mean_squared_error(y, preds)
               phys_loss   = self._physics_residual(x, preds)
               total_loss  = (
                   keras.ops.mean(data_loss)
                   + self.physics_weight * phys_loss
               )
           grads = tape.gradient(total_loss, self.trainable_variables)
           self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
           return {"loss": total_loss, "physics_loss": phys_loss}

   model = PhysicsGuidedForecaster(spec, physics_weight=0.05)
   model.compile(optimizer=keras.optimizers.Adam(1e-3))
   model.fit([x_static, x_dynamic, x_future], y, epochs=50)

----

Transfer Learning
-----------------

Pre-train on a large multi-site dataset, then fine-tune on a target site
with limited history.  The spec makes it straightforward to reproduce the
pre-training architecture exactly:

.. code-block:: python

   import keras
   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   pretrain_spec = BaseAttentiveSpec(
       static_input_dim=5,
       dynamic_input_dim=6,
       future_input_dim=4,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       attention_heads=8,
       dropout_rate=0.1,
       backend_name="tensorflow",
       head_type="point",
   )

   # Step 1 — pre-train on large dataset
   pretrained = BaseAttentiveV2Assembly().build(pretrain_spec)
   pretrained.compile(optimizer="adam", loss="mse")
   pretrained.fit(large_x, large_y, epochs=50, verbose=0)

   # Step 2 — clone weights into a fresh model instance
   transfer = BaseAttentiveV2Assembly().build(pretrain_spec)
   transfer.set_weights(pretrained.get_weights())

   # Step 3 — freeze early layers; only decoder and head stay trainable
   for layer in transfer.layers[:-8]:
       layer.trainable = False

   transfer.compile(
       optimizer=keras.optimizers.Adam(learning_rate=1e-5),
       loss="mse",
   )
   transfer.fit(target_x, target_y, epochs=20)

   # Step 4 (optional) — progressive unfreezing
   for layer in transfer.layers[-8:-4]:
       layer.trainable = True
   transfer.compile(
       optimizer=keras.optimizers.Adam(learning_rate=5e-6),
       loss="mse",
   )
   transfer.fit(target_x, target_y, epochs=10)

**When transfer learning helps:** new monitoring station with sparse
history, new building type without historical load, new language or region
where only a small labelled set is available.

----

Multi-Task Learning
--------------------

Share a single BaseAttentive kernel and attach multiple task heads.  The
spec-based kernel is built once; tasks add their own decoders on top:

.. code-block:: python

   import keras
   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   kernel_spec = BaseAttentiveSpec(
       static_input_dim=5,
       dynamic_input_dim=8,
       future_input_dim=4,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       attention_heads=8,
       backend_name="tensorflow",
       head_type="point",
   )

   class MultiTaskEnergyModel(keras.Model):
       def __init__(self):
           super().__init__()
           self.kernel       = BaseAttentiveV2Assembly().build(kernel_spec)
           self.demand_head  = keras.layers.Dense(1, name="demand")
           self.anomaly_head = keras.layers.Dense(
               1, activation="sigmoid", name="anomaly"
           )

       def call(self, inputs, training=False):
           shared   = self.kernel(inputs, training=training)
           # shared: (B, H, output_dim) — the per-step forecast
           context  = keras.ops.mean(shared, axis=1)   # (B, output_dim)
           return {
               "demand":  self.demand_head(shared),     # (B, H, 1)
               "anomaly": self.anomaly_head(context),   # (B, 1)
           }

   mt_model = MultiTaskEnergyModel()
   mt_model.compile(
       optimizer="adam",
       loss={"demand": "mse", "anomaly": "binary_crossentropy"},
       loss_weights={"demand": 2.0, "anomaly": 0.5},
   )
   mt_model.fit(
       [x_static, x_dynamic, x_future],
       {"demand": y_demand, "anomaly": y_anomaly},
       epochs=50,
   )

----

Domain-Specific Applications
==============================

Geophysical Hazard Forecasting
-------------------------------

BaseAttentive serves as the temporal forecasting kernel in
physics-informed geohazard systems.  The custom-encoder pattern from the
registry system makes it easy to embed domain knowledge:

**Earthquake hazard:**

.. code-block:: python

   from base_attentive import BaseAttentive

   seismic_hazard_model = BaseAttentive(
       static_input_dim=5,    # lat, lon, depth, fault_type, vs30
       dynamic_input_dim=4,   # mag_history, b_value, inter_event_time, stress_idx
       future_input_dim=2,    # coulomb_stress_change, season_forcing
       output_dim=1,          # exceedance probability
       forecast_horizon=12,   # 12-month hazard window
       mode="pihal",
       attention_levels=["cross", "memory"],
       memory_size=128,       # recall past seismic sequences
       embed_dim=64,
       num_heads=8,
   )

**Landslide risk:**

.. code-block:: python

   landslide_model = BaseAttentive(
       static_input_dim=6,    # slope, soil_type, vegetation, aspect, geology, lithology
       dynamic_input_dim=4,   # rainfall, groundwater, pore_pressure, displacement
       future_input_dim=2,    # rainfall_forecast, snowmelt
       output_dim=1,          # landslide probability
       forecast_horizon=7,
       mode="tft",
       scales=[1, 3, 7],      # daily / 3-day / weekly
       quantiles=[0.5, 0.8, 0.95],
   )

----

Financial Time Series
---------------------

.. code-block:: python

   from base_attentive import BaseAttentive
   from base_attentive.components import CRPSLoss

   financial_model = BaseAttentive(
       static_input_dim=4,    # sector, market_cap_log, beta, country
       dynamic_input_dim=8,   # returns, volume, volatility, RSI, MACD, etc.
       future_input_dim=3,    # macro_event_flag, earnings_flag, expiry_flag
       output_dim=1,          # return forecast
       forecast_horizon=5,    # 5-day ahead
       objective="transformer",       # short sequences → transformer
       num_encoder_layers=4,
       embed_dim=64,
       num_heads=8,
       dropout_rate=0.2,      # higher regularisation for noisy financial data
       quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
   )

   financial_model.compile(
       optimizer="adam",
       loss=CRPSLoss(mode="quantile",
                     quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]),
   )

----

Healthcare and Epidemiology
----------------------------

.. code-block:: python

   # ICU vital-sign forecasting
   icu_model = BaseAttentive(
       static_input_dim=6,    # age, sex, admission_type, comorbidities (×3)
       dynamic_input_dim=8,   # HR, SpO2, BP_sys, BP_dia, RR, Temp, FiO2, GCS
       future_input_dim=3,    # scheduled_meds, procedure_flag, shift_change
       output_dim=4,          # HR, SpO2, BP, RR forecast
       forecast_horizon=6,    # next 6 hours
       mode="tft",
       embed_dim=32,
       num_heads=4,
       quantiles=[0.1, 0.5, 0.9],
   )

   # Disease outbreak forecasting
   outbreak_model = BaseAttentive(
       static_input_dim=5,    # region, pop_density, healthcare_capacity, age_structure, climate
       dynamic_input_dim=5,   # cases, tests, positivity, mobility, interventions
       future_input_dim=4,    # mobility_plan, intervention_plan, season_sin, season_cos
       output_dim=2,          # cases, hospitalisations
       forecast_horizon=28,   # 4-week window
       mode="pihal",
       attention_levels=["cross", "hierarchical"],
       scales=[1, 7],
       quantiles=[0.1, 0.5, 0.9],
   )

----

Integration Patterns and Deployment
=====================================

Feature Engineering Guide
--------------------------

**Static features:**

- Normalise to comparable scales (standard scaler or min-max)
- Encode categorical variables (one-hot or learned embeddings)
- Keep cardinality manageable (4–12 features is a practical range)

**Dynamic past features:**

- Include raw measurements plus derived features:

  - Lags: ``t-1``, ``t-7``, ``t-24`` (depending on granularity)
  - Rate of change: ``x[t] - x[t-1]``
  - Rolling statistics: mean and standard deviation over a window
  - Cyclical encodings: ``sin(2π·h/24)``, ``cos(2π·h/24)`` for hour-of-day

- A practical starting range is 5–15 channels

**Known future features:**

- Use deterministic inputs only: calendar, seasonal, planned events
- Incorporate NWP or economic-model forecasts when available
- Represent uncertainty via multiple scenarios fed as separate model runs

Hyperparameter Guide
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Dataset size
     - ``embed_dim``
     - ``num_heads``
     - Notes
   * - Small (< 10 K)
     - 32
     - 4
     - Use dropout ≥ 0.2; consider ``pool.last`` pooling
   * - Medium (10 K–100 K)
     - 64
     - 8
     - Standard config; tune ``scales`` and ``attention_levels``
   * - Large (> 100 K)
     - 128
     - 16
     - Reduce dropout; consider transformer encoder

Start with ``mode="tft"`` for most applications.  Switch to
``mode="pihal"`` when long-range memory is needed (memory_size > 50)
or when the data has strong nested temporal structure.

Production Deployment Checklist
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concern
     - Approach
   * - Model versioning
     - Store the ``BaseAttentiveSpec`` JSON alongside each model checkpoint
   * - Input validation
     - Use ``validate_model_inputs`` at every inference entry point
   * - Distribution monitoring
     - Track rolling statistics on input features; alert on shift
   * - Retraining cadence
     - Monthly on new data; triggered retraining on performance degradation
   * - Uncertainty output
     - Always include quantile output (or CRPS loss) in production models
   * - Latency
     - Use ``make_fast_predict_fn`` (TF) or torch.compile (Torch) for hot paths
   * - Fallback
     - Keep a physics-based or persistence baseline as fallback
   * - Backend choice
     - TF for Serving / TFLite; Torch for CUDA GPU / MPS; JAX for TPU

----

Evaluation Metrics
-------------------

For regression (point forecast):

- **MAE** — mean absolute error; easy to interpret in original units
- **RMSE** — root mean squared error; sensitive to outliers
- **MAPE** — mean absolute percentage error; relative view

For probabilistic forecast:

- **CRPS** — proper scoring rule; rewards both calibration and sharpness
- **Coverage** — fraction of true values inside the predicted interval
- **Interval width** — narrower is better, given adequate coverage
- **Winkler score** — combined width + coverage penalty

For anomaly / classification outputs:

- **AUC-ROC** — threshold-independent performance
- **F1-score** — balance precision and recall at the operating threshold

----

See Also
--------

- :doc:`usage` — V2 configuration in depth
- :doc:`architecture_guide` — Registry / Assembly internals
- :doc:`configuration_guide` — Full parameter reference
- :doc:`backends/index` — Backend selection
- :doc:`api_reference` — Complete API docs
- `GitHub examples <https://github.com/earthai-tech/base-attentive/tree/master/examples>`_
  — Jupyter notebooks for each application
