Applications and Use Cases
===========================

BaseAttentive can be used as a standalone forecasting model or as a kernel
within larger deep learning architectures. This page outlines example
applications, integration patterns, and deployment considerations.

.. contents::
   :local:
   :depth: 2

Standalone Forecasting Applications
====================================

BaseAttentive can be applied to multi-step time series forecasting across
domains. It combines static context features with dynamic historical patterns
and known future information to produce multi-step forecasts.

Architecture Pattern
--------------------

All standalone applications follow this unified architecture:

.. code-block:: text

    Static Features (location, properties, type)
           ↓
    Dynamic Past (historical measurements)
           ↓  ← BaseAttentive hybrid/transformer attention
    Known Future (forecasted external conditions)
           ↓
    Multi-Step Predictions (uncertainty quantiles)

.. list-table:: Standalone Application Categories
   :widths: 30 40 30
   :header-rows: 1

   * - Application Domain
     - Input Features
     - Prediction Targets
   * - Air Quality
     - Location, altitude, urban/rural + Historical pollution + Weather forecast
     - PM2.5, NO₂, O₃ levels (24h)
   * - Energy Demand
     - Building type, area, units + Historical load + Weather, calendar
     - Electricity demand (48h), peak times
   * - Weather
     - Geography, elevation, terrain + Historical weather + Seasonal data
     - Temperature, pressure, precipitation (30+ steps)
   * - Traffic Flow
     - Road properties, lanes + Historical volume, speed + Time, events
     - Vehicle volume, congestion level (48 steps)

Air Quality Forecasting
-----------------------

**Challenge:** Air pollution varies with meteorology, human activity, and geography. Cities need real-time forecasts for health alerts.

**Example setup:**

- **Static Features:** Geographic coordinates, elevation, urban index, pollution source density
- **Dynamic Past:** 7 days of hourly PM2.5, NO₂, O₃, temperature, humidity
- **Known Future:** Weather forecast (wind speed, temperature, precipitation)
- **Output:** 24-hour PM2.5 forecast with uncertainty quantiles

**What this setup can capture:**

- Can represent recurring pollution patterns such as rush-hour peaks or seasonal changes
- Can incorporate weather forecasts and scenario-based inputs
- Allows joint prediction of several pollutants
- Quantile outputs can support threshold-based alerting strategies

**Use Cases:**

- Public health alerts (air quality index warnings)
- School/outdoor event planning
- Industrial emission controls
- Vulnerable population notifications

**Example Workflow:**

.. code-block:: python

    from base_attentive import BaseAttentive
    
    air_quality_model = BaseAttentive(
        static_dim=4,           # lat, lon, alt, urban_index
        dynamic_dim=5,          # PM2.5, NO2, O3, Temp, Humidity
        future_dim=2,           # wind_speed, temp_forecast
        forecast_horizon=24,
        n_quantiles=3,          # [10th, 50th, 90th percentiles]
        mode='hybrid'
    )
    
    model.fit([static_features, historical_data, weather_forecast], targets)
    predictions = model.predict([test_static, test_history, test_future])
    # predictions shape: (batch, 24, 3) → 24-hour forecast with 3 quantiles


Energy Demand Forecasting
--------------------------

**Challenge:** Electric grids must balance supply and demand in real-time. Peak demand prediction enables optimal resource allocation.

**Example setup:**

- **Static Features:** Building type, floor area, insulation level, HVAC capacity, solar installation
- **Dynamic Past:** 2 weeks of hourly loads, temperature, solar irradiance, time-of-use signals
- **Known Future:** Weather forecast, calendar data (weekday/weekend/holiday), planned maintenance
- **Output:** 48-hour demand forecast with peak identification

**What this setup can capture:**

- Can represent building-specific consumption patterns
- Can account for daily and weekly seasonality
- Can relate demand to heating and cooling conditions
- Can be coupled with demand-response rules
- Produces multi-step forecasts for planning and dispatch

**Use Cases:**

- Real-time grid balancing and frequency regulation
- Renewable energy integration (solar/wind variability)
- Peak shaving and demand response
- Cost minimization with time-of-use pricing
- Microgrids and smart buildings

**Integration Example:**

.. code-block:: python

    # Train on building portfolio
    energy_models = {}
    for building_id, data in buildings.items():
        model = BaseAttentive(
            static_dim=5,        # Property features
            dynamic_dim=5,       # Load, temp, solar, hour_sin, hour_cos
            future_dim=2,        # Weather, day_type
            forecast_horizon=48, # 2-day forecast
            mode='hybrid'
        )
        model.fit(data['train_x'], data['train_y'])
        energy_models[building_id] = model
    
    # Real-time forecasting
    for building_id, model in energy_models.items():
        demand_forecast = model.predict(current_features)
        activate_demand_response_if(demand_forecast > threshold)


Weather Prediction
------------------

**Challenge:** Weather systems are chaotic. Even small prediction windows require capturing multi-scale atmospheric interactions.

**Example setup:**

- **Static Features:** Geographic coordinates, elevation, terrain type, urban heat island index
- **Dynamic Past:** 5 days of 2-hourly weather (temperature, pressure, humidity, wind components)
- **Known Future:** Seasonal encoding, jet stream position indicators
- **Output:** 30-hour deterministic forecast (2-hourly step)

**What this setup can capture:**

- Can combine geographic context with recent atmospheric history
- Memory attention can retain longer structures in the sequence
- Can include seasonal or climatological covariates
- Can be used within ensemble workflows for uncertainty analysis
- Neural surrogates may reduce inference cost relative to physics-based NWP, although forecast quality depends on data coverage and evaluation design

**Use Cases:**

- Weather service operations
- Agricultural planning (frost warnings, irrigation)
- Renewable energy forecasting (solar, wind)
- Disaster management (extreme weather detection)
- Traffic and transportation

**Implementation:**

.. code-block:: python

    weather_model = BaseAttentive(
        static_dim=4,           # lat, lon, elevation, terrain
        dynamic_dim=5,          # T, P, RH, wind_u, wind_v
        forecast_horizon=30,    # 60 hours of 2-hourly data
        lookback_window=120,    # 10 days
        mode='transformer',     # Pure attention for weather
        attention_stack=[
            ('cross_attention', {'heads': 8}),
            ('self_attention', {'heads': 8}),
            ('memory_attention', {'heads': 4, 'memory_size': 32}),
        ]
    )


Traffic Flow Prediction
-----------------------

**Challenge:** Traffic patterns have complex dependencies on time-of-day, incidents, events, and weather.

**Example setup:**

- **Static Features:** Road segment properties (type, lanes, speed limit, urban/highway)
- **Dynamic Past:** 24 hours of 5-minute traffic (volume, speed, occupancy, incident flags)
- **Known Future:** Time-of-day, day-of-week, known events, weather forecast
- **Output:** 4-hour traffic predictions (5-minute resolution)

**What this setup can capture:**

- Can represent road-specific patterns and rush-hour dynamics
- Can incorporate incident indicators and propagation effects
- Can include scheduled events such as concerts or road work
- Can account for weather-related changes in traffic flow
- Can support downstream routing or congestion-management systems

**Use Cases:**

- Real-time navigation and route optimization
- Congestion pricing and demand management
- Traffic signal control optimization
- Public transit prioritization
- Emergency vehicle routing

BaseAttentive as a Kernel in Larger Models
==========================================

Beyond standalone forecasting, BaseAttentive can also be used as a kernel
component within larger neural networks and decision pipelines.

Ensemble Methods
----------------

**Pattern:** Combine multiple BaseAttentive architectures with different attention mechanisms.

.. code-block:: text

    Ensemble Architecture:
    
    Inputs
      ↓
    BaseAttentive Kernel 1 (hybrid: cross_attn + self_attn)
    BaseAttentive Kernel 2 (transformer: multi-layer self_attn)
    BaseAttentive Kernel 3 (memory: recurrent + attention)
      ↓
    Meta-learner (learnable weights or weighted average)
      ↓
    Aggregated Predictions

**Typical effects:**

- Can reduce prediction variance across model instances
- May improve calibration of predictive intervals
- Helps compare behavior under distribution shift
- Combines different modeling assumptions on difficult cases
- Adds redundancy if one member degrades

**Example Application:**

.. code-block:: python

    # Energy forecasting ensemble
    kernels = [
        BaseAttentive(..., mode='hybrid'),    # Data + temporal patterns
        BaseAttentive(..., mode='transformer'), # Long-range interactions
        BaseAttentive(..., attention_stack=[
            ('memory_attention', {...}),       # Recurrent dynamics
        ])
    ]
    
    # Combine predictions
    predictions = [k.predict(inputs) for k in kernels]
    ensemble_pred = np.mean(predictions, axis=0)      # Average
    ensemble_uncertainty = np.std(predictions, axis=0) # Calibrated uncertainty


Physics-Guided Networks
------------------------

**Pattern:** Embed domain knowledge and conservation laws into the neural network training process.

**Approach:**

1. BaseAttentive learns data-driven patterns from observations
2. Physics layer enforces conservation laws (e.g., energy balance)
3. Hybrid loss combines data fit and physics constraints

.. code-block:: text

    Physics Loss = λ₁ × Data Loss + λ₂ × Constraint Loss
    
    Example (Energy Conservation):
      Energy_t+1 ≈ Energy_t × decay + Solar_Production
      Constraint Loss = MSE(prediction, physics_prediction)

**Application Examples:**

- **Energy Systems:** Enforce Kirchhoff's laws, energy balance
- **Fluid Dynamics:** Incorporate momentum, continuity equations
- **Climate:** Include mass, heat conservation
- **Geophysics:** Respect seismic wave physics

**Typical effects:**

- Encourages physically plausible outputs
- Can improve extrapolation when the constraints are informative
- May reduce data requirements through regularization
- Constraint violations can help identify modeling gaps
- Often useful for longer forecast horizons

**Implementation:**

.. code-block:: python

    class PhysicsGuidedEnsemble(tf.keras.Model):
        def __init__(self, ensemble, physics_weight=0.1):
            super().__init__()
            self.ensemble = ensemble
            self.physics_weight = physics_weight
        
        def train_step(self, data):
            x, y = data
            
            with tf.GradientTape() as tape:
                pred = self.ensemble(x, training=True)
                
                # Data loss
                data_loss = tf.keras.losses.mse(y, pred)
                
                # Physics loss (energy balance example)
                physics_loss = self.compute_physics_loss(x, pred)
                
                # Combined
                total_loss = data_loss + self.physics_weight * physics_loss
            
            # Backpropagation
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            return {'loss': total_loss, 'physics_loss': physics_loss}


Transfer Learning for Data-Limited Domains
-------------------------------------------

**Pattern:** Pre-train on large multi-site dataset, fine-tune on specific location.

**Workflow:**

1. **Pre-training phase:** Train BaseAttentive on 50+ locations
   - Learns general time series patterns
   - Captures domain-specific behaviors (diurnal cycles, seasonality)
   - Produces reusable feature representations

2. **Fine-tuning phase:** Adapt to target location with limited data
   - Freeze early attention layers (general patterns)
   - Train later layers and decoder (location-specific)
   - Can reduce the amount of target-site data needed when source and target domains are related

**Typical effects:**

- Can help when target-site history is limited
- Reuses information learned during pre-training
- May reduce overfitting on small target datasets
- Fine-tuning remains important when domains differ substantially

**Application Examples:**

- **New building:** No historical energy data, but buildings of similar type available
- **New weather station:** Transfer from nearby locations
- **Emerging pollutant:** Transfer from similar chemical compounds
- **New city:** Transfer from cities with similar geography and climate

**Implementation:**

.. code-block:: python

    # Pre-train on 100+ buildings
    pretrained = BaseAttentive(...)
    pretrained.fit(large_dataset, epochs=50)
    
    # Clone and freeze encoder
    transfer_model = clone_model(pretrained)
    for layer in transfer_model.layers[:-8]:
        layer.trainable = False
    
    # Fine-tune on target building
    transfer_model.compile(optimizer=Adam(lr=1e-5), loss='mse')
    transfer_model.fit(target_data, epochs=20)
    
    # Optional: progressive unfreezing
    for layer in transfer_model.layers[-8:-4]:
        layer.trainable = True
    transfer_model.compile(optimizer=Adam(lr=1e-5), loss='mse')
    transfer_model.fit(target_data, epochs=10)
    

Multi-Task Learning
--------------------

**Pattern:** Predict multiple correlated quantities jointly using shared BaseAttentive representations.

**Example:** Energy System Prediction

.. code-block:: text

    Shared BaseAttentive Kernel
      ↓
    Task 1: Energy Demand ← Dense Decoder
    Task 2: Peak Hour ← Dense Decoder + Softmax
    Task 3: Anomaly Detection ← Dense Decoder + Sigmoid
    Task 4: Grid Frequency ← Dense Decoder

**Typical effects:**

- Shared representations can reduce overfitting
- Related tasks can regularize one another
- A single forward pass can produce multiple outputs
- Joint training may help when tasks share structure
- Inference can remain compact in deployment

**Loss Combination:**

.. code-block:: python

    total_loss = (
        w1 * MSE(demand_pred, demand_true) +
        w2 * CrossEntropy(peak_pred, peak_true) +
        w3 * CrossEntropy(anomaly_pred, anomaly_true) +
        w4 * MSE(frequency_pred, frequency_true)
    )

**Typical Weights:**

- Primary prediction task: 2.0-3.0 weight
- Supporting tasks: 0.5-1.0 weight
- Adjust based on task importance and data availability

Domain-Specific Applications
=============================

Geophysical Hazard Forecasting (GeoPhysics & GeoPrior Inspired)
---------------------------------------------------------------

Applying BaseAttentive to physics-guided geohazard prediction:

**Applications:**

1. **Earthquake Hazard Assessment**
   - Static: Location, fault type, geological properties
   - Dynamic: Historical seismicity, stress indicators
   - Future: Forecasted stress changes
   - Output: Seismic hazard probability

2. **Landslide Risk Prediction**
   - Static: Slope angle, soil properties, vegetation
   - Dynamic: Historical rainfall, groundwater level
   - Future: Weather forecast
   - Output: Landslide probability and timing

3. **Volcanic Eruption Forecasting**
   - Static: Volcano characteristics, past eruption patterns
   - Dynamic: Seismic activity, gas emissions, deformation
   - Future: Seasonal forcing
   - Output: Eruption probability, time window

**Physics-Guided Constraints:**

- Coulomb stress changes (earthquakes)
- Infinite slope stability criterion (landslides)
- Magma rheology and pressure buildup (volcanoes)

**Integration with GeoPrior:**

BaseAttentive can serve as the temporal prediction kernel in GeoPrior-style systems:

.. code-block:: text

    Spatial Prior (Gaussian Process / Neural Field)
              ↓
    Combine with Temporal Model (BaseAttentive)
              ↓
    Joint Space-Time Hazard Map
              ↓
    Risk Assessment → Early Warning

Financial Time Series Forecasting
---------------------------------

**Application:** Stock price, volatility, and risk prediction

**BaseAttentive Configuration:**

- **Static:** Sector, company fundamentals, market cap category
- **Dynamic:** 1-year price history, volumes, technical indicators
- **Future:** Economic calendar events, policy releases
- **Output:** Return distribution, volatility, value-at-risk

**Key Considerations:**

- Handle extreme events and regime changes
- Incorporate market microstructure (bid-ask spreads)
- Account for correlation breakdowns during crises
- Ensemble methods for risk management

Healthcare and Epidemiology
----------------------------

**Applications:**

1. **Patient Vital Sign Forecasting**
   - Predict upcoming vital sign degradation
   - Enable preventive interventions

2. **Disease Outbreak Prediction**
   - Static: Region, demographics, healthcare capacity
   - Dynamic: Historical case counts, testing rates
   - Future: Mobility patterns, interventions
   - Output: Case forecasts, hospitalization needs

3. **Seasonal Disease Incidence**
   - Respiratory infections, allergies, etc.
   - Enable resource allocation and public health messaging

Integration Patterns and Deployment Notes
=========================================

Production Deployment Checklist
-------------------------------

.. list-table:: BaseAttentive Production Readiness
   :widths: 30 20 50
   :header-rows: 1

   * - Component
     - Requirement
     - Implementation
   * - Model Versioning
     - Track all model changes
     - Git LFS + model metadata (data snapshot, performance metrics)
   * - Data Validation
     - Catch distribution drift early
     - Statistical tests on input features
   * - Performance Monitoring
     - Track prediction accuracy
     - Automated alerts on metrics degradation
   * - Retraining Pipeline
     - Keep model current
     - Monthly or quarterly retraining on recent data
   * - Uncertainty Quantification
     - Communicate prediction confidence
     - Quantile outputs or ensemble variance
   * - Latency Requirements
     - Production serving constraints
     - Model compression, batching, edge deployment if needed
   * - Fallback Mechanisms
     - Handle model failures
     - Physics-based baseline models or persistence forecasts
   * - Explainability
     - Understand predictions
     - Attention weight visualization, LIME/SHAP for individual predictions

Feature Engineering Guide
--------------------------

**Static Features:**

- Keep cardinality reasonable (1-10 features typical)
- Normalize to similar scales
- Include domain categories (e.g., building type, sector)

**Dynamic Past Features:**

- A small to moderate feature set (for example 5-15 channels) is often a practical starting range
- Include raw measurements and derived features:
  - Lags (t-1, t-7, t-365 for daily data)
  - Differences (rate of change)
  - Cyclical encodings (time-of-day via sine/cosine)
  - Aggregations (rolling means, standard deviations)

**Known Future Features:**

- Use deterministic features (calendar, seasonal, planned events)
- Incorporate forecasts (weather, economic indicators)
- Represent uncertainty via multiple scenarios

Hyperparameter Tuning Strategy
------------------------------

**Quick Start Configuration (prototyping):**

.. code-block:: python

    model = BaseAttentive(
        static_dim=your_static_dim,
        dynamic_dim=your_dynamic_dim,
        future_dim=your_future_dim,
        forecast_horizon=your_horizon,
        mode='hybrid',  # Balance attention and LSTM
        attention_stack=[
            ('cross_attention', {'heads': 4, 'dim': 64}),
            ('self_attention', {'heads': 4, 'dim': 64}),
        ]
    )

**Production Tuning:**

1. Adjust ``heads`` and ``dim`` based on data size:
   - Small data (< 10K samples): 4 heads, 64 dim
   - Medium data (10K-100K): 8 heads, 128 dim
   - Large data (> 100K): 16 heads, 256 dim

2. Stack depth (number of attention layers):
   - Start with 2-3 layers
   - Add more if overfitting is low and data abundant

3. Quantiles for uncertainty:
   - 3 quantiles ([0.1, 0.5, 0.9]): Fastest, basic uncertainty
   - 5 quantiles ([0.05, 0.25, 0.5, 0.75, 0.95]): Balanced
   - 11 quantiles: Maximum flexibility, slower inference

Evaluation Metrics Framework
----------------------------

**For Regression (most applications):**

- **MAE (Mean Absolute Error):** Business interpretability
- **RMSE (Root Mean Squared Error):** Sensitive to outliers
- **MAPE (Mean Absolute Percentage Error):** Relative performance
- **Quantile Loss:** Evaluate specific percentiles

**For Probabilistic Forecasts:**

- **Continuous Ranked Probability Score (CRPS):** Overall calibration
- **Coverage:** What fraction of true values fall in [q0.1, q0.9]?
- **Mean Width:** Are uncertainty intervals narrow (good) or wide?
- **Dawid-Sebastiani Score:** Combined sharpness and calibration

**For Classification (anomaly detection, peak prediction):**

- **AUC-ROC:** Threshold-independent performance
- **F1-score:** Balance precision and recall
- **Coverage at Risk Threshold:** How many true events caught?

References and Further Reading
=======================================

**BaseAttentive Framework:**

- See :doc:`api_reference` for complete API documentation
- See :doc:`architecture_guide` for architectural details
- See :doc:`torch_backend_guide` for GPU acceleration

**Related Deep Learning References:**

- Vaswani et al., 2017: "Attention is All You Need" (Transformers)
- Lim et al., 2021: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Salinas et al., 2020: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"

**Related Physics-Guided ML:**

- Raissi et al., 2019: "Physics-informed neural networks"
- Willard et al., 2022: "Integrating Physics-based Modeling with Machine Learning"

**Time Series Forecasting:**

- Makridakis et al., 2018: "The M4 Competition" (benchmark study)
- Hyndman & Athanasopoulos: "Forecasting: Principles and Practice" (free online textbook)

Getting Started
===============

To continue with an application-specific workflow:

1. **Explore examples:** Check the examples folder for notebooks on standalone applications and kernel-based architectures
2. **Quick start:** Follow :doc:`quick_start`
3. **Full API:** Consult :doc:`api_reference`
4. **Configuration**: Read :doc:`configuration_guide`

Need help?

- Open an issue on `GitHub <https://github.com/yourusername/base-attentive>`__
- Discuss on project forums
- Review other examples and documentation
