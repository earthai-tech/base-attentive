# BaseAttentive API Documentation

**Last Updated:** April 10, 2026

## Table of Contents

- [Overview](#overview)
- [BaseAttentive Class](#baseattentive-class)
- [Core Methods](#core-methods)
- [Parameters Reference](#parameters-reference)
- [Configuration Management](#configuration-management)
- [Input/Output Specifications](#inputoutput-specifications)
- [Backend Support](#backend-support)

---

## Overview

The **BaseAttentive** class is the primary interface for creating advanced sequence-to-sequence models with attention mechanisms. It serves as the foundational blueprint for time series forecasting models that process three distinct input types:

1. **Static Features** - unchanging across time (e.g., site properties)
2. **Dynamic Past Features** - historical time series data
3. **Known Future Features** - forecast-period exogenous variables

### Package Information

```python
Package:     base-attentive
Import name: base_attentive
Version:     1.0.0
License:     Apache-2.0
Author:      LKouadio
```

---

## BaseAttentive Class

### Class Definition

```python
from base_attentive import BaseAttentive

class BaseAttentive(Model, NNLearner):
    """
    Advanced encoder-decoder architecture with configurable attention mechanisms.
    
    Inherits from:
    - keras.Model: Keras neural network model
    - NNLearner: Scikit-learn compatible estimator interface
    """
```

### Quick Start Example

```python
import tensorflow as tf
from base_attentive import BaseAttentive

# Create a model
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    embed_dim=32,
    num_heads=8,
    dropout_rate=0.15
)

# Model is ready to compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

## Core Methods

### `__init__(**config)`

Initializes a BaseAttentive model with the specified configuration.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `static_input_dim` | int | Required | Number of static features |
| `dynamic_input_dim` | int | Required | Number of dynamic features |
| `future_input_dim` | int | Required | Number of future exogenous features |
| `output_dim` | int | Required | Number of output variables |
| `forecast_horizon` | int | Required | Forecast length in time steps |
| `architecture_config` | dict | See defaults | Architecture-level configuration |
| `embed_dim` | int | 32 | Embedding dimension |
| `hidden_units` | int | 64 | LSTM/Dense hidden units |
| `lstm_units` | int | 64 | LSTM layer units |
| `attention_units` | int | 64 | Attention mechanism units |
| `num_heads` | int | 4 | Multi-head attention heads |
| `dropout_rate` | float | 0.2 | Dropout probability |
| `activation` | str | 'relu' | Activation function |
| `use_residuals` | bool | True | Enable residual connections |
| `use_vsn` | bool | True | Use Variable Selection Network |
| `use_batch_norm` | bool | False | Use batch normalization |
| `quantiles` | list[float] | None | Quantiles for uncertainty (e.g., [0.1, 0.5, 0.9]) |

#### Configuration Dictionary

```python
architecture_config = {
    "encoder_type": "hybrid",  # or "transformer"
    "decoder_attention_stack": ["cross", "hierarchical", "memory"],
    "feature_processing": "vsn",  # or "dense"
}
```

---

### `get_config() -> dict`

Returns a dictionary representation of the model configuration for serialization.

```python
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24
)

config = model.get_config()
# Returns: {
#     'static_input_dim': 4,
#     'dynamic_input_dim': 8,
#     'future_input_dim': 6,
#     'output_dim': 2,
#     'forecast_horizon': 24,
#     'embed_dim': 32,
#     ...
# }
```

---

### `from_config(config) -> BaseAttentive` (ClassMethod)

Creates a BaseAttentive instance from a configuration dictionary.

```python
# Load model from saved config
config = {
    'static_input_dim': 4,
    'dynamic_input_dim': 8,
    'future_input_dim': 6,
    'output_dim': 2,
    'forecast_horizon': 24,
    'embed_dim': 32
}

model = BaseAttentive.from_config(config)
```

---

### `call(inputs, training=False) -> Tensor`

Executes the model forward pass on input tensors.

#### Parameters

- `inputs` (tuple of 3 tensors): 
  - `(static_features, dynamic_features, future_features)`
- `training` (bool): Whether in training mode

#### Returns

- **Point Forecasts**: `Tensor` of shape `(batch_size, forecast_horizon, output_dim)`
- **With Quantiles**: `Tensor` of shape `(batch_size, forecast_horizon, num_quantiles, output_dim)`

#### Example

```python
import numpy as np

# Prepare inputs
batch_size = 32
time_steps = 100

static = np.random.randn(batch_size, 4)           # (32, 4)
dynamic = np.random.randn(batch_size, time_steps, 8)   # (32, 100, 8)
future = np.random.randn(batch_size, 24, 6)       # (32, 24, 6)

# Forward pass
predictions = model([static, dynamic, future], training=False)
# predictions.shape → (32, 24, 2)
```

---

### `summary() -> None`

Prints a summary of the model architecture and parameters.

```python
model.summary()
# Output:
# Model: "BaseAttentive"
# __________________________________
# Layer (type)          Output Shape        Param #
# ==================================
# ...
```

---

### `reconfigure(**new_config) -> BaseAttentive`

Creates a new model instance with updated configuration while preserving the original.

```python
# Original model
model1 = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    embed_dim=32
)

# Create variant with different embed_dim
model2 = model1.reconfigure(embed_dim=64)

# model1 unchanged, model2 has new config
```

---

## Parameters Reference

### Required Parameters

| Parameter | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `static_input_dim` | int | ≥ 0 | Number of static input features |
| `dynamic_input_dim` | int | ≥ 0 | Number of dynamic input features |
| `future_input_dim` | int | ≥ 0 | Number of known future features |
| `output_dim` | int | ≥ 1 | Number of target output variables |
| `forecast_horizon` | int | ≥ 1 | Forecast horizon in time steps |

### Architectural Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `embed_dim` | int | 32 | [8, 512] | Embedding/representation dimension |
| `hidden_units` | int | 64 | [16, 1024] | Dense layer hidden units |
| `lstm_units` | int | 64 | [16, 1024] | LSTM cell units |
| `attention_units` | int | 64 | [16, 1024] | Attention mechanism dimension |
| `num_heads` | int | 4 | [1, 16] | Multi-head attention heads |

### Regularization Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `dropout_rate` | float | 0.2 | [0.0, 1.0] | Dropout probability |
| `activation` | str | 'relu' | See below | Activation function |
| `use_batch_norm` | bool | False | - | Apply batch normalization |
| `use_residuals` | bool | True | - | Use residual connections |

### Feature Processing Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `use_vsn` | bool | True | - | Use Variable Selection Network |
| `feature_processing` | str | 'vsn' | 'vsn', 'dense' | Feature processing method |

### Optional Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantiles` | list[float] | None | Quantiles for uncertainty (e.g., [0.1, 0.5, 0.9]) |

### Supported Activation Functions

```python
# Common activations
activation in ['relu', 'elu', 'selu', 'sigmoid', 'tanh', 'linear']
```

---

## Configuration Management

### Configuration Hierarchy

Configurations are applied in order of priority:

```
1. architecture_config dict (highest priority)
2. Explicit keyword arguments
3. Default values (lowest priority)
```

### Architecture Configuration

```python
# Default architecture
DEFAULT_ARCHITECTURE = {
    "encoder_type": "hybrid",
    "decoder_attention_stack": ["cross", "hierarchical", "memory"],
    "feature_processing": "vsn"
}

# Transformer-based alternative
TRANSFORMER_ARCHITECTURE = {
    "encoder_type": "transformer",
    "decoder_attention_stack": ["cross", "attention"],
    "feature_processing": "vsn"
}

# Use custom architecture
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    architecture_config=TRANSFORMER_ARCHITECTURE
)
```

### Configuration Merging Example

```python
# Base defaults
defaults = {
    'embed_dim': 32,
    'num_heads': 4,
    'dropout_rate': 0.2
}

# User configuration
user_config = {
    'static_input_dim': 4,
    'dynamic_input_dim': 8,
    'future_input_dim': 6,
    'output_dim': 2,
    'forecast_horizon': 24,
    'embed_dim': 64,  # Overrides default
    'architecture_config': {
        'encoder_type': 'transformer'
    }
}

# Final merged config
# → embed_dim: 64 (from user_config)
# → num_heads: 4 (from defaults)
# → encoder_type: 'transformer' (from architecture_config)
```

---

## Input/Output Specifications

### Input Tensor Shapes

The model expects a tuple of 3 tensors during the forward pass:

```python
inputs = (static_features, dynamic_features, future_features)

# Where:
# static_features:   (batch_size, static_dim)
# dynamic_features:  (batch_size, time_steps, dynamic_dim)
# future_features:   (batch_size, forecast_horizon, future_dim)
```

### Output Tensor Shapes

#### Without Quantiles

```python
# output_shape = (batch_size, forecast_horizon, output_dim)
predictions = model(inputs)  # shape: (B, H, D)
```

#### With Quantiles

```python
# output_shape = (batch_size, forecast_horizon, num_quantiles, output_dim)
predictions = model(inputs)  # shape: (B, H, Q, D)
```

### Valid Input Ranges

| Dimension | Min | Max | Notes |
|-----------|-----|-----|-------|
| batch_size | 1 | ∞ | Must be > 0 |
| time_steps | 1 | ∞ | History length for dynamic features |
| forecast_horizon | 1 | ∞ | Must match model config |
| *_dim | 0 | ∞ | Can be 0 (no features of this type) |

### Data Type Requirements

- **Input tensors**: `float32` or `float64`
- **Output tensors**: Same as inputs (typically `float32`)

---

## Backend Support

### Supported ML Frameworks

BaseAttentive supports multiple ML backends through Keras 3:

| Backend | Status | Requirements |
|---------|--------|--------------|
| TensorFlow | ✅ Primary | `tensorflow>=2.12.0` |
| JAX | 🔶 Experimental | `jax>=0.4.0` |
| PyTorch | 🔶 Experimental | `torch>=2.0.0` |

### Setting the Backend

```python
# Method 1: Environment Variable
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Method 2: Environment Variable (package-specific)
os.environ['BASE_ATTENTIVE_BACKEND'] = 'tensorflow'

# Method 3: Programmatic
from base_attentive import backend
backend.set_backend('tensorflow')
```

### Backend Capabilities

```python
from base_attentive import backend

# Get current backend
current = backend.get_backend()  # Returns: 'tensorflow'

# Get backend capabilities
caps = backend.get_backend_capabilities('tensorflow')
# Returns: {
#     'name': 'tensorflow',
#     'version': '2.12.0',
#     'supports_mixed_precision': True,
#     'supports_xla': True
# }

# Query available backends
available = backend.get_available_backends()
# Returns: ['tensorflow', 'jax', 'torch'] (or subset)
```

---

## Error Handling

### Common Errors and Solutions

#### Input Shape Mismatch

```python
# ❌ Error
dynamic = np.random.randn(32, 100)  # Missing feature dimension!
# → ValueError: Expected 3D tensor for dynamic features

# ✅ Correct
dynamic = np.random.randn(32, 100, 8)  # (batch, time, features)
```

#### Dimension Mismatch

```python
# ❌ Error
static = np.random.randn(32, 5)  # Expected 4, got 5
# → ValueError: static_input_dim mismatch

# ✅ Correct
static = np.random.randn(32, 4)  # Matches static_input_dim=4
```

#### Invalid Configuration

```python
# ❌ Error
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    embed_dim=-1  # Invalid!
)
# → ValueError: embed_dim must be > 0

# ✅ Correct
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    embed_dim=32
)
```

---

## Advanced Usage

### Creating Uncertainty Estimates with Quantiles

```python
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    quantiles=[0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
)

predictions = model([static, dynamic, future])
# shape: (batch_size, 24, 3, 2)
#         (batch,     horizon, quantiles, output_dim)

# Extract point forecast (median)
point_forecast = predictions[:, :, 1, :]  # 50th percentile

# Extract confidence intervals
lower = predictions[:, :, 0, :]  # 10th percentile
upper = predictions[:, :, 2, :]  # 90th percentile
```

### Hybrid vs Transformer Modes

```python
# Hybrid mode: Multi-scale LSTM + Attention
hybrid_model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    architecture_config={
        'encoder_type': 'hybrid'
    }
)

# Transformer mode: Pure attention
transformer_model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    architecture_config={
        'encoder_type': 'transformer'
    }
)
```

### Scikit-Learn Integration

```python
from sklearn.pipeline import Pipeline
from base_attentive import BaseAttentive

# Use model with sklearn pipeline
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24
)

# Get/set parameters (sklearn compatible)
params = model.get_params()
model.set_params(embed_dim=64)
```

---

## See Also

- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md)
- [Components Reference](COMPONENTS_REFERENCE.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
- [Quick Start Examples](examples/)
