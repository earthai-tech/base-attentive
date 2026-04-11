# BaseAttentive

[![Tests](https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/base-attentive/badge/?version=latest)](https://base-attentive.readthedocs.io/)
[![Code Quality](https://github.com/earthai-tech/base-attentive/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/earthai-tech/base-attentive/actions/workflows/code-quality.yml)
[![Publish](https://github.com/earthai-tech/base-attentive/actions/workflows/pypi-release.yml/badge.svg?branch=main)](https://github.com/earthai-tech/base-attentive/actions/workflows/pypi-release.yml)
[![Version](https://img.shields.io/github/v/release/earthai-tech/base-attentive?display_name=tag)](https://github.com/earthai-tech/base-attentive/releases)
[![License](https://img.shields.io/github/license/earthai-tech/base-attentive)](https://github.com/earthai-tech/base-attentive/blob/main/LICENSE)
[![Coverage](https://codecov.io/gh/earthai-tech/base-attentive/branch/main/graph/badge.svg)](https://codecov.io/gh/earthai-tech/base-attentive)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-46a758.svg)](https://docs.astral.sh/ruff/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A foundational blueprint for building powerful, data-driven, sequence-to-sequence time series forecasting models with advanced attention mechanisms.

## Overview

**BaseAttentive** is a sophisticated and highly configurable encoder-decoder architecture designed to process three distinct types of inputs:
- **Static features** - constant across time (e.g., geographical coordinates, site properties)
- **Dynamic past features** - historical time series (e.g., sensor readings, observations)
- **Known future features** - forecast-period exogenous variables (e.g., weather forecasts)

It fuses these inputs using a modular stack of attention mechanisms and serves as the core engine for models like **HALNet** and **PIHALNet**.

## Key Features

**Flexible Architecture**
- Hybrid mode: Multi-scale LSTM + Attention
- Transformer mode: Pure self-attention
- Configurable attention stack (cross, hierarchical, memory-augmented)

**Advanced Components**
- Variable Selection Networks (VSN) for learnable feature selection
- Multi-scale LSTM for hierarchical temporal patterns
- Cross-attention for encoder-decoder interaction
- Memory-augmented attention for long-term dependencies
- Dynamic time warping (DTW) for time-series alignment
- Quantile distribution modeling for uncertainty quantification

**Production-Ready**
- Keras 3 backed with configurable runtimes
- Serializable (save/load models)
- Input validation and parameter checking
- Extensive logging and debugging support

## Installation

### From PyPI (coming soon)
```bash
pip install base-attentive
```

### From source
```bash
git clone https://github.com/earthai-tech/base-attentive.git
cd base-attentive
pip install -e ".[dev,tensorflow]"
```

### Backend extras
```bash
pip install ".[tensorflow]"   # Recommended today
pip install ".[jax]"          # Experimental runtime path
pip install ".[torch]"        # Experimental runtime path
```

## Development Setup

If you use `make` (Linux, macOS, WSL, or Git Bash on Windows), the repository
now includes a `Makefile` with common development commands:

```bash
make install-tensorflow   # editable install with dev + TensorFlow extras
make test-fast            # quick local pytest pass
make lint                 # Ruff lint + format check
make format               # apply Ruff fixes and formatting
make build                # build wheel and sdist
```

Run `make help` to see the full command list.

## Quick Start

```python
import tensorflow as tf
from base_attentive import BaseAttentive

# Create a model
model = BaseAttentive(
    static_input_dim=4,           # 4 static features
    dynamic_input_dim=8,          # 8 dynamic features in history
    future_input_dim=6,           # 6 known future features
    output_dim=2,                 # 2 target variables
    forecast_horizon=24,          # 24-step ahead forecast
    quantiles=[0.1, 0.5, 0.9],   # Uncertainty quantiles
    embed_dim=32,
    attention_units=64,
    num_heads=8,
    dropout_rate=0.15,
)

# Prepare dummy data
BATCH_SIZE = 32
x_static = tf.random.normal([BATCH_SIZE, 4])
x_dynamic = tf.random.normal([BATCH_SIZE, 10, 8])  # 10 time steps
x_future = tf.random.normal([BATCH_SIZE, 24, 6])   # 24 forecast steps

# Make predictions
predictions = model([x_static, x_dynamic, x_future])
print(predictions.shape)  # (32, 24, 3, 2) - [batch, horizon, quantiles, outputs]
```

## Architecture Configuration

Override default architecture with `architecture_config`:

```python
transformer_config = {
    'encoder_type': 'transformer',          # Pure attention encoder
    'decoder_attention_stack': ['cross', 'hierarchical'],
    'feature_processing': 'dense'
}

model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    architecture_config=transformer_config
)
```

Available configuration options:
- **`encoder_type`**: `'hybrid'` (LSTM+Attention) or `'transformer'` (pure attention)
- **`decoder_attention_stack`**: List of attention layers: `['cross', 'hierarchical', 'memory']`
- **`feature_processing`**: `'vsn'` (learnable selection) or `'dense'` (standard layers)

## Documentation

Full documentation is available at: https://base-attentive.readthedocs.io

## Papers & References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Temporal Fusion Transformers (Lim et al., 2021)](https://arxiv.org/abs/1912.09363)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{baseattentive2024,
  author = {Kouadio, L.},
  title = {BaseAttentive: Advanced Attention-based Forecasting Models},
  year = {2024},
  url = {https://github.com/earthai-tech/base-attentive}
}
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## Acknowledgments

- Built on Keras 3 with TensorFlow-first support and experimental JAX/Torch paths
- Inspired by state-of-the-art time series forecasting research
