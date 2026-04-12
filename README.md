<div align="center">
  <img src="https://raw.githubusercontent.com/earthai-tech/base-attentive/master/docs/_static/base-attentive-logo-long.svg"
       alt="BaseAttentive" width="420" />
</div>

<br/>

<div align="center">

  <a href="https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml?query=branch%3Amaster">
    <img src="https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml/badge.svg?branch=master" alt="Tests" />
  </a>
  <a href="https://base-attentive.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/base-attentive/badge/?version=latest" alt="Documentation" />
  </a>
  <a href="https://github.com/earthai-tech/base-attentive/actions/workflows/pypi-release.yml?query=branch%3Amaster">
    <img src="https://github.com/earthai-tech/base-attentive/actions/workflows/pypi-release.yml/badge.svg?branch=master" alt="Publish" />
  </a>
  <a href="https://codecov.io/gh/earthai-tech/base-attentive">
    <img src="https://codecov.io/gh/earthai-tech/base-attentive/branch/master/graph/badge.svg" alt="Coverage" />
  </a>
  <a href="https://pypi.org/project/base-attentive/">
    <img src="https://img.shields.io/pypi/v/base-attentive.svg" alt="PyPI version" />
  </a>
  <a href="https://github.com/earthai-tech/base-attentive/releases">
    <img src="https://img.shields.io/github/v/release/earthai-tech/base-attentive?display_name=tag&sort=semver&include_prereleases" alt="Version" />
  </a>
  <a href="https://docs.astral.sh/ruff/">
    <img src="https://img.shields.io/badge/code%20style-ruff-46a758.svg" alt="Code Style" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License" />
  </a>

</div>

<br/>

<div align="center">
  <strong>A modular encoder-decoder architecture for sequence-to-sequence time series forecasting with layered attention mechanisms.</strong>
</div>

<br/>

## Overview

**BaseAttentive** is a modular encoder-decoder architecture designed to process three distinct types of inputs:
- **Static features** — constant across time (e.g., geographical coordinates, site properties)
- **Dynamic past features** — historical time series (e.g., sensor readings, observations)
- **Known future features** — forecast-period exogenous variables (e.g., weather forecasts)

It combines these inputs using a configurable attention stack and can serve as a building block for models such as **HALNet** and **PIHALNet**.

## Key Features

**Architecture options**
- Hybrid mode: Multi-scale LSTM + Attention (`objective="hybrid"`)
- Transformer mode: Pure self-attention (`objective="transformer"`)
- Operational shortcuts: TFT-like (`mode="tft"`), PIHALNet-like (`mode="pihal"`)
- Declarative attention stack via `attention_levels`

**Core components**
- Variable Selection Networks (VSN) for learnable feature weighting
- Multi-scale LSTM for hierarchical temporal patterns (`scales`, `multi_scale_agg`)
- Cross, hierarchical, and memory-augmented attention
- Transformer encoder/decoder blocks
- Quantile and probabilistic forecast heads

**V2 system**
- `BaseAttentiveSpec` / `BaseAttentiveComponentSpec` for backend-neutral config
- `ComponentRegistry` and `ModelRegistry` for pluggable components
- `BaseAttentiveV2Assembly` resolver/assembler pattern
- Multi-backend: TensorFlow (stable), JAX, PyTorch (experimental)

**Runtime support**
- Keras 3 multi-backend implementation
- `make_fast_predict_fn` for traced TF inference
- Input validation utilities

## Installation

```bash
pip install base-attentive
```

### Backend extras

```bash
pip install "base-attentive[tensorflow]"   # TensorFlow backend (stable)
pip install "base-attentive[jax]"          # JAX backend (experimental)
pip install "base-attentive[torch]"        # PyTorch backend (experimental)
pip install "base-attentive[all-backends]" # All backends
```

### From source

```bash
git clone https://github.com/earthai-tech/base-attentive.git
cd base-attentive
pip install -e ".[dev,tensorflow]"
```

## Development Setup

If you use `make` (Linux, macOS, WSL, or Git Bash on Windows), the repository
includes a `Makefile` with common development commands:

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
import numpy as np
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
    num_heads=8,
    dropout_rate=0.15,
)

# Prepare inputs
BATCH_SIZE = 32
x_static  = np.random.randn(BATCH_SIZE, 4).astype("float32")
x_dynamic = np.random.randn(BATCH_SIZE, 100, 8).astype("float32")  # 100 history steps
x_future  = np.random.randn(BATCH_SIZE, 24, 6).astype("float32")   # 24 forecast steps

# Make predictions
predictions = model([x_static, x_dynamic, x_future])
print(predictions.shape)  # (32, 24, 3, 2) — [batch, horizon, quantiles, outputs]
```

## Architecture Configuration

Override defaults via `architecture_config`:

```python
from base_attentive import BaseAttentive

model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24,
    mode="tft",                          # TFT-like shortcut
    attention_levels=["cross", "hierarchical"],
    scales=[1, 2, 4],                    # Multi-scale LSTM strides
    multi_scale_agg="average",
    architecture_config={
        "encoder_type": "transformer",   # Pure attention encoder
        "feature_processing": "vsn",     # Variable selection networks
    },
)
```

Available `architecture_config` keys:
- **`encoder_type`**: `'hybrid'` (LSTM+Attention) or `'transformer'` (pure attention)
- **`feature_processing`**: `'vsn'` (learnable selection) or `'dense'` (standard layers)

## Documentation

Full documentation: https://base-attentive.readthedocs.io

## Papers & References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Temporal Fusion Transformers (Lim et al., 2021)](https://arxiv.org/abs/1912.09363)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{baseattentive2026,
  author  = {Kouadio, L.},
  title   = {BaseAttentive: Modular Multi-Backend Encoder-Decoder Architecture for Probabilistic Time Series Forecasting},
  year    = {2026},
  version = {2.0.0rc1},
  url     = {https://github.com/earthai-tech/base-attentive}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
See the [Contributing Guide](https://base-attentive.readthedocs.io/en/latest/contributing.html) for details.

## Acknowledgments

- Built on [Keras 3](https://keras.io/) with TensorFlow-first support and experimental JAX/PyTorch paths
- Inspired by recent time series forecasting research (TFT, PIHALNet, HALNet)
