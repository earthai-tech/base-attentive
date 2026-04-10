# BaseAttentive Architecture Overview

## 📐 Standard Python Package Structure (PEP 517)

```
base-attentive/                        ← Repository root
│
├── pyproject.toml                    ← Modern package config (PEP 517/518)
├── README.md                         ← Project overview
├── LICENSE                           ← Apache 2.0
│
├── src/                              ← Source root (recommended)
│   │
│   └── base_attentive/               ← Main package namespace
│       │                                (Python: import base_attentive)
│       │
│       ├── __init__.py               ← Package entry point
│       │                                (exports: BaseAttentive)
│       │
│       ├── core/                     ← Core implementations
│       │   ├── __init__.py           ← (exports: BaseAttentive)
│       │   └── base_attentive.py     ← BaseAttentive class definition
│       │
│       ├── components/               ← NN components & layers
│       │   ├── __init__.py
│       │   ├── attention.py          ← (TODO) Attention layers
│       │   ├── encoders.py           ← (TODO) Encoder components
│       │   ├── decoders.py           ← (TODO) Decoder components
│       │   └── layers.py             ← (TODO) Custom layers
│       │
│       ├── compat/                   ← Compatibility layer
│       │   └── __init__.py           ← Scikit-learn wrappers
│       │
│       ├── utils/                    ← Utility functions
│       │   ├── __init__.py
│       │   ├── helpers.py            ← (TODO) Helper functions
│       │   └── shape_utils.py        ← (TODO) Shape handling
│       │
│       └── validation/               ← Input validation
│           ├── __init__.py
│           └── validators.py         ← (TODO) Data validators
│
├── tests/                            ← Test suite
│   ├── conftest.py                   ← Pytest fixtures
│   ├── test_imports.py               ← Import tests
│   ├── test_core.py                  ← (TODO) Core tests
│   └── test_components.py            ← (TODO) Component tests
│
└── docs/                             ← Documentation (optional)
    ├── source/
    │   ├── index.rst
    │   ├── api.rst
    │   └── examples/
    └── Makefile
```

---

## 🔗 Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    base_attentive                           │
│                   (__init__.py)                             │
│                  [Entry point]                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌────────┐   ┌──────────┐   ┌──────────┐
    │  core  │   │compat    │   │validation│
    │  ──    │   │ ────     │   │  ────    │
    │BaseAtt │   │Interval  │   │validate  │
    │         │   │StrOpt   │   │inputs    │
    └────┬───┘   └──────────┘   └────┬─────┘
         │                            │
         │                            │
         └────────────────┬───────────┘
                          │
                    Imports from:
                    ┌─────────────┐
                    │  components │
                    │ ─────────── │
                    │CrossAttention
                    │HierarchicalAtt
                    │MemoryAugAtt
                    │VariableSelect
                    │MultiScaleLSTM
                    │etc.
                    └─────────────┘
```

---

## 🏗️ Class Hierarchy

```
BaseAttentive (base_attentive.core.base_attentive)
│
├── Attributes
│   ├── Input dimensions
│   │   ├── static_input_dim
│   │   ├── dynamic_input_dim
│   │   └── future_input_dim
│   ├── Architecture configuration
│   │   ├── mode ('pihal_like' / 'tft_like')
│   │   ├── objective ('hybrid' / 'transformer')
│   │   └── architecture_config (dict)
│   ├── Hyperparameters
│   │   ├── embed_dim, hidden_units, lstm_units
│   │   ├── attention_units, num_heads
│   │   ├── dropout_rate, activation
│   │   └── ... (16+ params)
│   └── Flags
│       ├── use_residuals, use_vsn
│       ├── use_batch_norm, apply_dtw
│       └── quantiles (uncertainty)
│
├── Methods (stub phase)
│   ├── __init__(**config)
│   ├── get_config() → dict
│   ├── from_config(dict) → BaseAttentive (classmethod)
│   ├── summary() → None
│   └── __repr__() → str
│
└── Methods (to be implemented)
    ├── build() → None
    ├── call(inputs, training=False) → predictions
    ├── _build_attentive_layers() → None
    ├── run_encoder_decoder_core() → Tensor
    ├── apply_attention_levels() → Tensor
    └── save() / load()
```

---

## 🔀 Data Flow Architecture

```
Inputs (3 tensors)
    │
    ├─► Static:   (batch_size, static_dim)
    ├─► Dynamic:  (batch_size, time_steps, dynamic_dim)
    └─► Future:   (batch_size, horizon, future_dim)
         │
         ▼
    ┌─────────────────────────────┐
    │  Feature Processing (VSN)   │
    │  or Dense Layers            │
    └────────────┬────────────────┘
                 │
        ┌────────┴────────┬─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌────────┐      ┌────────┐      ┌──────────┐
    │ Static │      │Dynamic │      │ Future   │
    │Context │      │Context │      │Context   │
    └────────┘      └─┬──────┘      └─────┬────┘
                      │                   │
                      ▼                   ▼
                 ┌──────────────────────────────┐
                 │   ENCODER                    │
                 │ ───────────────────────────  │
                 │ Multi-Scale LSTM (hybrid)    │
                 │ or Transformer attention     │
                 │ + Dynamic Time Warping (DTW) │
                 └────────────┬─────────────────┘
                              │
                   Encoder sequences
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
    ┌──────────┐  ┌────────────────┐  ┌────────────────┐
    │Cross     │  │Hierarchical    │  │Memory-Augmented│
    │Attention │  │Attention       │  │Attention       │
    └─────┬────┘  └────────┬───────┘  └────────┬───────┘
          │                │                  │
          └────────────────┼──────────────────┘
                           │
                    ┌──────▼───────┐
                    │ Fusion Layer │
                    │ (attention)  │
                    └──────┬───────┘
                           │
                 ┌─────────▼─────────┐
                 │ Residual Connections
                 │ (if enabled)
                 └─────────┬─────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Time Aggregation   │
                │ (last/avg/flatten)  │
                └──────────┬──────────┘
                           │
                    ┌──────▼───────┐
                    │ Multi-Decoder│
                    │ (horizons)   │
                    └──────┬───────┘
                           │
        ┌──────────────────┤
        │                  │
        ▼                  ▼
    Point Forecast    (Optional)
    Multi-horizon     Quantile Modeling
    (B, H, D)         for Uncertainty
                      (B, H, Q, D)
```

---

## 🔧 Configuration Hierarchy

```
Configuration precedence (highest to lowest):
┌─────────────────────────────────────────────────────────┐
│ 1. architecture_config dict (user-provided)              │
│    └─ encoder_type: 'hybrid' | 'transformer'             │
│    └─ decoder_attention_stack: [layers]                  │
│    └─ feature_processing: 'vsn' | 'dense'               │
├─────────────────────────────────────────────────────────┤
│ 2. Explicit keyword arguments (user-provided)            │
│    └─ objective, use_vsn, attention_levels, etc.         │
├─────────────────────────────────────────────────────────┤
│ 3. Default values (code-defined)                         │
│    └─ embed_dim=32, hidden_units=64, etc.                │
└─────────────────────────────────────────────────────────┘

Example merging:
    defaults = {
        'embed_dim': 32,
        'objective': 'hybrid'
    }
    
    user_config = {
        'architecture_config': {
            'encoder_type': 'transformer'
        },
        'embed_dim': 64
    }
    
    final = {
        'embed_dim': 64,              ← user keyword
        'objective': 'hybrid',         ← default
        'encoder_type': 'transformer'  ← architecture_config
    }
```

---

## 📦 Package Distribution

```
PyPI Package: base-attentive
    ├── Python import name: base_attentive
    ├── Main class: BaseAttentive
    ├── Version: 0.1.0
    ├── License: Apache-2.0
    │
    └── Dependencies:
        ├── tensorflow>=2.12.0 (core)
        ├── scikit-learn>=1.0.0 (compat)
        └── numpy>=1.21.0 (arrays)
        
        Optional [dev]:
        ├── pytest>=7.0 (testing)
        ├── pytest-cov>=4.0 (coverage)
        └── black, flake8, mypy (code quality)
```

---

## 🎯 Development Workflow

```
┌──────────────────┐
│  Edit source     │
│  files in src/   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Install editable │
│ pip install -e . │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Changes take    │
│  effect         │
│  immediately    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Run tests       │
│  pytest tests/   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Import in code  │
│  from base_att.. │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Verify results  │
└──────────────────┘
```

---

## 💡 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`src/` layout** | Prevents import issues, standard in Python community |
| **Package name** | `base_attentive` mirrors repo `base-attentive`, clear purpose |
| **Modular subpackages** | Easy to extend: components/, utils/, validation/, compat/ |
| **Stub implementation** | Allows immediate testing; full code migrated incrementally |
| **PEP 517 config** | Modern, tool-agnostic package management |
| **Apache 2.0 license** | Consistent with geoprior-v3 |

---

## 🚀 Migration Workflow

```
Phase 1: Structure ✅
  ├─ Create directories
  ├─ Create pyproject.toml
  ├─ Create stub classes
  └─ Verify imports

Phase 2: Components ⏳
  ├─ Copy attention modules
  ├─ Copy encoder/decoder modules
  └─ Fix imports

Phase 3: Integration ⏳
  ├─ Copy full BaseAttentive
  ├─ Test with examples
  └─ Verify backward compat

Phase 4: Publication ⏳
  ├─ Write full tests
  ├─ Build documentation
  ├─ Test on PyPI (TestPyPI)
  └─ Publish to PyPI
```

