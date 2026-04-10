# BaseAttentive Package Structure & Migration Guide

## 📦 Repository Structure

The `base-attentive` repository is now set up with a modern Python package structure:

```
base-attentive/
├── pyproject.toml                    # Modern Python package config
├── README.md                         # Project overview
├── LICENSE                           # Apache 2.0 License
├── pytest.ini                        # Pytest configuration (optional)
│
├── src/
│   └── base_attentive/              # Main package (namespace)
│       ├── __init__.py              # Package entry point
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   └── base_attentive.py    # Core BaseAttentive class
│       │
│       ├── components/              # Attention & neural components
│       │   └── __init__.py          # (Placeholder for expansion)
│       │
│       ├── compat/                  # Compatibility layer
│       │   └── __init__.py          # Scikit-learn validators, etc.
│       │
│       ├── utils/                   # Helper utilities
│       │   └── __init__.py
│       │
│       └── validation/              # Input validation
│           └── __init__.py          # Model I/O validation
│
├── tests/
│   ├── conftest.py                  # Pytest fixtures
│   └── test_imports.py              # Basic functionality tests
│
└── docs/                            # Documentation (future)
    └── ...
```

## ✅ What's Included

### ✨ Current Implementation

| Component | Status | Notes |
|-----------|--------|-------|
| **BaseAttentive stub** | ✅ Implemented | Core class definition with full docstring |
| **Package structure** | ✅ Complete | Modern `src/` layout following PEP 517 |
| **Configuration system** | ✅ Working | `get_config()` and `from_config()` methods |
| **Validation layer** | ✅ Ready | Input shape & feature dimension checking |
| **Compatibility shims** | ✅ Included | Scikit-learn validators wrapper |
| **Tests framework** | ✅ Setup | Pytest configuration and basic imports test |
| **Documentation** | ✅ Provided | Comprehensive docstrings and README |

### 📝 To Be Completed (Migration from geoprior-v3)

These components need to be copied/refactored from `geoprior/models/`:

| Component | Source File | Destination | Status |
|-----------|-------------|-------------|--------|
| **VariableSelectionNetwork** | `components/layers.py` | `src/base_attentive/components/` | ⏳ Pending |
| **GatedResidualNetwork** | `components/layers.py` | `src/base_attentive/components/` | ⏳ Pending |
| **CrossAttention** | `components/attention.py` | `src/base_attentive/components/` | ⏳ Pending |
| **HierarchicalAttention** | `components/attention.py` | `src/base_attentive/components/` | ⏳ Pending |
| **MemoryAugmentedAttention** | `components/attention.py` | `src/base_attentive/components/` | ⏳ Pending |
| **MultiScaleLSTM** | `components/encoders.py` | `src/base_attentive/components/` | ⏳ Pending |
| **PositionalEncoding** | `components/encoders.py` | `src/base_attentive/components/` | ⏳ Pending |
| **Full BaseAttentive** | `models/_base_attentive.py` | `src/base_attentive/core/base_attentive.py` | ⏳ Pending |

---

## 🚀 Installation & Usage

### Option 1: Development Installation

```bash
cd d:/projects/base-attentive
pip install -e ".[dev]"
```

This installs the package in editable mode, allowing you to modify code and test immediately.

### Option 2: Regular Installation

```bash
cd d:/projects/base-attentive
pip install .
```

### Verify Installation

```python
from base_attentive import BaseAttentive

# Create a model
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    output_dim=2,
    forecast_horizon=24
)

print(model)
print(model.get_config())
```

---

## 📂 Migration Guide: Moving Code from geoprior-v3

### Step 1: Copy Attention Components

**Source:** `d:\projects\geoprior-v3\geoprior\models\components\`

Copy these Python modules to `src/base_attentive/components/`:

```
attention.py        → CrossAttention, HierarchicalAttention, MemoryAugmentedAttention
layers.py           → VariableSelectionNetwork, GatedResidualNetwork
encoders.py         → MultiScaleLSTM, PositionalEncoding
decoders.py         → MultiDecoder, QuantileDistribution
fusion.py           → MultiResolutionAttentionFusion
```

### Step 2: Copy Utility Functions

**Source:** `d:\projects\geoprior-v3\geoprior\models\comp_utils.py`

Copy `resolve_attention_levels()` and other helpers to `src/base_attentive/utils/`.

### Step 3: Update Imports

**Original (geoprior-v3):**
```python
from ..components import CrossAttention
from ..utils import set_default_params
from ..logging import get_logger
```

**New (base-attentive):**
```python
from base_attentive.components import CrossAttention
from base_attentive.utils import set_default_params
from base_attentive.compat import get_logger
```

### Step 4: Replace Full BaseAttentive Class

1. Read `d:\projects\geoprior-v3\geoprior\models\_base_attentive.py`
2. Copy the entire `BaseAttentive` class to `src/base_attentive/core/base_attentive.py`
3. Update all imports to use relative paths within base_attentive package
4. Remove geoprior-specific decorators and registrations

### Step 5: Dependencies

Update `pyproject.toml` if additional packages are needed:

```toml
dependencies = [
    "tensorflow>=2.12.0",
    "scikit-learn>=1.0.0",
    "numpy>=1.21.0",
    # Add more as needed
]
```

---

## 🔄 Integration with geoprior-v3

### Using base-attentive as a Dependency

Once published:

```bash
# In geoprior-v3 environment
pip install base-attentive
```

Then in geoprior code:

```python
# Option A: Import from base-attentive
from base_attentive import BaseAttentive

# Option B: Keep internal version, with optional fallback
try:
    from base_attentive import BaseAttentive as BaseAttentiveExternal
except ImportError:
    from geoprior.models._base_attentive import BaseAttentive as BaseAttentiveExternal
```

---

## 🧪 Testing

### Run Tests

```bash
cd d:/projects/base-attentive
pytest tests/ -v
```

### Add New Tests

Create test files in `tests/`:

```python
# tests/test_base_attentive.py
import pytest
from base_attentive import BaseAttentive

def test_model_creation():
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
        forecast_horizon=24
    )
    assert model.forecast_horizon == 24

def test_model_config(sample_inputs):
    model = BaseAttentive(
        static_input_dim=4,
        dynamic_input_dim=8,
        future_input_dim=6,
    )
    config = model.get_config()
    model2 = BaseAttentive.from_config(config)
    assert model2.static_input_dim == 4
```

---

## 📋 Package Name Recommendations

### Option 1: `base-attentive` (Current)
- **Package import:** `import base_attentive`
- **Class:** `from base_attentive import BaseAttentive`
- ✅ Clear, descriptive, matches GitHub repo name

### Option 2: `attentive`
- **Shorter, simpler**
- ❌ Risk of name collision with other packages

### Option 3: `attentive-forecaster`
- **More descriptive of purpose**
- ✅ Good for PyPI search visibility

**Recommendation:** Keep `base-attentive` (Option 1) for consistency.

---

## 📖 Documentation Structure

Create these in `docs/`:

```
docs/
├── source/
│   ├── index.rst              # Main index
│   ├── getting_started.rst    # Installation & quick start
│   ├── api/
│   │   ├── base_attentive.rst
│   │   ├── components.rst
│   │   └── validation.rst
│   └── examples/
│       ├── hybrid_model.ipynb
│       ├── transformer_model.ipynb
│       └── quantile_forecasting.ipynb
└── Makefile                   # Sphinx build
```

---

## 🎯 Next Steps

1. **✅ Phase 1 (Done):** Create package structure and stub
2. **⏳ Phase 2:** Copy components from geoprior-v3
3. **⏳ Phase 3:** Update imports and test fully
4. **⏳ Phase 4:** Add comprehensive tests
5. **⏳ Phase 5:** Build Sphinx documentation
6. **⏳ Phase 6:** Publish to PyPI

---

## 📍 Current Location

**Repository path:** `d:\projects\base-attentive\`

To start working:

```bash
cd d:\projects\base-attentive

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check package
python -c "from base_attentive import BaseAttentive; print(BaseAttentive)"
```

---

## ❓ Questions?

- **Import issues?** → Check that `pip install -e .` was run from the repo root
- **Import paths wrong?** →Update imports from `geoprior.models.*` to `base_attentive.*`
- **Missing dependencies?** → Add to `pyproject.toml` dependencies list

