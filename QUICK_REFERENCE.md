# BaseAttentive Package - Quick Reference

## 📦 Package Namespace

```
Package name (PyPI): base-attentive
Import name (Python): base_attentive
Main class: BaseAttentive
```

## 🗂️ Directory Layout Explanation

### `src/base_attentive/` Main Package
- **Location**: `src/base_attentive/`
- **Why `src/`?**: Industry standard (PEP 420), prevents import conflicts during testing
- **Package import**: `import base_attentive`

### `src/base_attentive/core/`
- **Purpose**: Core model implementations
- **Files**: 
  - `base_attentive.py` → Main `BaseAttentive` class definition
- **Import**: `from base_attentive.core import BaseAttentive`

### `src/base_attentive/components/`
- **Purpose**: Reusable neural network components
- **Planned contents**:
  - Attention layers (CrossAttention, HierarchicalAttention, etc.)
  - Encoder components (MultiScaleLSTM, PositionalEncoding)
  - Decoder components (MultiDecoder)
  - Feature selection (VariableSelectionNetwork)
- **Import**: `from base_attentive.components import CrossAttention`

### `src/base_attentive/compat/`
- **Purpose**: Compatibility shims and adapters
- **Current contents**:
  - Scikit-learn validator imports (Interval, StrOptions)
  - Fallback implementations for older versions
- **Import**: `from base_attentive.compat import Interval`

### `src/base_attentive/utils/`
- **Purpose**: Utility functions and helpers
- **Planned contents**:
  - Default parameter setup
  - Attention level resolution
  - Shape utilities
- **Import**: `from base_attentive.utils import set_default_params`

### `src/base_attentive/validation/`
- **Purpose**: Input validation and data checking
- **Current contents**:
  - `validate_model_inputs()` - checks tensor shapes
- **Import**: `from base_attentive.validation import validate_model_inputs`

### `tests/`
- **Purpose**: Unit and integration tests
- **Files**:
  - `conftest.py` - pytest fixtures
  - `test_*.py` - test modules
- **Run tests**: `pytest tests/ -v`

### `docs/` (Optional)
- **Purpose**: Sphinx documentation
- **Contents**: API docs, examples, tutorials

---

## ⚙️ Configuration Management

### Model Configuration (Hierarchical)

```python
# 1. Package defaults (defined in code)
defaults = {
    "embed_dim": 32,
    "hidden_units": 64,
    "num_heads": 4,
}

# 2. Architecture configuration (user override)
arch_config = {
    "encoder_type": "transformer",
    "decoder_attention_stack": ["cross", "hierarchical"],
}

# 3. Model instantiation (applies all)
model = BaseAttentive(
    static_input_dim=4,
    dynamic_input_dim=8,
    future_input_dim=6,
    architecture_config=arch_config,  # Highest priority
    embed_dim=64,                       # Overrides default
    **defaults  # Provides baseline
)
```

### Architecture Config Keys

| Key | Type | Options | Default |
|-----|------|---------|---------|
| `encoder_type` | str | `'hybrid'`, `'transformer'` | `'hybrid'` |
| `decoder_attention_stack` | list | `['cross', 'hierarchical', 'memory']` | All three |
| `feature_processing` | str | `'vsn'`, `'dense'` | `'vsn'` |

---

## 🔄 Import Patterns

### Pattern 1: Direct Import (Recommended)
```python
from base_attentive import BaseAttentive

model = BaseAttentive(...)
```

### Pattern 2: Modular Import
```python
from base_attentive.core import BaseAttentive
from base_attentive.components import CrossAttention
from base_attentive.validation import validate_model_inputs

model = BaseAttentive(...)
attn = CrossAttention(...)
static, dynamic, future = validate_model_inputs(...)
```

### Pattern 3: Namespace Import
```python
import base_attentive

model = base_attentive.BaseAttentive(...)
print(base_attentive.__version__)
```

---

## 🧪 Testing Pattern

```python
# tests/test_core.py
import pytest
from base_attentive import BaseAttentive
from base_attentive.validation import validate_model_inputs

class TestBaseAttentive:
    
    @pytest.fixture
    def model(self):
        return BaseAttentive(
            static_input_dim=4,
            dynamic_input_dim=8,
            future_input_dim=6,
        )
    
    def test_config(self, model):
        config = model.get_config()
        model2 = BaseAttentive.from_config(config)
        assert model2.static_input_dim == model.static_input_dim
```

---

## 📦 Installation Patterns

### Editable Install (Development)
```bash
pip install -e ".[dev]"
```
- Changes to source files are immediately available
- Best for active development

### Regular Install
```bash
pip install .
```
- Package is copied to site-packages
- Best for testing packaging/distribution

### With Optional Dependencies
```bash
pip install ".[dev,docs]"
```

---

## 📊 Dependencies Structure

```
base_attentive/
├── core/ (requires TensorFlow)
├── components/ (requires TensorFlow)
├── validation/ (requires TensorFlow)
├── compat/ (requires scikit-learn)
└── utils/ (no external deps)
```

Minimal set for import:
```
tensorflow>=2.12.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

---

## 🚀 Publishing to PyPI

When ready:

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

---

## 🎯 Implementation Checklist

### Phase 1: Structure (✅ DONE)
- [x] Create directory structure
- [x] Create pyproject.toml
- [x] Create BaseAttentive stub
- [x] Create __init__.py files
- [x] Add documentation

### Phase 2: Core Components (⏳ TODO)
- [ ] Copy attention mechanisms
- [ ] Copy encoder components
- [ ] Copy decoder components
- [ ] Update all imports

### Phase 3: Integration
- [ ] Update BaseAttentive with full implementation
- [ ] Test all components
- [ ] Add comprehensive tests

### Phase 4: Polish
- [ ] Write API documentation
- [ ] Create usage examples
- [ ] Add type hints
- [ ] Performance optimization

---

## 📝 Notes

- **Organization:** The `src/` layout prevents path conflicts and is recommended by setuptools
- **Namespace:** `base_attentive` (with underscore) in Python, `base-attentive` (with hyphen) on PyPI
- **Stub Phase:** Current implementation allows immediate testing of package structure
- **Migration:** Copying code from geoprior-v3 requires minimal refactoring with correct src/ layout
- **Naming Convention:** Classes use PascalCase (`BaseAttentive`), modules use snake_case (`base_attentive`)

