# Development Guide

**Last Updated:** April 10, 2026

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Contributing](#contributing)
- [Build & Deployment](#build--deployment)

---

## Getting Started

### Prerequisites

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Git**: Version control
- **Make** (optional): For convenience commands
- **Conda**: Recommended for environment management

### Quick Setup

```bash
# Clone repository
git clone https://github.com/earthai-tech/base-attentive.git
cd base-attentive

# Create conda environment
conda create -n base-attentive python=3.10
conda activate base-attentive

# Install in editable mode with dev extras
pip install -e ".[dev,tensorflow]"

# Verify installation
python -c "from base_attentive import BaseAttentive; print('✅ Ready')"
```

### Using Make

```bash
# Install with TensorFlow backend
make install-tensorflow

# Run all checks
make all

# Show all available commands
make help
```

---

## Development Setup

### Environment Setup

#### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n base-attentive python=3.10 -y

# Activate
conda activate base-attentive

# Install dependencies
pip install -e ".[dev,tensorflow]"
```

#### Option 2: Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,tensorflow]"
```

### Installing Extras

```bash
# Development tools (testing, linting, formatting)
pip install -e ".[dev]"

# TensorFlow backend (primary)
pip install -e ".[tensorflow]"

# JAX backend (experimental)
pip install -e ".[jax]"

# PyTorch backend (experimental)
pip install -e ".[torch]"

# All extras
pip install -e ".[dev,tensorflow,jax,torch]"
```

### IDE Setup

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

#### PyCharm

1. **Preferences → Project → Python Interpreter**
   - Click "Add..." and select your virtual environment
2. **Preferences → Tools → Python Integrated Tools**
   - Set Default test runner to "pytest"
3. **Preferences → Editor → Code Style → Python**
   - Set line length to 95 (Ruff default)

---

## Project Structure

### Directory Layout

```
base-attentive/
├── src/
│   └── base_attentive/              ← Main package (import name)
│       ├── __init__.py              ← Package entry point
│       ├── backend.py               ← Keras backend detection
│       ├── logging.py               ← Logging utilities
│       ├── validation.py            ← Input validation
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── docs.py              ← Docstring components
│       │   └── property.py          ← NNLearner mixin
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base_attentive.py    ← Main model class
│       │   ├── checks.py            ← Runtime checks
│       │   └── handlers.py          ← Error handlers
│       │
│       ├── components/              ← Neural network components
│       │   ├── __init__.py
│       │   ├── attention.py         ← Attention layers
│       │   ├── encoder_decoder.py   ← Encoder/decoder components
│       │   ├── losses.py            ← Loss functions
│       │   ├── *_utils.py           ← Helper utilities
│       │   └── ...
│       │
│       ├── compat/                  ← Compatibility layer
│       │   ├── __init__.py
│       │   ├── sklearn.py           ← Scikit-learn validators
│       │   ├── tf.py                ← TensorFlow compatibility
│       │   └── types.py             ← Type definitions
│       │
│       ├── models/                  ← Model utilities
│       │   ├── __init__.py
│       │   ├── comp_utils.py        ← Component utilities
│       │   └── utils.py             ← Configuration utilities
│       │
│       ├── utils/                   ← General utilities
│       │   ├── __init__.py
│       │   ├── deps_utils.py        ← Dependency checking
│       │   └── generic_utils.py     ← Generic helpers
│       │
│       └── validation/              ← Validation module
│           ├── __init__.py
│           └── validators.py        ← Data validators
│
├── tests/                           ← Test suite
│   ├── conftest.py                  ← Pytest fixtures
│   ├── test_*.py                    ← Test modules
│   └── __pycache__/
│
├── examples/                        ← Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_hybrid_vs_transformer.ipynb
│   └── 03_attention_stack_configuration.ipynb
│
├── docs/                            ← Documentation (optional)
│   ├── source/
│   └── Makefile
│
├── Configuration Files
│   ├── pyproject.toml               ← Modern package config (PEP 517)
│   ├── pytest.ini                   ← Pytest configuration
│   ├── setup.cfg                    ← Setup configuration
│   └── tox.ini                      ← Tox configuration
│
├── Documentation Files
│   ├── README.md                    ← Project overview
│   ├── ARCHITECTURE.md              ← Architecture overview
│   ├── ARCHITECTURE_DEEP_DIVE.md    ← Detailed architecture
│   ├── API_DOCUMENTATION.md         ← API reference
│   ├── DEVELOPMENT_GUIDE.md         ← This file
│   ├── SETUP_GUIDE.md               ← Setup instructions
│   ├── QUICK_REFERENCE.md           ← Quick lookup
│   ├── CONTRIBUTING.md              ← Contribution guide
│   └── CODE_OF_CONDUCT.md           ← Community standards
│
├── CI/CD
│   ├── .github/
│   │   └── workflows/
│   │       ├── tests.yml            ← Test workflow
│   │       ├── code-quality.yml     ← Linting workflow
│   │       ├── documentation.yml    ← Docs build
│   │       └── pypi-release.yml     ← Release workflow
│   │
│   └── Makefile                     ← Development commands
│
├── Git Configuration
│   ├── .gitignore                   ← Ignore patterns
│   └── .gitattributes
│
└── Root Files
    ├── LICENSE                      ← Apache 2.0
    ├── MANIFEST.md                  ← Project manifest
    └── PUSH_CHECKLIST.md            ← Pre-push checklist
```

### When to Add Files

| Location | Use Case |
|----------|----------|
| `src/base_attentive/` | Core model code |
| `src/base_attentive/components/` | Reusable NN components |
| `src/base_attentive/utils/` | Helper utilities |
| `src/base_attentive/compat/` | Framework compatibility |
| `tests/` | Unit/integration tests |
| `examples/` | Jupyter notebooks |
| `docs/` | Sphinx documentation |

---

## Code Standards

### Python Style

We follow **PEP 8** with these conventions:

```python
# Line length: 95 characters (Ruff default)
# Quote style: Double quotes (PEP 257 docstrings)
# Indentation: 4 spaces

# Imports grouped: Built-in → Third-party → Local
from __future__ import annotations

import copy
import warnings
from numbers import Integral

import numpy as np
import tensorflow as tf

from .local_module import LocalClass
```

### Type Hints

All functions should have type hints:

```python
from typing import Any, Iterable, Optional, Union

def process_tensor(
    tensor: tf.Tensor,
    scale: float = 1.0,
    normalize: bool = True
) -> tf.Tensor:
    """Process a tensor with optional normalization.
    
    Parameters
    ----------
    tensor : tf.Tensor
        Input tensor to process
    scale : float, default=1.0
        Scaling factor
    normalize : bool, default=True
        Whether to normalize
        
    Returns
    -------
    tf.Tensor
        Processed tensor
    """
    # Implementation
    return result
```

### Docstrings

Use NumPy-style docstrings:

```python
def example_function(param1: int, param2: str = "default") -> dict:
    """Brief description in one line.
    
    Longer description explaining the function in more detail.
    Can span multiple lines and paragraphs.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, default="default"
        Description of param2
        
    Returns
    -------
    dict
        Description of return value
        
    Raises
    ------
    ValueError
        Description of when ValueError is raised
        
    See Also
    --------
    related_function : Description
    
    Examples
    --------
    >>> result = example_function(10, "test")
    >>> print(result)
    {'status': 'success'}
    """
    pass
```

### Logging

Use the logging module consistently:

```python
from base_attentive.logging import get_logger

logger = get_logger(__name__)

logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### Error Handling

```python
# Use specific exceptions
def validate_input(value: float) -> None:
    if value < 0:
        raise ValueError(f"Expected positive value, got {value}")
    
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric type, got {type(value)}")

# Include context in error messages
try:
    result = risky_operation()
except SomeError as e:
    logger.error(f"Operation failed: {e}")
    raise RuntimeError(f"Could not complete operation: {e}") from e
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_backend.py -v

# Run specific test
pytest tests/test_backend.py::TestBackendModule::test_get_backend_invalid -v

# Run with coverage
pytest tests/ --cov=src/base_attentive --cov-report=html

# Run in watch mode (with pytest-watch)
ptw tests/
```

### Using Make

```bash
# Quick local tests
make test-fast

# Full test suite
make test

# With coverage report
make test-cov

# View coverage HTML
make cov-report  # Then open htmlcov/index.html
```

### Writing Tests

```python
import pytest
import numpy as np
from base_attentive import BaseAttentive

class TestBaseAttentive:
    """Tests for BaseAttentive class."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return BaseAttentive(
            static_input_dim=4,
            dynamic_input_dim=8,
            future_input_dim=6,
            output_dim=2,
            forecast_horizon=24,
            embed_dim=16  # Smaller for tests
        )
    
    def test_initialization(self, model):
        """Test model initializes correctly."""
        assert model.output_dim == 2
        assert model.forecast_horizon == 24
        # More assertions...
    
    def test_forward_pass(self, model):
        """Test forward pass with valid inputs."""
        static = np.random.randn(2, 4).astype('float32')
        dynamic = np.random.randn(2, 100, 8).astype('float32')
        future = np.random.randn(2, 24, 6).astype('float32')
        
        output = model([static, dynamic, future], training=False)
        
        assert output.shape == (2, 24, 2)
```

### Test Organization

```
tests/
├── conftest.py              ← Shared fixtures
├── test_imports.py          ← Import tests
├── test_backend.py          ← Backend tests
├── test_core.py             ← Core functionality
├── test_validation.py       ← Validation tests
└── test_base_attentive_main.py  ← Main model tests
```

---

## Code Quality

### Linting & Formatting

```bash
# Check code style
make lint

# Auto-fix code style
make format

# Full quality check
make quality
```

### Tools Used

| Tool | Purpose | Config |
|------|---------|--------|
| Ruff | Linting & formatting | `pyproject.toml` |
| Black | Code formatting | (Ruff handles it) |
| isort | Import sorting | (Ruff handles it) |
| mypy | Type checking | `pyproject.toml` |
| bandit | Security scanning | (In quality checks) |

### Pre-Commit Hook

```bash
# Install pre-commit
pip install pre-commit

# Create git hook
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Contributing

### Workflow

1. **Fork & Branch**
   ```bash
   git clone https://github.com/yourusername/base-attentive.git
   cd base-attentive
   git checkout -b feature/description
   ```

2. **Make Changes**
   - Follow code standards above
   - Add tests for new functionality
   - Update documentation

3. **Test & Lint**
   ```bash
   make all  # Runs tests + linting
   ```

4. **Commit**
   ```bash
   git commit -m "Add feature: description"
   ```

5. **Push & PR**
   ```bash
   git push origin feature/description
   # Create PR on GitHub
   ```

### Before Submitting PR

```bash
# Use the checklist
cat PUSH_CHECKLIST.md

# Quick pre-push verification
make all
```

---

## Build & Deployment

### Building Package

```bash
# Build wheel and sdist
make build

# Or manually
python -m pip install build
python -m build

# Output in dist/
# ├── base_attentive-1.0.0-py3-none-any.whl
# └── base_attentive-1.0.0.tar.gz
```

### Publishing to PyPI

```bash
# (Maintainers only)

# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*

# Upload to TestPyPI (for testing)
twine upload -r testpypi dist/*
```

### Version Management

Update version in:
1. `src/base_attentive/__init__.py`: `__version__ = "X.Y.Z"`
2. `pyproject.toml`: `version = "X.Y.Z"`

Follow [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

---

## Continuous Integration

### GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yml` | Push/PR | Run tests on Python 3.9-3.12 |
| `code-quality.yml` | Push/PR | Lint, format, security checks |
| `documentation.yml` | Push to main | Build Sphinx docs |
| `pypi-release.yml` | Tag release | Auto-publish to PyPI |

### CI Status Badge

```markdown
[![Tests](https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml/badge.svg)](https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml)
```

---

## Troubleshooting

### Common Issues

#### ImportError: No module named 'tensorflow'

```bash
# Solution: Install TensorFlow
pip install tensorflow>=2.12.0
# Or reinstall with extras
pip install -e ".[tensorflow]"
```

#### Test failures with different backends

```bash
# Solution: Explicitly set backend
export KERAS_BACKEND=tensorflow
pytest tests/
```

#### Type checking errors

```bash
# Solution: Run mypy
mypy src/base_attentive/

# Fix type issues in code
```

---

## Additional Resources

- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md)
- [API Documentation](API_DOCUMENTATION.md)
