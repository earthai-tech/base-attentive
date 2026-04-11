# Backend Module Documentation

## Quick Start

```python
from base_attentive.backend import get_backend, set_backend, get_backend_capabilities

# Auto-detect best backend
backend = get_backend()

# Or explicitly select
backend = set_backend("tensorflow")

# Check capabilities
caps = get_backend_capabilities()
print(f"Using {caps['name']} v{caps['version']}")
```

## Important Note

⚠️ **File Removal**: The legacy `backend.py` file has been replaced with a removal notice. If you see it in your repo, it can be safely deleted:

```bash
git rm src/base_attentive/backend.py
git commit -m "Remove deprecated backend.py shim"
```

All imports automatically resolve to the `backend/` package. This is standard Python package behavior.

## Module Architecture

The `base_attentive.backend` module provides an intelligent, modular backend selection system for multi-framework support through Keras 3.

### Module Structure

```
backend/
├── __init__.py           # Main entry point with public API
├── base.py               # Base Backend class definition
├── implementations.py    # Concrete backend implementations (TensorFlow, JAX, Torch)
├── detector.py          # Intelligent backend detection and fallback logic
└── version_check.py     # Version checking utilities
```

## Key Features

### 🎯 Intelligent Backend Selection

The system automatically detects and selects the best available backend:

1. **Environment Variable** (`BASE_ATTENTIVE_BACKEND`) — Explicit user override
2. **Keras Environment** (`KERAS_BACKEND`) — Fallback to Keras configuration
3. **Previous Configuration** — In-process backend if already set
4. **Auto-Detection** — Intelligent selection from available backends
5. **Auto-Installation** — Default TensorFlow if no backend available

### 🔄 Smart Fallback Strategy

```python
Priority Order:
1. TensorFlow (tensorflow) ✓ Fully supported
2. JAX (jax)              ⚠ Experimental
3. PyTorch (torch)        ⚠ Experimental
```

### 🔍 Version Checking

Automatic version compatibility validation:
- TensorFlow ≥ 2.10.0 required
- Automatic version detection and reporting
- Clear error messages for incompatible versions

### 📦 Auto-Installation

If no backend is installed:
```python
from base_attentive.backend import ensure_default_backend
backend = ensure_default_backend(auto_install=True)
# Automatically installs tensorflow[and-cuda] if needed
```

## Usage Examples

### Basic Usage

```python
from base_attentive.backend import get_backend, set_backend

# Auto-detect best available backend
backend = get_backend()
print(backend.name)  # E.g., 'tensorflow'

# Explicitly set backend
backend = set_backend("jax")

# Get current backend
backend = get_backend()
```

### Querying Capabilities

```python
from base_attentive.backend import get_backend_capabilities, detect_available_backends

# Get current backend capabilities
caps = get_backend_capabilities()
print(f"Backend: {caps['name']}")
print(f"Version: {caps['version']}")
print(f"Supported: {caps['supported']}")

# Detect all available backends
all_backends = detect_available_backends()
for name, info in all_backends.items():
    if info['available']:
        print(f"{name}: {info['version']}")
```

### Environment Configuration

```bash
# Explicitly set backend via environment
export BASE_ATTENTIVE_BACKEND=tensorflow

# Or use Keras environment variable
export KERAS_BACKEND=jax
```

### Version Checking

```python
from base_attentive.backend import (
    get_backend_version,
    check_tensorflow_compatibility,
    version_at_least,
)

# Get installed version
tf_version = get_backend_version("tensorflow")
print(f"TensorFlow: {tf_version}")

# Check compatibility
is_compatible, msg = check_tensorflow_compatibility()
print(msg)

# Manual version checking
if version_at_least("2.15.0", min_required="2.10.0"):
    print("Version OK")
```

## Backend Implementation Details

### TensorFlow Backend

- **Name**: `tensorflow`
- **Status**: ✅ Fully Supported
- **Required**: TensorFlow ≥ 2.10.0
- **Features**: Full integration with all components

```python
from base_attentive.backend import TensorFlowBackend

backend = TensorFlowBackend()
print(backend.supports_base_attentive)  # True
```

### JAX Backend

- **Name**: `jax`
- **Status**: ⚠️ Experimental
- **Required**: Keras 3 + JAX
- **Note**: Some features not yet implemented

```python
from base_attentive.backend import JaxBackend

backend = JaxBackend()
print(backend.experimental)  # True
print(backend.blockers)      # List of known limitations
```

### PyTorch Backend

- **Name**: `torch` or `pytorch`
- **Status**: ⚠️ Experimental
- **Required**: Keras 3 + PyTorch
- **Note**: Limited feature parity

```python
from base_attentive.backend import TorchBackend

backend = TorchBackend()
print(backend.experimental)  # True
```

## Advanced Topics

### Custom Backend Detection

```python
from base_attentive.backend import select_best_backend, detect_available_backends

# Get detailed backend information
backends_info = detect_available_backends()

# Intelligently select best backend
best = select_best_backend(
    prefer="jax",              # Prefer JAX if available
    require_supported=True     # Only return supported backends
)
```

### Error Handling

```python
from base_attentive.backend import get_backend

try:
    # Request a specific backend
    backend = get_backend("tensorflow")
except ValueError as e:
    print(f"Backend not available: {e}")
    # Fall back to auto-detection
    backend = get_backend()
```

## Migration Guide

### From Old `backend.py` to New Module

**Old style (deprecated but still works):**
```python
from base_attentive.backend import get_backend
```

**New style (recommended):**
```python
from base_attentive.backend import get_backend
```

The API remains the same! The backward compatibility layer in `backend.py` automatically re-exports everything from the new `backend` package.

### Breaking Changes

None! The new module is fully backward compatible.

## Internal Architecture

### Backend Public API Exports

From `base_attentive.backend/__init__.py`:

```python
# Base classes
Backend
TensorFlowBackend, JaxBackend, TorchBackend, PyTorchBackend

# Core functions
get_backend()
set_backend()
get_available_backends()
get_backend_capabilities()
normalize_backend_name()

# Detection and selection
detect_available_backends()
select_best_backend()
ensure_default_backend()

# Version utilities
get_backend_version()
check_tensorflow_compatibility()
parse_version()
version_at_least()
```

## Future Enhancements

- [ ] Support for additional backends (ONNX, Hugging Face)
- [ ] Automatic backend benchmarking
- [ ] Backend health checks on startup
- [ ] Per-layer backend override
- [ ] Automatic model serialization across backends

## Troubleshooting

### "Backend 'tensorflow' is not available"

```python
from base_attentive.backend import get_backend_version
# Check what's installed
tf_version = get_backend_version("tensorflow")
# Install if missing
# pip install tensorflow[and-cuda]
```

### "Keras already loaded with different backend"

Restart Python after changing backends via `set_backend()`.

### Version Compatibility Issues

```python
from base_attentive.backend import check_tensorflow_compatibility
is_compatible, msg = check_tensorflow_compatibility()
print(msg)  # Diagnostic message
```

## References

- [Keras 3 Multi-Backend Documentation](https://keras.io/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/docs/)
