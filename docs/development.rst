Development & Contributing
==========================

Setting up Development Environment
----------------------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10+
- Git
- Conda or virtualenv

Quick Setup
~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/earthai-tech/base-attentive.git
   cd base-attentive

   # Create environment
   conda create -n base-attentive python=3.10
   conda activate base-attentive

   # Install with dev extras
   pip install -e ".[dev,tensorflow]"

   # Verify
   python -c "from base_attentive import BaseAttentive; print('Ready')"

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=src/base_attentive --cov-report=html

   # Run a specific test file
   pytest tests/test_backend.py -v

   # Run a specific test case
   pytest tests/test_backend.py::TestBackendUtils::test_normalize_backend_name -v

Code Quality
~~~~~~~~~~~~

.. code-block:: bash

   # Check code style
   ruff check src/ tests/

   # Auto-fix style issues
   ruff format src/ tests/

   # Type checking
   mypy src/base_attentive/

Contributing Workflow
---------------------

1. **Fork & Branch**

   .. code-block:: bash

      git clone https://github.com/yourusername/base-attentive.git
      cd base-attentive
      git checkout -b feature/description

2. **Make Changes**

   - Follow PEP 8 style guide
   - Add docstrings (NumPy format)
   - Add ``from __future__ import annotations`` at the top of every module
   - Add tests for new functionality
   - Update documentation

3. **Test Locally**

   .. code-block:: bash

      pytest tests/ -v
      ruff check src/ tests/
      mypy src/base_attentive/

4. **Commit & Push**

   .. code-block:: bash

      git add .
      git commit -m "feat: Add feature description"
      git push origin feature/description

5. **Create Pull Request**

   - Go to GitHub and create PR
   - Link related issues
   - Describe changes clearly

Code Style
----------

Python Style Guide
~~~~~~~~~~~~~~~~~~

Follow **PEP 8** with these conventions:

.. code-block:: python

   # Line length: 95 characters
   # Quotes: Double quotes
   # Indentation: 4 spaces

   from __future__ import annotations

   import copy
   from typing import Any

   import numpy as np

   from .local import LocalClass

Type Hints
~~~~~~~~~~

All functions must have type hints:

.. code-block:: python

   from __future__ import annotations
   from typing import Optional, Union

   import numpy as np

   def process_features(
       features: np.ndarray,
       scale: float = 1.0,
       normalize: bool = True,
   ) -> np.ndarray:
       """Process array features with optional scaling.

       Parameters
       ----------
       features : np.ndarray
           Input features to process
       scale : float, default=1.0
           Scaling factor
       normalize : bool, default=True
           Whether to normalize

       Returns
       -------
       np.ndarray
           Processed features
       """
       pass

Docstrings
~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def example_function(param1: int, param2: str = "default") -> dict:
       """Brief one-line description.

       Longer description explaining the function in detail.
       Can span multiple paragraphs.

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
           When value is invalid

       See Also
       --------
       related_function : Related function

       Examples
       --------
       >>> result = example_function(10, "test")
       >>> print(result)
       {'status': 'success'}
       """
       pass

Testing
-------

Test Organization
~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── conftest.py                       # Shared fixtures
   ├── test_imports.py                   # Import / public-API surface tests
   ├── test_backend.py                   # Backend selection and utility tests
   ├── test_core.py                      # Core functionality
   ├── test_validation.py                # Input validation
   ├── test_base_attentive_main.py       # End-to-end model tests
   ├── test_cov_backend_gaps.py          # Backend module coverage (v2)
   ├── test_cov_compat_registry.py       # Compat / registry coverage (v2)
   ├── test_cov_components_keras.py      # Keras component tests (v2)
   ├── test_cov_components_pure.py       # Pure-Python component utils (v2)
   ├── test_cov_implementations_gaps.py  # Implementation gaps (v2)
   └── test_cov_validation_module.py     # Validation module (v2)

Writing Tests
~~~~~~~~~~~~~

.. code-block:: python

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
               embed_dim=16,  # Smaller for tests
           )

       def test_initialization(self, model):
           """Test model initializes correctly."""
           assert model.output_dim == 2
           assert model.forecast_horizon == 24

       def test_forward_pass(self, model):
           """Test forward pass with valid inputs."""
           static = np.random.randn(2, 4).astype('float32')
           dynamic = np.random.randn(2, 100, 8).astype('float32')
           future = np.random.randn(2, 24, 6).astype('float32')

           output = model([static, dynamic, future], training=False)
           assert output.shape == (2, 24, 2)

Test Coverage
~~~~~~~~~~~~~

Coverage has been substantially improved in v1.0.0 with six new dedicated
coverage test files added alongside the existing suite.

Priority areas for continued improvement:

- Backend-specific integration paths (JAX, Torch)
- Serialization round-trips (``get_config`` / ``from_config``)
- Edge cases and error handling in component constructors

Repository Workflow
-------------------

The repository includes a top-level ``Makefile`` with common development
targets:

.. code-block:: bash

   make install-tensorflow
   make install-dev
   make test-fast
   make test
   make lint
   make format
   make build
   make docs

Build & Release
---------------

Building Package
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install build tools
   pip install build twine

   # Build distribution
   python -m build

   # Output:
   # dist/base_attentive-1.0.0-py3-none-any.whl
   # dist/base_attentive-1.0.0.tar.gz

Publishing to PyPI
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Upload to PyPI (maintainers only)
   twine upload dist/*

   # Or test PyPI first
   twine upload -r testpypi dist/*

Version Management
~~~~~~~~~~~~~~~~~~

Update version in:

1. ``src/base_attentive/__init__.py``

   .. code-block:: python

      __version__ = "1.0.0"

2. ``pyproject.toml``

   .. code-block:: toml

      version = "1.0.0"

Follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

CI/CD Workflows
---------------

GitHub Actions
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Workflow
     - Trigger
     - Purpose
   * - ``tests.yml``
     - Push / pull request
     - Run the test suite across supported Python versions
   * - ``code-quality.yml``
     - Push / pull request
     - Run linting and quality checks
   * - ``documentation.yml``
     - Push / pull request
     - Build and upload documentation artifacts
   * - ``pypi-release.yml``
     - Release / workflow dispatch
     - Build and publish package distributions

Documentation Workflow
----------------------

The repository uses Sphinx for documentation and provides both local and CI
build paths.

Local build:

.. code-block:: bash

   pip install -e ".[docs]"
   python -m sphinx -b html docs docs/_build/html

GitHub Actions:

- ``.github/workflows/documentation.yml`` builds the docs on pushes and pull
  requests to main; the built site is uploaded as a workflow artifact

Read the Docs:

- The repository can also be connected to Read the Docs through the
  root-level ``.readthedocs.yaml`` configuration file.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'tensorflow'**

.. code-block:: bash

   pip install tensorflow>=2.12.0
   pip install -e ".[tensorflow]"

**Test failures with different backends**

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   pytest tests/

**Type checking errors**

.. code-block:: bash

   mypy src/base_attentive/

**Documentation build fails**

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html -W

Resources
---------

- `Contributing Guide <../CONTRIBUTING.md>`_
- `Code of Conduct <../CODE_OF_CONDUCT.md>`_
- `Architecture Overview <../ARCHITECTURE.md>`_
- `API Documentation <api_reference.html>`_

Getting Help
~~~~~~~~~~~~

- Open an issue on `GitHub <https://github.com/earthai-tech/base-attentive/issues>`_
- Check existing issues and discussions
- Read the documentation at `<https://base-attentive.readthedocs.io/>`_

Contribution focus
------------------

Documentation contributions are especially valuable in these areas:

- Runnable examples with the v2 API
- Backend compatibility notes (JAX, Torch)
- Model configuration recipes
- Serialization and deployment guidance
