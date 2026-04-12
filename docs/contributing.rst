Contributing Guidelines
=======================

We welcome contributions to BaseAttentive! This guide will help you get started.

Code of Conduct
---------------

We are committed to providing a welcoming and inspiring community for all. Please
refer to our `Code of Conduct <../CODE_OF_CONDUCT.md>`_ for our community
standards.

Getting Started
---------------

1. **Fork the repository**

   Click the "Fork" button in the GitHub interface to create your own copy.

2. **Clone your fork**

   .. code-block:: bash

      git clone https://github.com/yourusername/base-attentive.git
      cd base-attentive

3. **Add upstream remote**

   .. code-block:: bash

      git remote add upstream https://github.com/earthai-tech/base-attentive.git

4. **Create a feature branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

5. **Set up development environment**

   .. code-block:: bash

      conda create -n base-attentive python=3.10
      conda activate base-attentive
      pip install -e ".[dev,tensorflow]"

Types of Contributions
----------------------

**Bug Reports**

Submit a GitHub issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

**Feature Requests**

Submit a GitHub issue with:
- Clear title and description
- Use case and motivation
- Proposed API/usage
- Alternative approaches considered

**Documentation**

Improve docstrings, examples, or guides:
- Fix typos and clarity
- Add examples
- Improve structure
- Add missing sections

**Code Quality**

Improve test coverage, performance, or style:
- Add tests for untested code
- Optimize performance
- Refactor for clarity
- Fix linting issues

**Bug Fixes & Features**

Implement fixes and features:
- Follow all guidelines below
- Include tests
- Update documentation
- Add changelog entry

Pull Request Process
--------------------

1. **Before you start**

   - Check for existing issues/PRs
   - Discuss major changes first
   - Install dev dependencies

2. **Make your changes**

   - Follow code standards (see below)
   - Add tests for new functionality
   - Update docstrings
   - Update relevant documentation

3. **Test locally**

   .. code-block:: bash

      # Run tests
      pytest tests/ -v

      # Check coverage
      pytest tests/ --cov=src/base_attentive

      # Check style
      ruff check src/ tests/

      # Format
      ruff format src/ tests/

      # Type checking
      mypy src/base_attentive/

4. **Commit with clear messages**

   .. code-block:: bash

      git add .
      git commit -m "feat: Add feature description

      - Detailed explanation of changes
      - Additional context if needed
      - Fixes #issue_number"

5. **Push to your fork**

   .. code-block:: bash

      git push origin feature/your-feature-name

6. **Create a pull request**

   - Go to GitHub and create a PR
   - Link related issues with "Fixes #123"
   - Describe your changes clearly
   - Explain why the change is needed

7. **Respond to feedback**

   - Be open to suggestions
   - Make requested changes
   - Push updates to the same branch

Code Style Guide
----------------

**Python Style**

- Follow **PEP 8**
- Line length: 95 characters
- Use double quotes for strings
- 4 spaces for indentation
- Add ``from __future__ import annotations`` at the top of every module

**Type Hints**

All functions must have type hints:

.. code-block:: python

   from __future__ import annotations
   from typing import Optional, Tuple

   import numpy as np

   def predict(
       features: np.ndarray,
       quantiles: Optional[list[float]] = None,
   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
       """Make predictions with optional quantiles."""
       pass

**Docstrings**

Use NumPy-style docstrings:

.. code-block:: python

   from __future__ import annotations

   import numpy as np

   def validate_inputs(
       static: np.ndarray,
       dynamic: np.ndarray,
       future: np.ndarray,
   ) -> None:
       """Validate input array shapes and types.

       Parameters
       ----------
       static : np.ndarray
           Static features of shape (batch, static_dim)
       dynamic : np.ndarray
           Dynamic features of shape (batch, time_steps, dynamic_dim)
       future : np.ndarray
           Future features of shape (batch, horizon, future_dim)

       Raises
       ------
       ValueError
           If any array has incompatible shape or type
       TypeError
           If input is not an array

       Examples
       --------
       >>> validate_inputs(s, d, f)  # Returns None if valid
       """
       pass

**Imports**

Group and sort imports:

.. code-block:: python

   # 1. Standard library
   from __future__ import annotations
   from pathlib import Path
   import re

   # 2. Third-party
   import numpy as np
   import keras

   # 3. Local
   from .utils import helper_function
   from .components import BaseComponent

Testing Requirements
--------------------

**Coverage**

- New code must have test coverage
- Aim for >80% line coverage on new modules
- Run: ``pytest tests/ --cov=src/base_attentive --cov-report=html``

**Test Organization**

.. code-block:: python

   import pytest
   import numpy as np
   from base_attentive import BaseAttentive

   class TestMyFeature:
       """Tests for my feature."""

       @pytest.fixture
       def model(self):
           """Create a test model."""
           return BaseAttentive(
               static_input_dim=4,
               dynamic_input_dim=8,
               future_input_dim=6,
               output_dim=1,
               forecast_horizon=12,
               embed_dim=16,
           )

       def test_basic_functionality(self, model):
           """Test basic usage."""
           static  = np.random.randn(2, 4).astype("float32")
           dynamic = np.random.randn(2, 50, 8).astype("float32")
           future  = np.random.randn(2, 12, 6).astype("float32")

           result = model([static, dynamic, future], training=False)
           assert result.shape == (2, 12, 1)

       def test_edge_case(self, model):
           """Test edge case handling."""
           # Test boundary conditions
           pass

       def test_error_handling(self):
           """Test error messages."""
           with pytest.raises(ValueError):
               BaseAttentive(
                   static_input_dim=-1,
                   dynamic_input_dim=8,
                   future_input_dim=6,
                   output_dim=1,
                   forecast_horizon=12,
               )

**Running Tests**

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_backend.py

   # Run specific test
   pytest tests/test_backend.py::TestBackend::test_method

   # Run with coverage
   pytest tests/ --cov=src/base_attentive

Documentation Requirements
---------------------------

**Update Documentation**

- Update docstrings if API changes
- Add examples for new features
- Update relevant .rst files in docs/
- Run Sphinx locally to verify: ``sphinx-build -b html docs docs/_build/html``

**Docstring Format**

.. code-block:: python

   """Brief description (one line).

   Longer description explaining the function, class, or module
   in more detail. Can span multiple lines and paragraphs.

   Parameters
   ----------
   param1 : type
       Description of param1
   param2 : type, optional
       Description of param2

   Returns
   -------
   type
       Description of return value

   Raises
   ------
   ExceptionType
       When this exception is raised

   See Also
   --------
   related_function : Brief description
   AnotherClass : Brief description

   Examples
   --------
   >>> result = function(arg1, arg2)
   >>> print(result)
   expected_output
   """

**Documentation File Updates**

- Update table of contents if adding new sections
- Update cross-references
- Add links to new features
- Keep ASCII diagrams up to date

Common Pitfalls to Avoid
------------------------

**Don't:**

- Commit to main branch directly
- Skip tests or documentation
- Use single-letter variable names
- Add large datasets to repo
- Ignore style warnings
- Change unrelated code

**Do:**

- Create feature branches
- Write tests for all code
- Use descriptive names
- Keep commits focused
- Follow style guide
- Keep changes minimal and focused

Commit Message Format
---------------------

Use conventional commits format:

.. code-block:: text

   type(scope): subject

   body

   footer

**Types:**

- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation changes
- ``style``: Code style (not logic)
- ``refactor``: Code restructuring
- ``perf``: Performance improvements
- ``test``: Test additions or changes
- ``chore``: Build/dependency changes

**Examples:**

.. code-block:: text

   feat(core): Add quantile modeling support

   - Implement QuantileDistributionModeling layer
   - Add quantile_loss function
   - Update BaseAttentive to support quantiles parameter
   - Add tests for quantile functionality

   Fixes #123

   docs(api): Update BaseAttentive docstring

   - Add quantile parameter description
   - Add examples for probabilistic forecasts

Review Process
--------------

**What we look for:**

- Code follows style guide
- Tests pass and coverage is adequate
- Documentation is clear and complete
- Changes are focused and minimal
- Commit messages are clear
- No breaking changes (or documented)

**Timeline:**

- Small PRs (< 100 lines): 1-2 days
- Medium PRs (100-500 lines): 2-5 days
- Large PRs (> 500 lines): 5-10 days
- Complex features: May require discussion

Getting Help
------------

- Read the `Documentation <https://base-attentive.readthedocs.io/>`_
- Ask in `GitHub Discussions <https://github.com/earthai-tech/base-attentive/discussions>`_
- Open an `Issue <https://github.com/earthai-tech/base-attentive/issues>`_ for bugs
- Email: etanoyau@gmail.com

Thank You!
----------

Thank you for contributing to BaseAttentive! Your efforts help make this
project better for everyone.

See Also
--------

- `Code of Conduct <../CODE_OF_CONDUCT.md>`_
- `Development Guide <development.html>`_
- `Architecture Overview <../ARCHITECTURE.md>`_
