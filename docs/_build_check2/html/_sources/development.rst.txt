Development
===========

Repository workflow
-------------------

The repository includes a top-level ``Makefile`` with common development
targets:

.. code-block:: bash

   make install-tensorflow
   make test-fast
   make lint
   make format
   make build
   make docs

Testing
-------

For a focused local test run:

.. code-block:: bash

   pytest tests/test_base_attentive_main.py -q
   pytest tests/test_backend.py -q
   pytest tests/test_validation.py -q

For a full repository check:

.. code-block:: bash

   pytest tests/ -q

Documentation workflow
----------------------

The repository uses Sphinx for documentation and provides both local and CI
build paths.

Local build:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

GitHub Actions:

- ``.github/workflows/documentation.yml`` builds the docs on pushes and pull
  requests
- the built site is uploaded as a workflow artifact

Read the Docs:

- the repository can also be connected to Read the Docs through the
  root-level ``.readthedocs.yaml`` configuration file

Contribution focus
------------------

Documentation contributions are especially valuable in these areas:

- runnable examples
- backend compatibility notes
- model configuration recipes
- serialization and deployment guidance
