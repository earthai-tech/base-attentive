Installation
============

Requirements
------------

BaseAttentive supports Python 3.10 and newer.

The core package depends on:

- ``keras>=3``
- ``numpy>=1.23``
- ``scikit-learn>=1.2``

Runtime extras
--------------

Install the package together with the runtime backend you want to use:

.. code-block:: bash

   pip install "base-attentive[tensorflow]"
   pip install "base-attentive[jax]"
   pip install "base-attentive[torch]"

If you are working from a source checkout:

.. code-block:: bash

   git clone https://github.com/earthai-tech/base-attentive.git
   cd base-attentive
   pip install -e ".[dev,tensorflow]"

For local development only:

.. code-block:: bash

   pip install -e ".[dev,tensorflow]"

If you use the repository ``Makefile``, common setup commands are:

.. code-block:: bash

   make install-dev
   make install-tensorflow
   make install-jax
   make install-torch

Backend selection
-----------------

The runtime backend must be selected **before** importing ``base_attentive``
or ``keras``:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow   # default / stable
   export KERAS_BACKEND=jax          # experimental
   export KERAS_BACKEND=torch        # experimental

You can also set the backend programmatically before the first import:

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "tensorflow"
   from base_attentive import BaseAttentive

The ``BASE_ATTENTIVE_BACKEND`` environment variable can also be used; it takes
precedence over ``KERAS_BACKEND``.

Verifying your install
----------------------

After installation, validate the import path, version, and backend resolution:

.. code-block:: python

   import base_attentive
   from base_attentive import get_backend, get_available_backends

   print(base_attentive.__version__)   # e.g. "1.0.0"
   print(get_backend().name)           # e.g. "tensorflow"
   print(get_available_backends())     # ['tensorflow', ...]

Build the documentation locally
---------------------------------

Install the docs dependencies and build the Sphinx site:

.. code-block:: bash

   pip install -e ".[docs,tensorflow]"
   python -m sphinx -b html docs docs/_build/html

The generated HTML lives in ``docs/_build/html``.

Notes
-----

- TensorFlow is the current full runtime path used in examples and tests.
- JAX and Torch are available through the backend abstraction layer, but the
  full ``BaseAttentive`` execution path is still experimental on those runtimes.
- Python 3.9 is no longer supported as of v1.0.0; use Python 3.10 or newer.
