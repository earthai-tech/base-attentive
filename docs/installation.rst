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

Install the package with the runtime you want to use:

.. code-block:: bash

   pip install "base-attentive[tensorflow]"
   pip install "base-attentive[jax]"
   pip install "base-attentive[torch]"

For local development:

.. code-block:: bash

   pip install -e ".[dev,tensorflow]"

If you use the repository ``Makefile``, common setup commands are available:

.. code-block:: bash

   make install-dev
   make install-tensorflow
   make install-jax
   make install-torch

Backend selection
-----------------

The runtime backend can be selected before importing the package:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   export KERAS_BACKEND=jax
   export KERAS_BACKEND=torch

The same can also be controlled with ``BASE_ATTENTIVE_BACKEND``.

Notes
-----

- TensorFlow is the recommended runtime today.
- JAX and Torch are available through the backend abstraction layer, but the
  full ``BaseAttentive`` execution path is still experimental on those
  runtimes.
