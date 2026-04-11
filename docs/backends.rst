Backend Guide
=============

BaseAttentive is designed around a Keras 3 runtime layer. Today, the package
has comprehensive support for TensorFlow and exposes an experimental path for JAX and Torch.

Support status
--------------

+------------+--------------+------------------------------------------------+
| Backend    | Status       | Notes                                          |
+============+==============+================================================+
| TensorFlow | Recommended  | Thoroughly tested runtime for ``BaseAttentive``      |
+------------+--------------+------------------------------------------------+
| JAX        | Experimental | Runtime abstraction exists, but the full model |
|            |              | path is not yet first-class                    |
+------------+--------------+------------------------------------------------+
| Torch      | Experimental | Runtime abstraction exists, but the full model |
|            |              | path is not yet first-class                    |
+------------+--------------+------------------------------------------------+

Selecting a backend
-------------------

Set the backend before importing ``base_attentive`` or ``keras``:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   export BASE_ATTENTIVE_BACKEND=tensorflow

The package also exposes helper functions for runtime inspection:

.. code-block:: python

   from base_attentive import get_backend
   from base_attentive import get_available_backends
   from base_attentive import get_backend_capabilities
   from base_attentive import set_backend

   set_backend("tensorflow")
   backend = get_backend()
   print(backend.name)
   print(get_available_backends())
   print(get_backend_capabilities("jax"))

How backend resolution works
----------------------------

When you call ``get_backend()`` without an explicit name, BaseAttentive checks
backends in this order:

1. ``BASE_ATTENTIVE_BACKEND``
2. ``KERAS_BACKEND``
3. the backend previously set in the current Python process
4. ``tensorflow``

Current recommendation
----------------------

Use TensorFlow for training, testing, and serialized model workflows today.
JAX and Torch are important compatibility targets, but you should currently
treat them as exploratory rather than production-ready for the full
``BaseAttentive`` model.
