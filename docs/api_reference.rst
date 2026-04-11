API Reference
=============

Public package entry points
---------------------------

These are the main public imports most users interact with:

.. code-block:: python

   from base_attentive import BaseAttentive
   from base_attentive import get_backend, set_backend
   from base_attentive import (
       get_available_backends,
       get_backend_capabilities,
   )

Top-level package
-----------------

.. automodule:: base_attentive
   :members:
   :exclude-members: __getattr__

Backend utilities
-----------------

.. automodule:: base_attentive.backend
   :members:
   :show-inheritance:

Learner mixin
-------------

.. automodule:: base_attentive.api.property
   :members:
   :show-inheritance:

Validation helpers
------------------

.. automodule:: base_attentive.validation
   :members:
   :show-inheritance:

Core model
----------

.. autoclass:: base_attentive.core.base_attentive.BaseAttentive
   :members:
   :show-inheritance:

Components
----------

Variable Selection Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.VariableSelectionNetwork
   :members:
   :show-inheritance:

Multi-Scale LSTM
~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.MultiScaleLSTM
   :members:
   :show-inheritance:

Multi-Decoder
~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.MultiDecoder
   :members:
   :show-inheritance:

Cross-Attention
~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.CrossAttention
   :members:
   :show-inheritance:

Hierarchical Attention
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.HierarchicalAttention
   :members:
   :show-inheritance:

Memory-Augmented Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.MemoryAugmentedAttention
   :members:
   :show-inheritance:
