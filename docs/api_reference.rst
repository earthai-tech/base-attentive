API Reference
=============

Public package entry points
---------------------------

.. code-block:: python

   from base_attentive import BaseAttentive
   from base_attentive import make_fast_predict_fn
   from base_attentive import get_backend, set_backend
   from base_attentive import (
       get_available_backends,
       get_backend_capabilities,
       normalize_backend_name,
       detect_available_backends,
       select_best_backend,
       ensure_default_backend,
   )
   from base_attentive.validation import (
       validate_model_inputs,
       maybe_reduce_quantiles_bh,
       ensure_bh1,
   )
   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec
   from base_attentive.registry import (
       ComponentRegistry, ModelRegistry,
       DEFAULT_COMPONENT_REGISTRY, DEFAULT_MODEL_REGISTRY,
   )

Top-level package
-----------------

.. automodule:: base_attentive
   :members:
   :exclude-members: __getattr__

Core model
----------

.. autoclass:: base_attentive.core.base_attentive.BaseAttentive
   :members:
   :show-inheritance:

V2 Configuration Schema
-----------------------

``BaseAttentiveSpec`` is a frozen dataclass that fully describes a V2 model
without referencing any backend objects.

.. autoclass:: base_attentive.config.schema.BaseAttentiveSpec
   :members:
   :show-inheritance:

.. autoclass:: base_attentive.config.schema.BaseAttentiveComponentSpec
   :members:
   :show-inheritance:

Example:

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec

   spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=32,
       hidden_units=64,
       attention_heads=4,
       dropout_rate=0.1,
       head_type="point",
       backend_name="tensorflow",
       components=BaseAttentiveComponentSpec(
           sequence_pooling="pool.last",
       ),
   )

Registry
--------

The registry stores named builder functions (for components) and assembler
functions (for complete models), keyed by ``(name, backend)``.

.. autoclass:: base_attentive.registry.ComponentRegistry
   :members:
   :show-inheritance:

.. autoclass:: base_attentive.registry.ModelRegistry
   :members:
   :show-inheritance:

Pre-populated default registries:

.. code-block:: python

   from base_attentive.registry import (
       DEFAULT_COMPONENT_REGISTRY,
       DEFAULT_MODEL_REGISTRY,
   )

   DEFAULT_COMPONENT_REGISTRY.list_registered()
   DEFAULT_COMPONENT_REGISTRY.has("encoder.temporal_self_attention", backend="generic")
   builder = DEFAULT_COMPONENT_REGISTRY.resolve(
       "encoder.temporal_self_attention", backend="generic",
   )

Resolver / Assembly
-------------------

.. autofunction:: base_attentive.resolver.component_resolver.build_component

.. autoclass:: base_attentive.resolver.assembly.BaseAttentiveV2Assembly
   :members:
   :show-inheritance:

Backend utilities
-----------------

.. automodule:: base_attentive.backend
   :members:
   :show-inheritance:

Core backend functions:

.. code-block:: python

   from base_attentive import get_backend, set_backend
   from base_attentive import get_available_backends, get_backend_capabilities
   from base_attentive import normalize_backend_name

   b = get_backend()
   print(b.name)                        # e.g. 'tensorflow'
   set_backend("tensorflow")
   get_available_backends()             # ['tensorflow', ...]
   get_backend_capabilities()           # {'name': ..., 'version': ..., ...}
   normalize_backend_name("tf")         # -> "tensorflow"

Detection and selection:

.. code-block:: python

   from base_attentive import detect_available_backends, select_best_backend
   from base_attentive import ensure_default_backend

   info = detect_available_backends()
   # {'tensorflow': {'available': True, 'version': '2.x'}, ...}

   best = select_best_backend()
   name = ensure_default_backend()

Version compatibility checking:

.. code-block:: python

   from base_attentive.backend import (
       check_tensorflow_compatibility,
       check_torch_compatibility,
       get_backend_version,
       version_at_least,
   )

   ok, msg = check_tensorflow_compatibility()
   ok, msg = check_torch_compatibility()
   ver     = get_backend_version("tensorflow")
   ok      = version_at_least("2.13.0", "2.12.0")

PyTorch device utilities
------------------------

.. autofunction:: base_attentive.backend.torch_is_available
.. autofunction:: base_attentive.backend.get_torch_version
.. autofunction:: base_attentive.backend.get_torch_device

.. autoclass:: base_attentive.backend.TorchDeviceManager
   :members:
   :show-inheritance:

Example:

.. code-block:: python

   from base_attentive.backend import TorchDeviceManager, get_torch_device

   device  = get_torch_device(prefer="cuda", verbose=True)

   manager = TorchDeviceManager(prefer="cuda")
   print(manager.device)
   print(manager.get_available_devices())
   info = manager.get_device_info()
   manager.clear_gpu_cache()

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

.. code-block:: python

   from base_attentive.validation import (
       validate_model_inputs,
       maybe_reduce_quantiles_bh,
       ensure_bh1,
   )

   static, dynamic, future = validate_model_inputs(
       [x_static, x_dynamic, x_future],
       static_input_dim=4,
       dynamic_input_dim=8,
   )

   reduced  = maybe_reduce_quantiles_bh(predictions)
   reshaped = ensure_bh1(output)

Runtime helpers
---------------

.. automodule:: base_attentive.runtime
   :members:
   :show-inheritance:

.. code-block:: python

   from base_attentive import make_fast_predict_fn

   fast_predict = make_fast_predict_fn(
       model,
       jit_compile=True,
       reduce_retracing=True,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )
   predictions = fast_predict([x_static, x_dynamic, x_future])

Component utilities
--------------------

.. autofunction:: base_attentive.components.utils.resolve_attn_levels
.. autofunction:: base_attentive.components.utils.configure_architecture
.. autofunction:: base_attentive.components.utils.resolve_fusion_mode

.. code-block:: python

   from base_attentive.components.utils import (
       resolve_attn_levels,
       configure_architecture,
       resolve_fusion_mode,
   )

   resolve_attn_levels(None)                # ['cross', 'hierarchical', 'memory']
   resolve_attn_levels("cross")             # ['cross']
   resolve_attn_levels(1)                   # ['cross']
   resolve_attn_levels(["cross", "memory"]) # ['cross', 'memory']

   arch = configure_architecture(
       objective="hybrid",
       use_vsn=True,
       attention_levels=["cross", "hierarchical"],
   )

   resolve_fusion_mode(None)      # 'integrated'
   resolve_fusion_mode("disjoint") # 'disjoint'

Key Components
--------------

Variable Selection Network
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.VariableSelectionNetwork
   :members:
   :show-inheritance:

Multi-Scale LSTM
~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.MultiScaleLSTM
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

Transformer Encoder Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.TransformerEncoderLayer
   :members:
   :show-inheritance:

Transformer Decoder Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.TransformerDecoderLayer
   :members:
   :show-inheritance:

Multi-Decoder
~~~~~~~~~~~~~

.. autoclass:: base_attentive.components.MultiDecoder
   :members:
   :show-inheritance:
