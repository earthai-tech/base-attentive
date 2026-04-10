Components Reference
====================

Core Components
---------------

Variable Selection Network (VSN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: VariableSelectionNetwork
   :show-inheritance:

Multi-Scale LSTM
~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: MultiScaleLSTM
   :show-inheritance:

Multi-Decoder
~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: MultiDecoder
   :show-inheritance:

Attention Layers
----------------

Cross-Attention
~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: CrossAttention
   :show-inheritance:

Hierarchical Attention
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: HierarchicalAttention
   :show-inheritance:

Memory-Augmented Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: MemoryAugmentedAttention
   :show-inheritance:

Utility Components
------------------

Gating Residual Network
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: GatedResidualNetwork
   :show-inheritance:

Layer Utilities
~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components.layer_utils
   :members:
   :show-inheritance:

Loss Functions
--------------

Quantile Loss
~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: QuantileLoss
   :show-inheritance:

Multi-Objective Loss
~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: MultiObjectiveLoss
   :show-inheritance:

Temporal Components
-------------------

Dynamic Time Window
~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: DynamicTimeWindow
   :show-inheritance:

Positional Encoding
~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: PositionalEncoding
   :show-inheritance:

Helper Functions
----------------

Time Series Aggregation
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: base_attentive.components
   :members: aggregate_multiscale_on_3d, aggregate_time_window_output
   :show-inheritance:

Component Architecture
----------------------

Composition Pattern
~~~~~~~~~~~~~~~~~~~

Components are designed to be composable:

.. code-block:: python

   from keras import layers
   from base_attentive.components import (
       VariableSelectionNetwork,
       CrossAttention,
       MultiDecoder
   )

   class CustomModel(layers.Layer):
       def __init__(self, **config):
           super().__init__()
           self.vsn = VariableSelectionNetwork(**config)
           self.attention = CrossAttention(**config)
           self.decoder = MultiDecoder(**config)

       def call(self, inputs, training=False):
           selected = self.vsn(inputs)
           attended = self.attention(selected)
           output = self.decoder(attended)
           return output

Common Patterns
---------------

Feature Selection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import VariableSelectionNetwork

   vsn = VariableSelectionNetwork(
       input_dim=10,
       output_dim=8,
       hidden_units=32,
   )

   # Learns which features matter
   selected_features = vsn(raw_features)

Multi-Head Attention
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import CrossAttention

   attention = CrossAttention(
       query_dim=32,
       key_dim=32,
       value_dim=32,
       num_heads=8,
       output_dim=32,
   )

   # Apply attention
   output = attention([query, key, value])

Multi-Horizon Decoding
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import MultiDecoder

   decoder = MultiDecoder(
       input_dim=64,
       output_dim=2,
       forecast_horizon=24,
       hidden_units=32,
   )

   # Generate 24 steps ahead
   forecasts = decoder(encoded_input)
   # Shape: (batch, 24, 2)

Integration Examples
--------------------

Full Model Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from keras import layers, Model
   from base_attentive.components import (
       VariableSelectionNetwork,
       MultiScaleLSTM,
       CrossAttention,
       MultiDecoder,
   )

   # Inputs
   static_input = layers.Input((4,), name='static')
   dynamic_input = layers.Input((100, 8), name='dynamic')
   future_input = layers.Input((24, 6), name='future')

   # Feature selection
   vsn = VariableSelectionNetwork(output_dim=32)
   static_embedded = vsn(static_input)
   dynamic_embedded = vsn(dynamic_input)
   future_embedded = vsn(future_input)

   # Encoding
   lstm = MultiScaleLSTM(scales=3, output_dim=64)
   dynamic_encoded = lstm(dynamic_embedded)

   # Attention
   attention = CrossAttention(num_heads=8)
   attended = attention([dynamic_encoded, dynamic_encoded, dynamic_encoded])

   # Decoding
   decoder = MultiDecoder(output_dim=2, forecast_horizon=24)
   output = decoder(attended)

   # Model
   model = Model(
       inputs=[static_input, dynamic_input, future_input],
       outputs=output
   )

See Also
--------

- :doc:`api_reference` - Core API reference
- :doc:`architecture_guide` - Architecture details
