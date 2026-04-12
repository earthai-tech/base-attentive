Components Reference
====================

This page summarizes the reusable neural-network components that power
``BaseAttentive``. All components are importable from
``base_attentive.components``.

.. code-block:: python

   from base_attentive.components import (
       # Attention
       CrossAttention, HierarchicalAttention, MemoryAugmentedAttention,
       ExplainableAttention, TemporalAttentionLayer,
       MultiResolutionAttentionFusion,
       # Temporal
       MultiScaleLSTM, DynamicTimeWindow,
       # Gating / normalization
       VariableSelectionNetwork, GatedResidualNetwork,
       LearnedNormalization, StaticEnrichmentLayer,
       # Transformer blocks
       TransformerEncoderLayer, TransformerDecoderLayer,
       TransformerEncoderBlock, TransformerDecoderBlock,
       MultiDecoder,
       # Forecast heads
       PointForecastHead, QuantileHead, GaussianHead,
       MixtureDensityHead, QuantileDistributionModeling, CombinedHeadLoss,
       # Positional / embedding
       PositionalEncoding, TSPositionalEncoding,
       MultiModalEmbedding, Activation,
       # Losses
       AdaptiveQuantileLoss, MultiObjectiveLoss, CRPSLoss, AnomalyLoss,
       QuantileLoss, MeanSquaredErrorLoss, HuberLoss, WeightedLoss,
       # Layer utilities
       Gate, LayerScale, ResidualAdd, SqueezeExcite1D, StochasticDepth,
   )

Core Components
---------------

Variable Selection Network
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.VariableSelectionNetwork``

Signature: ``VariableSelectionNetwork(units, num_inputs)``

Learns feature-wise gating weights and projects inputs into a more useful
representation. Each feature receives a learnable importance weight.

.. math::

   \mathbf{z} = g(\mathbf{x}) \odot \mathbf{x}

where :math:`g(\mathbf{x})` is a gating function and :math:`\odot` is
element-wise multiplication.

.. code-block:: python

   vsn    = VariableSelectionNetwork(units=32, num_inputs=8)
   output = vsn([feat_1, feat_2, ..., feat_8])  # list of input tensors

Gated Residual Network
~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.GatedResidualNetwork``

Signature: ``GatedResidualNetwork(units, dropout_rate=0.0)``

Applies a gated linear unit followed by a residual connection and layer
normalization. The backbone of TFT-style gating.

.. code-block:: python

   grn    = GatedResidualNetwork(units=32, dropout_rate=0.1)
   output = grn(x)                     # without context
   output = grn([x, context_vector])   # with optional context

Learned Normalization
~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.LearnedNormalization``

Learnable layer normalization with per-channel scale and shift.

Static Enrichment Layer
~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.StaticEnrichmentLayer``

Signature: ``StaticEnrichmentLayer(units)``

Enriches temporal features with static context information.

.. code-block:: python

   enriched = StaticEnrichmentLayer(units=32)([temporal, static_context])

Multi-Scale LSTM
~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.MultiScaleLSTM``

Signature: ``MultiScaleLSTM(lstm_units, scales=None, return_sequences=False)``

Encodes temporal context across multiple resolutions. Each scale processes
a sub-sampled sequence, then outputs are concatenated.

.. math::

   \mathbf{h}^{(s)} = \text{LSTM}(\mathbf{x}[::s])

   \mathbf{y} = \text{concat}(\mathbf{h}^{(1)}, \mathbf{h}^{(2)}, \ldots)

.. code-block:: python

   lstm   = MultiScaleLSTM(lstm_units=64, scales=[1, 2, 4])
   output = lstm(x)   # x: (batch, time_steps, features)

Dynamic Time Window
~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.DynamicTimeWindow``

Signature: ``DynamicTimeWindow(max_window_size)``

Adaptively selects temporal context windows up to ``max_window_size`` steps.

.. code-block:: python

   dtw    = DynamicTimeWindow(max_window_size=10)
   output = dtw(x)   # x: (batch, time_steps, features)

Attention Layers
----------------

Cross-Attention
~~~~~~~~~~~~~~~

Import: ``base_attentive.components.CrossAttention``

Signature: ``CrossAttention(units, num_heads)``

Fuses decoder queries with encoder memory using scaled dot-product attention.

.. math::

   \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V

.. code-block:: python

   ca     = CrossAttention(units=32, num_heads=4)
   output = ca([query, context])

Hierarchical Attention
~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.HierarchicalAttention``

Signature: ``HierarchicalAttention(units, num_heads)``

Multi-level attention for richer contextual relationships.

.. math::

   \mathbf{a}^{(l)} = \text{softmax}(W^{(l)} \mathbf{h}^{(l)})

.. code-block:: python

   ha     = HierarchicalAttention(units=32, num_heads=4)
   output = ha(x)

Memory-Augmented Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.MemoryAugmentedAttention``

Signature: ``MemoryAugmentedAttention(units, num_heads)``

Maintains a persistent memory bank for long-range pattern retrieval.

.. code-block:: python

   maa    = MemoryAugmentedAttention(units=32, num_heads=4)
   output = maa(x)

Temporal Attention Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.TemporalAttentionLayer``

Signature: ``TemporalAttentionLayer(units, num_heads)``

General-purpose temporal multi-head attention.

.. code-block:: python

   tal    = TemporalAttentionLayer(units=32, num_heads=4)
   output = tal([x, x])   # self-attention

Explainable Attention
~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.ExplainableAttention``

Signature: ``ExplainableAttention(units, num_heads)``

Returns attention weights alongside the output for interpretability.

Multi-Resolution Attention Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.MultiResolutionAttentionFusion``

Signature: ``MultiResolutionAttentionFusion(units, num_heads)``

Fuses representations from multiple temporal resolutions using attention.

.. code-block:: python

   mraf   = MultiResolutionAttentionFusion(units=32, num_heads=4)
   output = mraf([fine_features, coarse_features])

Transformer Blocks
------------------

TransformerEncoderLayer
~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.TransformerEncoderLayer``

Signature: ``TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout_rate=0.1)``

Single transformer encoder layer: self-attention + feed-forward + layer norm.

.. code-block:: python

   enc    = TransformerEncoderLayer(embed_dim=32, num_heads=4, ffn_dim=64)
   output = enc(x)   # x: (batch, seq_len, embed_dim)

TransformerDecoderLayer
~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.TransformerDecoderLayer``

Signature: ``TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout_rate=0.1)``

Single transformer decoder layer: masked self-attention + cross-attention +
feed-forward + layer norm.

.. code-block:: python

   dec    = TransformerDecoderLayer(embed_dim=32, num_heads=4, ffn_dim=64)
   output = dec([tgt, memory])   # tgt: (B, T_dec, D), memory: (B, T_enc, D)

TransformerEncoderBlock
~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.TransformerEncoderBlock``

Signature: ``TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, num_layers, dropout_rate=0.1)``

Stack of ``TransformerEncoderLayer`` blocks.

TransformerDecoderBlock
~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.TransformerDecoderBlock``

Signature: ``TransformerDecoderBlock(embed_dim, num_heads, ffn_dim, num_layers, dropout_rate=0.1)``

Stack of ``TransformerDecoderLayer`` blocks.

Multi-Decoder
~~~~~~~~~~~~~

Import: ``base_attentive.components.MultiDecoder``

Signature: ``MultiDecoder(output_dim, num_horizons)``

Projects the latent representation into horizon-wise forecast outputs with
one dense layer per horizon step.

.. math::

   \hat{\mathbf{y}}_t = W_t \mathbf{h} + \mathbf{b}_t

.. code-block:: python

   md     = MultiDecoder(output_dim=2, num_horizons=24)
   output = md(h)   # h: (batch, embed_dim)

Forecast Heads
--------------

PointForecastHead
~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.PointForecastHead``

Signature: ``PointForecastHead(output_dim, forecast_horizon)``

Single dense output head for point forecasting.

.. code-block:: python

   head   = PointForecastHead(output_dim=2, forecast_horizon=24)
   output = head(h)   # output: (batch, 24, 2)

QuantileHead
~~~~~~~~~~~~

Import: ``base_attentive.components.QuantileHead``

Signature: ``QuantileHead(quantiles, output_dim, forecast_horizon)``

Outputs a separate dense projection for each quantile level.

.. code-block:: python

   head   = QuantileHead(quantiles=[0.1, 0.5, 0.9], output_dim=2, forecast_horizon=24)
   output = head(h)   # output: (batch, 24, 3, 2)

GaussianHead
~~~~~~~~~~~~

Import: ``base_attentive.components.GaussianHead``

Signature: ``GaussianHead(output_dim)``

Outputs mean and log-variance parameters for a Gaussian predictive distribution.

MixtureDensityHead
~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.MixtureDensityHead``

Signature: ``MixtureDensityHead(num_components, output_dim)``

Outputs mixture weights, means, and variances for a Gaussian Mixture Model.

QuantileDistributionModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.QuantileDistributionModeling``

Signature: ``QuantileDistributionModeling(quantiles, output_dim)``

Full quantile distribution modeling head.

CombinedHeadLoss
~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.CombinedHeadLoss``

Signature: ``CombinedHeadLoss(output_dim, forecast_horizon)``

Combines forecast head with loss computation.

Positional Encoding and Embedding
-----------------------------------

PositionalEncoding
~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.PositionalEncoding``

Signature: ``PositionalEncoding(max_length=2048)``

Sinusoidal positional encoding (added to input embeddings).

.. code-block:: python

   pe     = PositionalEncoding(max_length=200)
   output = pe(x)   # x: (batch, seq_len, embed_dim)

TSPositionalEncoding
~~~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.TSPositionalEncoding``

Signature: ``TSPositionalEncoding(max_position, embed_dim)``

Time-series-specific positional encoding with explicit position and
embedding dimensions.

.. code-block:: python

   tspe   = TSPositionalEncoding(max_position=100, embed_dim=32)
   output = tspe(x)

MultiModalEmbedding
~~~~~~~~~~~~~~~~~~~

Import: ``base_attentive.components.MultiModalEmbedding``

Signature: ``MultiModalEmbedding(embed_dim)``

Embeds multi-modal inputs into a shared embedding space.

Activation
~~~~~~~~~~

Import: ``base_attentive.components.Activation``

Signature: ``Activation(activation)``

Wrapped Keras activation layer supporting all standard activations:
``'relu'``, ``'elu'``, ``'selu'``, ``'sigmoid'``, ``'tanh'``, ``'linear'``,
``'gelu'``, ``'swish'``.

Loss Functions
--------------

All loss classes extend ``keras.losses.Loss`` and are registered as
Keras-serializable.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``MeanSquaredErrorLoss``
     - Standard MSE loss
   * - ``QuantileLoss(quantiles)``
     - Pinball / quantile loss for given quantile levels
   * - ``HuberLoss``
     - Huber (smooth L1) loss
   * - ``WeightedLoss(base_loss, weight)``
     - Wraps any loss with a scalar weight
   * - ``AdaptiveQuantileLoss(quantiles)``
     - Adaptive quantile regression loss
   * - ``MultiObjectiveLoss()``
     - Combines multiple loss terms
   * - ``CRPSLoss()``
     - Continuous Ranked Probability Score
   * - ``AnomalyLoss()``
     - Loss for anomaly-aware training

.. code-block:: python

   from base_attentive.components import AdaptiveQuantileLoss

   loss   = AdaptiveQuantileLoss(quantiles=[0.1, 0.5, 0.9])
   result = loss(y_true, y_pred)

Layer Utilities
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class / Function
     - Description
   * - ``Gate(units)``
     - GLU-style gating layer
   * - ``LayerScale(init_value)``
     - Per-channel learnable scaling (ViT-style)
   * - ``ResidualAdd()``
     - Adds two tensors as a residual connection
   * - ``SqueezeExcite1D(ratio)``
     - Squeeze-and-excite channel re-weighting for 1D sequences
   * - ``StochasticDepth(drop_rate)``
     - Stochastic depth / drop-path regularization
   * - ``apply_residual(x, f)``
     - Functional residual: ``x + f``
   * - ``broadcast_like(x, ref, time_axis)``
     - Broadcast 2D tensor to 3D by expanding a time axis
   * - ``ensure_rank_at_least(x, rank)``
     - Expand dims until tensor reaches minimum rank
   * - ``maybe_expand_time(x)``
     - Expand 2D ``(B, D)`` to ``(B, 1, D)`` if needed
   * - ``drop_path(x, drop_prob, training)``
     - Functional drop-path
   * - ``create_causal_mask(length)``
     - Create upper-triangular causal attention mask
   * - ``combine_masks(mask_a, mask_b, mode)``
     - Combine two boolean masks (``'and'``, ``'or'``, ``'xor'``)
   * - ``pad_mask_from_lengths(lengths, max_len)``
     - Boolean padding mask from sequence lengths
   * - ``sequence_mask_3d(data, lengths, mask_2d)``
     - Apply a 2D mask to a 3D tensor
   * - ``aggregate_multiscale_on_3d(x)``
     - Aggregate multi-scale 3D features
   * - ``aggregate_time_window_output(x)``
     - Aggregate dynamic time window output

Utility Functions (components.utils)
--------------------------------------

``resolve_attn_levels(att_levels)``
   Converts ``att_levels`` to a canonical list of attention type strings.

   .. code-block:: python

      from base_attentive.components.utils import resolve_attn_levels

      resolve_attn_levels(None)             # ['cross', 'hierarchical', 'memory']
      resolve_attn_levels("hier")           # ['hierarchical']
      resolve_attn_levels([1, 3])           # ['cross', 'memory']

``configure_architecture(objective, use_vsn, attention_levels, architecture_config)``
   Builds the final architecture config dict from layered inputs.

``resolve_fusion_mode(fusion_mode)``
   Returns ``'integrated'`` or ``'disjoint'``.

See Also
--------

- :doc:`api_reference` — Full API autodoc
- :doc:`architecture_guide` — Architecture overview
- :doc:`usage` — Usage patterns
