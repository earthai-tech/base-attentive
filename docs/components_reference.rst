Components Reference
====================

This page summarizes the most important reusable neural-network components
that power ``BaseAttentive``. The classes and helpers listed here are useful
when you want to assemble custom encoder-decoder models without relying only
on the top-level ``BaseAttentive`` abstraction.

Core Components
---------------

Variable Selection Network
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.VariableSelectionNetwork``

Learns feature-wise gating weights and projects inputs into a more useful
representation before they are passed deeper into the model. Each feature
receives a learnable weight indicating its importance.

.. math::

   \mathbf{z} = g(\mathbf{x}) \odot \mathbf{x}

where :math:`g(\mathbf{x})` is a gating function and :math:`\odot` denotes
element-wise multiplication.

Multi-Scale LSTM
~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.MultiScaleLSTM``

Encodes temporal context across multiple resolutions. Each scale processes
sub-sampled sequences, then outputs are concatenated.

.. math::

   \mathbf{h}^{(s)} = \text{LSTM}(\mathbf{x}[::s])

   \mathbf{y} = \text{concat}(\mathbf{h}^{(1)}, \mathbf{h}^{(2)}, \ldots, \mathbf{h}^{(S)})

where :math:`s` is the scale factor and :math:`S` is the number of scales.

Multi-Decoder
~~~~~~~~~~~~~

Import path: ``base_attentive.components.MultiDecoder``

Projects the latent representation into horizon-wise forecast outputs.
For each forecast step, a separate dense layer projects features to the output dimension.

.. math::

   \hat{\mathbf{y}}_t = W_t \mathbf{h} + \mathbf{b}_t

where :math:`W_t` is a horizon-specific weight matrix and :math:`\mathbf{h}` is the encoder hidden state.

Attention Layers
----------------

Cross-Attention
~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.CrossAttention``

Fuses decoder queries with encoder memory using scaled dot-product attention.
This is the key bridge between historical context and forecast generation.

.. math::

   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V

where :math:`Q` is the decoder query, :math:`K` and :math:`V` are encoder key and value.

**Multi-Head Variant:**

.. math::

   \text{MultiHead}(Q, K, V) = \text{concat}(\text{head}_1, \ldots, \text{head}_h) W^O

   \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

Hierarchical Attention
~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.HierarchicalAttention``

Adds structure-aware attention on top of temporal features, capturing
richer contextual relationships at multiple levels of abstraction.

.. math::

   \mathbf{a}^{(l)} = \text{softmax}(W^{(l)} \mathbf{h}^{(l)})

   \mathbf{z}^{(l)} = \sum_i a_i^{(l)} \mathbf{h}_i^{(l)}

where :math:`l` denotes the hierarchy level.

Memory-Augmented Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.MemoryAugmentedAttention``

Provides an extended attention mechanism for longer-range dependencies,
augmenting standard attention with external memory.

.. math::

   \text{Score}(Q, M) = \frac{QM^T}{\sqrt{d}} + \text{memory\_bias}

   \text{Output} = \text{softmax}(\text{Score}) V

where :math:`M` is the memory matrix and :math:`\text{memory\_bias}` provides learnable offsets.

Temporal Attention Layer
~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.TemporalAttentionLayer``

Conditions query context with temporal information. Processes sequences
across time, learning temporal dependencies.

.. math::

   \mathbf{y}_t = \text{Attention}(Q_t, K_{1:t}, V_{1:t})

This ensures the model respects causality when processing sequential data.

Explainable Attention
~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.ExplainableAttention``

Provides interpretable attention weights that highlight which parts
of the input are most influential for each prediction.

.. math::

   \alpha_{ij} = \frac{\exp(s_{ij})}{\sum_k \exp(s_{ik})}

   \text{Importance}_i = \sum_j |\alpha_{ij}|

Multi-Resolution Attention Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.MultiResolutionAttentionFusion``

Combines attention outputs computed at multiple resolutions.

.. math::

   \text{Output} = \sum_{r} w_r \cdot \text{Attention}(\mathbf{x}^{(r)})

where :math:`r` indexes resolutions and :math:`w_r` are learnable weights.

Transformer Layers
-------------------

Transformer Encoder Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.TransformerEncoderLayer``

Standard transformer encoder block with multi-head self-attention
and position-wise feed-forward network.

.. math::

   \text{Attn} = \text{MultiHeadAttention}(\mathbf{x}, \mathbf{x}, \mathbf{x})

   \mathbf{z} = \text{LayerNorm}(\mathbf{x} + \text{Attn})

   \mathbf{y} = \text{LayerNorm}(\mathbf{z} + \text{FFN}(\mathbf{z}))

where :math:`\text{FFN}(\mathbf{z}) = \max(0, \mathbf{z}W_1 + b_1)W_2 + b_2`.

Transformer Decoder Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.TransformerDecoderLayer``

Transformer decoder block with self-attention, cross-attention, and FFN.

.. math::

   \text{Self-Attn} = \text{MultiHeadAttention}(\mathbf{y}, \mathbf{y}, \mathbf{y})

   \mathbf{z}_1 = \text{LayerNorm}(\mathbf{y} + \text{Self-Attn})

   \text{Cross-Attn} = \text{MultiHeadAttention}(\mathbf{z}_1, \mathbf{x}, \mathbf{x})

   \mathbf{z}_2 = \text{LayerNorm}(\mathbf{z}_1 + \text{Cross-Attn})

   \text{Output} = \text{LayerNorm}(\mathbf{z}_2 + \text{FFN}(\mathbf{z}_2))

Transformer Encoder Block
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.TransformerEncoderBlock``

Stack of Transformer encoder layers for deep sequential processing.

Transformer Decoder Block
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.TransformerDecoderBlock``

Stack of Transformer decoder layers for multi-step ahead forecasting.

Supporting Layers
-----------------

Gated Residual Network
~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.GatedResidualNetwork``

Used throughout the architecture for stable feature transformation with gating.
Enables skip connections and controlled information flow.

.. math::

   \mathbf{g} = \text{sigmoid}(W_g \mathbf{h} + b_g)

   \mathbf{y} = \mathbf{g} \odot f(\mathbf{x}) + (1 - \mathbf{g}) \odot \mathbf{x}

where :math:`f` is a residual network and :math:`\mathbf{g}` is the gate.

Learned Normalization
~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.LearnedNormalization``

Instance normalization with learnable scale and shift parameters.

.. math::

   \mathbf{y} = \gamma \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

where :math:`\mu, \sigma` are computed per sample and :math:`\gamma, \beta` are learnable.

Static Enrichment Layer
~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.StaticEnrichmentLayer``

Enriches dynamic features with static context information through gating.

.. math::

   \mathbf{z}_t = \mathbf{x}_t + W[\mathbf{x}_t; \mathbf{s}]

where :math:`\mathbf{s}` denotes static features and :math:`[\cdot]` is concatenation.

Dynamic Time Window
~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.DynamicTimeWindow``

Supports dynamic alignment and adaptive temporal windowing. Slices
sequences at variable positions for alignment.

Layer Scale
~~~~~~~~~~~

Import path: ``base_attentive.components.LayerScale``

Per-channel trainable scale vector (inspired by ConvNeXt).

.. math::

   \mathbf{y} = \gamma \odot \mathbf{x}

where :math:`\gamma` is a learnable per-channel scaling vector.

Residual Add
~~~~~~~~~~~~

Import path: ``base_attentive.components.ResidualAdd``

Simple residual connection combining inputs.

.. math::

   \mathbf{y} = \mathbf{x} + f(\mathbf{x})

Squeeze-Excite (1D)
~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.SqueezeExcite1D``

Channel-wise attention mechanism. Recalibrates channel importance.

.. math::

   \mathbf{s} = \sigma(W_2 \text{ReLU}(W_1 \mathbf{z}))

   \mathbf{y} = \mathbf{s} \odot \mathbf{x}

where :math:`\mathbf{z}` is global average pooling of :math:`\mathbf{x}`.

Stochastic Depth
~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.StochasticDepth``

Randomly drops residual branches during training for regularization.

.. math::

   \mathbf{y} = \begin{cases}
   \mathbf{x} + f(\mathbf{x}) & \text{with probability } 1-p \\
   \mathbf{x} & \text{with probability } p
   \end{cases}

Gate Layer
~~~~~~~~~~

Import path: ``base_attentive.components.Gate``

Simple multiplicative gating mechanism.

.. math::

   \mathbf{y} = \mathbf{g} \odot \mathbf{x}

where :math:`\mathbf{g} \in (0, 1)` is a gating vector.

Positional Encoding
~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.PositionalEncoding``

Injects absolute positional information into embeddings.

.. math::

   PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})

   PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})

where :math:`pos` is position and :math:`d` is embedding dimension.

Time Series Positional Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.TSPositionalEncoding``

Time-series specific positional encoding handling irregular sampling or variable-length sequences.

.. math::

   PE_t = \text{embed}(\Delta t) + \text{embed}(\text{dayofweek}) + \cdots

where :math:`\Delta t` is time delta and additional features capture temporal patterns.

Embedding Layers
-----------------

Multi-Modal Embedding
~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.MultiModalEmbedding``

Embeds multi-modal inputs (static, dynamic, future) into a unified space.

.. math::

   \mathbf{e}_s = W_s \mathbf{x}_s + b_s

   \mathbf{e}_d = W_d \mathbf{x}_d + b_d

   \mathbf{e}_f = W_f \mathbf{x}_f + b_f

where subscripts denote static, dynamic, and future modalities.

Activation Layer
~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.Activation``

Flexible activation layer supporting any Keras or custom activation function.

.. math::

   \mathbf{y} = \phi(\mathbf{x})

where :math:`\phi` can be ReLU, GELU, Tanh, or any other function.

Prediction Heads
----------------

Quantile Head
~~~~~~~~~~~~~

Import path: ``base_attentive.components.QuantileHead``

Predicts multiple quantiles for probabilistic forecasting.

.. math::

   \hat{y}_{q} = W_q \mathbf{h} + b_q \quad \forall q \in \{0.1, 0.5, 0.9, \ldots\}

Point Forecast Head
~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.PointForecastHead``

Standard point forecast (mean prediction).

.. math::

   \hat{y} = W \mathbf{h} + b

Gaussian Head
~~~~~~~~~~~~~

Import path: ``base_attentive.components.GaussianHead``

Predicts parametric Gaussian distribution: mean :math:`\mu` and standard deviation :math:`\sigma`.

.. math::

   \mu = W_\mu \mathbf{h} + b_\mu

   \sigma = \text{softplus}(W_\sigma \mathbf{h} + b_\sigma) + \epsilon

Mixture Density Head
~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.MixtureDensityHead``

Predicts mixture of Gaussians for multi-modal distributions.

.. math::

   p(y|\mathbf{x}) = \sum_{k=1}^K \pi_k N(y | \mu_k, \sigma_k^2)

Quantile Distribution Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.QuantileDistributionModeling``

Explicitly models the predictive distribution via quantile regression.

.. math::

   \mathcal{L}_q = \frac{1}{n} \sum_{i=1}^n q \cdot (y_i - \hat{y}_q)^+ + (1-q) \cdot (\hat{y}_q - y_i)^+

where :math:`(x)^+ = \max(x, 0)`.

Combined Head Loss
~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.CombinedHeadLoss``

Aggregates losses from multiple prediction heads into a single objective.

.. math::

   \mathcal{L}_{\text{total}} = \sum_h \lambda_h \mathcal{L}_h

where :math:`\lambda_h` are loss weights per head.

Loss Functions
---------------

Quantile Loss
~~~~~~~~~~~~~

Import path: ``base_attentive.components.QuantileLoss``

Used for probabilistic forecasting when optimizing quantile outputs.

.. math::

   \mathcal{L}_q(y, \hat{y}) = (q - \mathbb{1}[y < \hat{y}]) \cdot (y - \hat{y})

Adaptive Quantile Loss
~~~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.AdaptiveQuantileLoss``

Quantile loss with adaptive weighting based on prediction confidence.

.. math::

   \mathcal{L} = w(\hat{\sigma}) \cdot \mathcal{L}_q + \lambda \hat{\sigma}

where :math:`w` weighs by predicted uncertainty and :math:`\lambda` penalizes overconfidence.

Multi-Objective Loss
~~~~~~~~~~~~~~~~~~~~

Import path: ``base_attentive.components.MultiObjectiveLoss``

Combines multiple loss terms into a single optimization target.

.. math::

   \mathcal{L}_{\text{total}} = \sum_i \alpha_i \mathcal{L}_i

where :math:`\alpha_i` are learnable or fixed weights.

CRPS Loss
~~~~~~~~~

Import path: ``base_attentive.components.CRPSLoss``

Continuous Ranked Probability Score for probabilistic forecasts.

**Quantile approximation:**

.. math::

   \text{CRPS}_q \approx \frac{1}{M} \sum_{m=1}^M |y - \hat{y}_{(m)}|

**Gaussian closed form:**

.. math::

   \text{CRPS}_N(y) = \sigma \left( \frac{y-\mu}{\sigma} \phi\left(\frac{y-\mu}{\sigma}\right) 
   + \Phi\left(\frac{y-\mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right)

Anomaly Loss
~~~~~~~~~~~~

Import path: ``base_attentive.components.AnomalyLoss``

Specialized loss for anomaly detection combining reconstruction and outlier terms.

.. math::

   \mathcal{L} = \|y - \hat{y}\|^2 + \lambda \cdot \text{anomaly\_score}(y, \hat{y})


Utility Functions
-----------------

Tensor Management
~~~~~~~~~~~~~~~~~

``maybe_expand_time``
   Expands time dimension if needed for compatibility.

.. math::

   \text{shape} = \begin{cases}
   (\text{batch}, \text{time}, \text{features}) & \text{if } \text{rank} = 2 \\
   (\text{batch}, \text{time}, \text{features}) & \text{if } \text{rank} = 3
   \end{cases}

``broadcast_like``
   Broadcasts tensor to match shape of reference tensor.

.. math::

   \text{broadcast}(\mathbf{x}, \text{shape}(\mathbf{ref}))

``ensure_rank_at_least``
   Ensures tensor has minimum rank by adding dimensions.

Residual Patterns
~~~~~~~~~~~~~~~~~

``apply_residual``
   Applies residual connection with optional transformation.

.. math::

   \mathbf{y} = \mathbf{x} + \alpha \cdot f(\mathbf{x})

where :math:`\alpha` is an optional scaling factor.

``drop_path``
   Stochastic depth variant dropping entire spatial paths.

.. math::

   \mathbf{y} = \begin{cases}
   \mathbf{x} + f(\mathbf{x}) & \text{with probability } 1-p \\
   \mathbf{x} & \text{with probability } p
   \end{cases}

Masking Utilities
~~~~~~~~~~~~~~~~~

``create_causal_mask``
   Creates a causal (lower-triangular) attention mask for autoregressive models.

.. math::

   M_{ij} = \begin{cases}
   0 & \text{if } i \geq j \\
   -\infty & \text{if } i < j
   \end{cases}

``combine_masks``
   Combines multiple masks via logical AND operation.

``pad_mask_from_lengths``
   Creates padding mask from sequence lengths.

.. math::

   M_i = \begin{cases}
   0 & \text{if } i < \text{length} \\
   -\infty & \text{if } i \geq \text{length}
   \end{cases}

``sequence_mask_3d``
   Creates 3D masks for batch-wise sequences.

Temporal Aggregation
~~~~~~~~~~~~~~~~~~~~

``aggregate_multiscale_on_3d``
   Reduces 3D tensor across time using multiple scales.

.. math::

   \mathbf{y}^{(s)} = \text{pool}(\mathbf{x}[::s])

``aggregate_time_window_output``
   Aggregates outputs within temporal windows.

.. math::

   \mathbf{y}_w = \text{agg}(\{\mathbf{x}_t : t \in \text{window}_w\})

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
       MultiDecoder,
       TransformerEncoderLayer,
   )

   class CustomModel(layers.Layer):
       def __init__(self, **config):
           super().__init__()
           self.vsn = VariableSelectionNetwork(**config)
           self.encoder = TransformerEncoderLayer(**config)
           self.attention = CrossAttention(**config)
           self.decoder = MultiDecoder(**config)

       def call(self, inputs, training=False):
           # Select important features
           selected = self.vsn(inputs)
           
           # Encode with transformer
           encoded = self.encoder(selected, training=training)
           
           # Cross-attend to encoder
           attended = self.attention(
               [encoded, encoded, encoded],
               training=training
           )
           
           # Decode to forecast horizon
           output = self.decoder(attended, training=training)
           return output

Common Patterns
---------------

Feature Selection with VSN
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import VariableSelectionNetwork

   vsn = VariableSelectionNetwork(
       input_dim=10,
       output_dim=8,
       hidden_units=32,
       dropout_rate=0.1,
   )

   # Learns which features matter
   selected_features = vsn(raw_features, training=True)

Multi-Head Attention
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import CrossAttention

   attention = CrossAttention(
       num_heads=8,
       dropout_rate=0.1,
   )

   # Apply multi-head attention
   # Q: decoder hidden state, K,V: encoder hidden states
   output = attention(
       [query, encoder_output, encoder_output],
       training=True
   )

Probabilistic Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import (
       MultiScaleLSTM,
       QuantileHead,
       QuantileLoss,
   )

   # Multi-scale encoder
   encoder = MultiScaleLSTM(
       lstm_units=64,
       scales=[1, 2, 3],
       return_sequences=False
   )

   # Quantile prediction head
   head = QuantileHead(
       output_dim=1,
       quantiles=[0.1, 0.5, 0.9]
   )

   # Quantile loss
   loss_fn = QuantileLoss()

Multi-Horizon Decoding
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import MultiDecoder

   decoder = MultiDecoder(
       output_dim=1,
       forecast_horizon=24,
   )

   # Projects encoder output to horizon forecasts
   forecasts = decoder(encoded)  # shape: (batch, 24, 1)

Transformer Encoder-Decoder Stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.components import (
       TransformerEncoderBlock,
       TransformerDecoderBlock,
   )

   encoder = TransformerEncoderBlock(
       embed_dim=64,
       num_heads=8,
       ffn_dim=256,
       num_layers=3,
   )

   decoder = TransformerDecoderBlock(
       embed_dim=64,
       num_heads=8,
       ffn_dim=256,
       num_layers=3,
   )

   # Encode historical context
   encoder_output = encoder(x_hist)

   # Decode forecast
   decoder_output = decoder(
       x_future,
       encoder_output=encoder_output
   )

Regularization Techniques
--------------------------

Stochastic Depth
~~~~~~~~~~~~~~~~

Randomly drops residual branches during training to improve generalization:

.. code-block:: python

   from base_attentive.components import StochasticDepth

   depth = StochasticDepth(drop_rate=0.1)
   
   # Usage in residual block
   output = x + depth(f(x), training=training)

Layer Scale
~~~~~~~~~~~

Per-channel scaling helps stabilize training of deep networks:

.. code-block:: python

   from base_attentive.components import LayerScale

   scale = LayerScale(init_value=1e-4)
   
   # Scale residual branch before adding
   output = x + scale(f(x))

Squeeze-Excite Attention
~~~~~~~~~~~~~~~~~~~~~~~

Channel-wise attention recalibrating channel importance:

.. code-block:: python

   from base_attentive.components import SqueezeExcite1D

   se = SqueezeExcite1D(reduction=16)
   
   # Recalibrate channels
   output = se(x)

Mathematical Notation Reference
--------------------------------

**Activation Functions:**

- :math:`\sigma(x)` — sigmoid
- :math:`\tanh(x)` — hyperbolic tangent  
- :math:`\text{ReLU}(x) = \max(0, x)`
- :math:`\text{GELU}(x) = x \Phi(x)` where :math:`\Phi` is the CDF of standard normal

**Operations:**

- :math:`\odot` — element-wise (Hadamard) product
- :math:`\otimes` — outer product
- :math:`[\cdot]` — concatenation
- :math:`\|\cdot\|_2` — L2 norm
- :math:`\mathbb{E}[\cdot]` — expectation

**Distributions:**

- :math:`N(\mu, \sigma^2)` — normal distribution
- :math:`\Phi(x)` — CDF of standard normal
- :math:`\phi(x)` — PDF of standard normal

**Common Layer Operations:**

- LayerNorm: :math:`\text{LN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta`
- BatchNorm: :math:`\text{BN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mathbb{E}[\mathbf{x}]}{\sqrt{\text{Var}[\mathbf{x}] + \epsilon}} + \beta`
- Dropout: :math:`\text{Dropout}(\mathbf{x}) = \mathbf{x} \odot \mathbf{m}` where :math:`\mathbf{m} \sim \text{Bernoulli}(1-p)`

References and Further Reading
-------------------------------

- `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., 2017)
- `Multi-Head Attention <https://arxiv.org/abs/1512.08756>`_ (Bahdanau et al., 2015)
- `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ (Ba et al., 2016)
- `Quantile Regression <https://arxiv.org/abs/1709.01907>`_ (Koenker & Bassett, 1978)
- `CRPS Loss <https://www.jstor.org/stable/2988369>`_ (Hersbach, 2000)
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
