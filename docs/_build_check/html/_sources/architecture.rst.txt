Architecture Guide
==================

BaseAttentive is built as an encoder-decoder forecasting kernel that combines
three feature streams:

- static features with shape ``(batch, static_features)``
- dynamic historical features with shape
  ``(batch, history_steps, dynamic_features)``
- known future covariates with shape
  ``(batch, horizon, future_features)``

Conceptual flow
---------------

At a high level, the model follows this path:

1. ingest static, historical, and future features
2. project features into a shared embedding space
3. encode temporal context with a hybrid or transformer-style encoder
4. apply the configured decoder attention stack
5. produce point or quantile forecasts over the forecast horizon

Core configuration knobs
------------------------

``BaseAttentive`` exposes most architectural choices directly on the model
constructor.

Important model-level parameters include:

- ``static_input_dim``
- ``dynamic_input_dim``
- ``future_input_dim``
- ``output_dim``
- ``forecast_horizon``
- ``embed_dim``
- ``attention_units``
- ``num_heads``
- ``dropout_rate``
- ``quantiles``

Architecture-level choices live in ``architecture_config``.

Example:

.. code-block:: python

   architecture_config = {
       "encoder_type": "transformer",
       "decoder_attention_stack": ["cross", "hierarchical"],
       "feature_processing": "dense",
   }

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       architecture_config=architecture_config,
   )

Encoder modes
-------------

``encoder_type`` supports:

- ``"hybrid"`` for a mixed temporal-processing path
- ``"transformer"`` for an attention-first encoder path

Feature processing
------------------

``feature_processing`` supports:

- ``"vsn"`` to enable variable-selection-style feature processing
- ``"dense"`` to use standard dense feature projection

Decoder attention stack
-----------------------

``decoder_attention_stack`` is an ordered list of decoder-side attention
components. The exact component set depends on the model configuration, but
common values include:

- ``"cross"``
- ``"hierarchical"``
- ``"memory"``

Output modes
------------

If ``quantiles`` is not set, the model returns point forecasts with a shape
like ``(batch, horizon, output_dim)``.

If ``quantiles`` is provided, the model returns a quantile-aware output with a
shape like ``(batch, horizon, num_quantiles, output_dim)``.
