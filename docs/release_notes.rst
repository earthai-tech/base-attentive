Release Notes
=============

This page documents the releases and key changes to the BaseAttentive project.

Versioning
----------

BaseAttentive follows `Semantic Versioning <https://semver.org/>`_:

- |MAJOR| — incompatible API changes
- |MINOR| — new functionality (backward-compatible)
- |PATCH| — bug fixes (backward-compatible)

Change Categories
-----------------

- |Feature| — new capabilities and enhancements
- |Bugfix| — resolved issues
- |Dependencies| — package updates and compatibility changes
- |Internal| — refactoring, documentation, and architecture improvements

**v1.0.0** — |MAJOR| First Stable Release
==========================================

**V2 Architecture**

- |Feature| Introduced ``BaseAttentiveSpec`` and ``BaseAttentiveComponentSpec``
  frozen dataclasses for backend-neutral model configuration; replaces ad-hoc
  dicts with a typed, validated schema
- |Feature| New ``ComponentRegistry`` and ``ModelRegistry`` classes for
  pluggable component and model registration keyed by ``(name, backend)``
- |Feature| New ``BaseAttentiveV2Assembly`` resolver/assembly pattern for
  constructing models from specs; enables full backend portability
- |Feature| Registered ``projection.dense`` key in the generic V2 component
  registry (fallback for all backends)

**New ``BaseAttentive`` Parameters**

- |Feature| ``mode`` — operational mode shortcut: ``'tft'``, ``'tft_like'``,
  ``'pihal'``, ``'pihal_like'``, or ``None``
- |Feature| ``attention_levels`` — declarative attention stack control;
  accepts string, list, int (1=cross, 2=hierarchical, 3=memory), or ``None``
- |Feature| ``scales`` — explicit multi-scale LSTM strides, e.g. ``[1, 2, 4]``
  or ``'auto'``
- |Feature| ``multi_scale_agg`` — how to merge multi-scale outputs:
  ``'last'``, ``'average'``, ``'flatten'``, ``'sum'``, ``'concat'``
- |Feature| ``final_agg`` — final sequence aggregation:
  ``'last'``, ``'average'``, ``'flatten'``
- |Feature| ``num_encoder_layers`` — configurable encoder depth (default: 2)
- |Feature| ``vsn_units`` — override VSN projection size independently of
  ``embed_dim``
- |Feature| ``max_window_size`` — maximum dynamic time window size (default: 10)
- |Feature| ``memory_size`` — memory bank size for memory-augmented attention
  (default: 100)
- |Feature| ``apply_dtw`` — toggle Dynamic Time Warping alignment (default: True)

**New Backend Utilities**

- |Feature| ``normalize_backend_name(name)`` — normalises aliases
  (e.g. ``"tf"`` → ``"tensorflow"``)
- |Feature| ``detect_available_backends()`` — returns availability and
  version info for all installed backends
- |Feature| ``select_best_backend(require_supported)`` — selects best
  available backend automatically
- |Feature| ``ensure_default_backend(auto_install)`` — guarantees a backend
  is ready
- |Feature| ``check_tensorflow_compatibility()`` — returns ``(bool, msg)``
- |Feature| ``check_torch_compatibility()`` — returns ``(bool, msg)``
- |Feature| ``get_backend_version(name)`` — returns installed version string
- |Feature| ``version_at_least(version_str, minimum)`` — version comparison
- |Feature| ``TorchDeviceManager`` class — CUDA / MPS / CPU device management
  with lazy selection, cache clearing, and device info reporting

**New Components**

- |Feature| ``TransformerEncoderBlock`` / ``TransformerDecoderBlock`` — stacked
  transformer encoder/decoder blocks with configurable ``num_layers``
- |Feature| ``PointForecastHead(output_dim, forecast_horizon)`` — dense point
  forecast output head
- |Feature| ``QuantileHead(quantiles, output_dim, forecast_horizon)`` — separate
  projection per quantile level
- |Feature| ``GaussianHead(output_dim)`` — outputs mean and log-variance for a
  Gaussian predictive distribution
- |Feature| ``MixtureDensityHead(num_components, output_dim)`` — Gaussian
  Mixture Model forecast head
- |Feature| ``TSPositionalEncoding(max_position, embed_dim)`` — time-series
  positional encoding with explicit position and embedding dimensions
- |Feature| ``AdaptiveQuantileLoss(quantiles)`` — adaptive quantile regression
  loss
- |Feature| ``MultiObjectiveLoss()`` — combines multiple loss terms
- |Feature| ``CRPSLoss()`` — Continuous Ranked Probability Score
- |Feature| ``AnomalyLoss()`` — loss for anomaly-aware training
- |Feature| ``Gate(units)`` — GLU-style gating layer
- |Feature| ``LayerScale(init_value)`` — per-channel learnable scaling
  (ViT-style)
- |Feature| ``SqueezeExcite1D(ratio)`` — squeeze-and-excite for 1D sequences
- |Feature| ``StochasticDepth(drop_rate)`` — stochastic depth / drop-path

**Component Utilities**

- |Feature| ``resolve_attn_levels(att_levels)`` — canonical attention list
  from any supported input form
- |Feature| ``configure_architecture(objective, use_vsn, attention_levels,
  architecture_config)`` — merges defaults with user arguments into a final
  architecture config dict
- |Feature| ``resolve_fusion_mode(fusion_mode)`` — returns
  ``'integrated'`` or ``'disjoint'``

**Bug Fixes**

- |Bugfix| Fixed relative import depth in ``components/utils.py``
  (3-level ``...utils`` → 2-level ``..utils``), resolving
  ``ImportError`` on direct module import
- |Bugfix| Registered ``"projection.dense"`` in the generic V2 component
  registry, fixing ``KeyError: "Unknown component key: 'projection.dense'"``
  on backend fallback

**Documentation**

- |Internal| Full documentation site rewrite for v2 API: all 14 RST pages
  updated with correct constructor signatures, new parameters, new architecture
  sections (registry / resolver / assembly), and backend utilities
- |Internal| Custom Sphinx CSS theme with brand colors
  (primary ``#92278f``, secondary ``#ee2a7b``)
- |Internal| HTML README header with centered logo and badges

**Dependencies**

- |Dependencies| Keras >= 3.0.0 required; TensorFlow, JAX, and PyTorch
  supported as backends via Keras 3 multi-backend
- |Dependencies| NumPy >= 1.23, scikit-learn >= 1.2 required
- |Dependencies| Python 3.10+ required (3.9 no longer supported)

**Internal**

- |Internal| Comprehensive test suite covering components, backend utilities,
  compatibility layer, registry, validation, and implementations modules
- |Internal| Test coverage improved significantly with 6 new test files
- |Internal| ``from __future__ import annotations`` added throughout for
  forward-reference compatibility

**v0.5.0** — |MINOR| Pre-release
=================================

**Features**

- |Feature| Added ``ExplainableAttention`` for interpretable attention weights
- |Feature| Introduced ``MultiResolutionAttentionFusion`` component

**Bug Fixes**

- |Bugfix| Fixed edge case in causal masking for variable-length sequences
- |Deprecated| Marked ``BatchNormalization`` as deprecated in favour of
  ``LayerNormalization``

**Internal**

- |Internal| Performance improvements in multi-scale aggregation (~15% speedup)

**v0.4.2** — |PATCH| Patch Release
=====================================

- |Fix| Critical fix for numerical instability in ``QuantileLoss``
- |Security| Updated dependencies to patch a security vulnerability in
  TensorFlow

Badge Reference
---------------

- |Feature| — New features and enhancements
- |Bugfix| — Bug fixes (also: |Fix| or |Bug|)
- |Dependencies| — Dependency updates
- |Internal| — Internal improvements
- |Breaking| — Breaking changes
- |Deprecated| — Deprecated features
- |Security| — Security updates
- |MAJOR| — Major version change
- |MINOR| — Minor version change
- |PATCH| — Patch version change

Getting Help
------------

- `GitHub Issues <https://github.com/earthai-tech/base-attentive/issues>`_
- `Documentation <https://base-attentive.readthedocs.io/>`_
- `Contributing Guide <contributing.html>`_

Changelog
---------

For the complete changelog, see the
`GitHub Releases <https://github.com/earthai-tech/base-attentive/releases>`_
page.
