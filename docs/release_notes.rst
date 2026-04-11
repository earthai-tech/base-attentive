Release Notes
=============

This page documents the releases and key changes to the BaseAttentive project.

Development
-----------

Ongoing development and feature work. Check the `GitHub Issues <https://github.com/earthai-tech/base-attentive/issues>`_ and `Pull Requests <https://github.com/earthai-tech/base-attentive/pulls>`_ for current progress.

Versioning
----------

BaseAttentive follows `Semantic Versioning <https://semver.org/>`_:

- |MAJOR| — incompatible API changes
- |MINOR| — new functionality (backward-compatible)
- |PATCH| — bug fixes (backward-compatible)

Change Categories
-----------------

Within each release, changes are organized as:

- |Feature| — new capabilities and enhancements
- |Bugfix| — resolved issues
- |Dependencies| — package updates and compatibility changes
- |Internal| — refactoring, documentation, and architecture improvements

Getting Help
------------

For issues, feature requests, or questions:

- `GitHub Issues <https://github.com/earthai-tech/base-attentive/issues>`_
- `Documentation <https://base-attentive.readthedocs.io/>`_
- `Contributing Guide <contributing.html>`_

Example Release Format
----------------------

**v1.0.0** — |MAJOR| Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A major version bump with breaking changes and significant new features.

**Features**

- |Feature| Added Transformer encoder-decoder support for pure attention-based forecasting
- |Feature| Introduced MemoryAugmentedAttention for improved long-range dependency modeling
- |Feature| New QuantileDistributionModeling head for uncertainty quantification

**Bug Fixes**

- |Bugfix| Fixed gradient overflow in MultiScaleLSTM with very deep stacks
- |Fix| Corrected layer normalization in cross-attention mechanisms
- |Bug| Resolved shape mismatch in dynamic time window slicing

**Breaking Changes**

- |Breaking| Removed deprecated `use_time_distributed` parameter from GatedResidualNetwork
- API signature of `VariableSelectionNetwork` changed to support time-distributed inputs

**Dependencies**

- |Dependencies| Updated Keras requirement to >=3.0.0
- |Dependencies| Added support for JAX and PyTorch backends

**Internal**

- |Internal| Refactored attention utilities for better code maintainability
- |Internal| Improved test coverage to 92%
- |Internal| Enhanced documentation with mathematical formulations

**v0.5.0** — |MINOR| Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features**

- |Feature| Added ExplainableAttention for interpretable attention weights
- |Feature| Introduced MultiResolutionAttentionFusion component

**Bug Fixes**

- |Bugfix| Fixed edge case in causal masking for variable-length sequences
- |Deprecated| Marked BatchNormalization as deprecated in favor of LayerNormalization

**Internal**

- |Internal| Performance improvements in multi-scale aggregation (15% speedup)

**v0.4.2** — |PATCH| Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- |Fix| Critical bug fix for numerical instability in QuantileLoss
- |Security| Updated dependencies to patch security vulnerability in tensorflow

Badge Reference
----------------

Available badges for use in change logs and documentation:

- |Feature| - New features and enhancements
- |Bugfix| - Bug fixes (also available as |Fix| or |Bug|)
- |Dependencies| - Dependency updates
- |Internal| - Internal improvements
- |Breaking| - Breaking changes
- |Deprecated| - Deprecated features
- |Security| - Security updates
- |MAJOR| - Major version change
- |MINOR| - Minor version change
- |PATCH| - Patch version change

Changelog
---------

For the complete changelog, see the `GitHub Releases <https://github.com/earthai-tech/base-attentive/releases>`_ page.
