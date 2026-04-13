"""Tests for backend-specific V2 implementations.

Tests that backend-optimized components (TensorFlow, PyTorch, JAX) can be:
- Imported without errors
- Registered with the registry
- Preferentially selected over generic implementations
"""

from __future__ import annotations

import pytest

from base_attentive.registry import (
    ComponentRegistry,
    get_backend_capability_report,
)


class TestTensorFlowBackendImplementations:
    """Test TensorFlow-specific V2 components."""

    def test_tensorflow_implementations_can_be_imported(self):
        """TensorFlow backend module should be importable when TF is available."""
        try:
            import tensorflow  # noqa: F401

            from base_attentive.implementations import (
                tensorflow as tf_impl,
            )

            assert tf_impl is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_tensorflow_denseprojection_builder_is_callable(self):
        """TensorFlow dense projection builder should return a layer."""
        try:
            from base_attentive.implementations.tensorflow import (
                _build_tf_dense_projection,
            )

            layer = _build_tf_dense_projection(units=64)
            assert layer is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_tensorflow_encoder_builder_returns_layer(self):
        """TensorFlow temporal encoder builder should return a layer."""
        try:
            from base_attentive.implementations.tensorflow import (
                _build_tf_temporal_self_attention_encoder,
            )

            encoder = _build_tf_temporal_self_attention_encoder(
                units=32,
                hidden_units=64,
                num_heads=4,
            )
            assert encoder is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_tensorflow_components_registered_with_tensorflow_backend(
        self,
    ):
        """TensorFlow components should be registered with 'tensorflow' backend."""
        try:
            from base_attentive.implementations.tensorflow import (
                ensure_tensorflow_v2_registered,
            )

            registry = ComponentRegistry()
            ensure_tensorflow_v2_registered(registry)

            # Check that components are registered for tensorflow backend
            dense_reg = registry.resolve(
                "projection.dense", backend="tensorflow"
            )
            assert dense_reg is not None
            assert dense_reg.backend == "tensorflow"
        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestPyTorchBackendImplementations:
    """Test PyTorch-specific V2 components."""

    def test_torch_implementations_can_be_imported(self):
        """PyTorch backend module should be importable when Torch is available."""
        try:
            import torch  # noqa: F401

            from base_attentive.implementations import (
                torch as torch_impl,
            )

            assert torch_impl is not None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_torch_dense_projection_builder_is_callable(self):
        """PyTorch dense projection builder should return a module."""
        try:
            from base_attentive.implementations.torch import (
                _build_torch_dense_projection,
            )

            layer = _build_torch_dense_projection(
                units=64, in_features=32
            )
            assert layer is not None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_torch_encoder_builder_returns_module(self):
        """PyTorch temporal encoder builder should return a module."""
        try:
            from base_attentive.implementations.torch import (
                _build_torch_temporal_self_attention_encoder,
            )

            encoder = _build_torch_temporal_self_attention_encoder(
                units=32,
                hidden_units=64,
                num_heads=4,
            )
            assert encoder is not None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_torch_components_registered_with_torch_backend(self):
        """PyTorch components should be registered with 'torch' backend."""
        try:
            from base_attentive.implementations.torch import (
                ensure_torch_v2_registered,
            )

            registry = ComponentRegistry()
            ensure_torch_v2_registered(registry=registry)

            # Check that components are registered for torch backend
            dense_reg = registry.resolve(
                "projection.dense", backend="torch"
            )
            assert dense_reg is not None
            assert dense_reg.backend == "torch"
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestJAXBackendImplementations:
    """Test JAX-specific V2 components."""

    def test_jax_implementations_can_be_imported(self):
        """JAX backend module should be importable when JAX is available."""
        try:
            import jax  # noqa: F401

            from base_attentive.implementations import jax as jax_impl

            assert jax_impl is not None
        except ImportError:
            pytest.skip("JAX not installed")

    def test_jax_dense_projection_builder_is_callable(self):
        """JAX dense projection builder should return a function."""
        try:
            from base_attentive.implementations.jax import (
                _build_jax_dense_projection,
            )

            projection = _build_jax_dense_projection(units=64)
            assert callable(projection)
        except ImportError:
            pytest.skip("JAX not installed")

    def test_jax_encoder_builder_returns_object(self):
        """JAX temporal encoder builder should return an encoder object."""
        try:
            from base_attentive.implementations.jax import (
                _build_jax_temporal_self_attention_encoder,
            )

            encoder = _build_jax_temporal_self_attention_encoder(
                units=32,
                hidden_units=64,
                num_heads=4,
            )
            assert encoder is not None
        except ImportError:
            pytest.skip("JAX not installed")

    def test_jax_components_registered_with_jax_backend(self):
        """JAX components should be registered with 'jax' backend."""
        try:
            from base_attentive.implementations.jax import (
                ensure_jax_v2_registered,
            )

            registry = ComponentRegistry()
            ensure_jax_v2_registered(registry)

            # Check that components are registered for jax backend
            dense_reg = registry.resolve(
                "projection.dense", backend="jax"
            )
            assert dense_reg is not None
            assert dense_reg.backend == "jax"
        except ImportError:
            pytest.skip("JAX not installed")


class TestBackendPreference:
    """Test that backend-specific implementations are preferred over generic."""

    def test_tensorflow_preferred_over_generic_when_registered(self):
        """Registry should prefer TensorFlow implementations."""
        try:
            from base_attentive.implementations.generic import (
                ensure_generic_v2_registered,
            )
            from base_attentive.implementations.tensorflow import (
                ensure_tensorflow_v2_registered,
            )

            registry = ComponentRegistry()

            # Register generic first
            ensure_generic_v2_registered(component_registry=registry)

            # Then register tensorflow
            ensure_tensorflow_v2_registered(registry=registry)

            # TensorFlow should be returned when requesting tensorflow backend
            tf_dense = registry.resolve(
                "projection.dense", backend="tensorflow"
            )
            assert tf_dense.backend == "tensorflow"
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_fallback_to_generic_when_backend_not_available(self):
        """Registry should fall back to generic when backend not registered."""
        from base_attentive.implementations.generic import (
            ensure_generic_v2_registered,
        )

        registry = ComponentRegistry()
        ensure_generic_v2_registered(component_registry=registry)

        # Request a backend that isn't registered - should fall back to generic
        result = registry.resolve(
            "projection.dense", backend="unknown_backend"
        )
        assert result.backend == "generic"


class TestBackendCapabilities:
    """Test backend capability reporting."""

    def test_tensorflow_backend_reports_v2_support(self):
        """TensorFlow backend should report V2 support."""
        try:
            import tensorflow  # noqa: F401

            report = get_backend_capability_report("tensorflow")
            assert report.name == "tensorflow"
            assert report.supports_base_attentive_v2 is True
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_torch_backend_reports_v2_support(self):
        """PyTorch backend should report V2 support."""
        try:
            import torch  # noqa: F401

            report = get_backend_capability_report("torch")
            assert report.name == "torch"
            assert report.supports_base_attentive_v2 is True
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_jax_backend_reports_v2_support(self):
        """JAX backend should report V2 support."""
        try:
            import jax  # noqa: F401

            report = get_backend_capability_report("jax")
            assert report.name == "jax"
            assert report.supports_base_attentive_v2 is True
        except ImportError:
            pytest.skip("JAX not installed")


class TestBackendComponentsIntegration:
    """Integration tests for backend components."""

    def test_all_backend_projections_registered(self):
        """All backend implementations should register projection types."""
        try:
            from base_attentive.implementations.tensorflow import (
                ensure_tensorflow_v2_registered,
            )

            registry = ComponentRegistry()
            ensure_tensorflow_v2_registered(registry)

            projection_types = [
                "projection.dense",
                "projection.static",
                "projection.dynamic",
                "projection.future",
                "projection.hidden",
            ]

            for proj_type in projection_types:
                result = registry.resolve(
                    proj_type, backend="tensorflow"
                )
                assert result is not None, (
                    f"{proj_type} not registered"
                )
                assert result.backend == "tensorflow"
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_all_backend_operations_registered(self):
        """All backend implementations should register operations."""
        try:
            from base_attentive.implementations.tensorflow import (
                ensure_tensorflow_v2_registered,
            )

            registry = ComponentRegistry()
            ensure_tensorflow_v2_registered(registry)

            operations = [
                "encoder.temporal_self_attention",
                "pool.mean",
                "pool.last",
                "fusion.concat",
                "head.point_forecast",
                "head.quantile",
            ]

            for op in operations:
                result = registry.resolve(op, backend="tensorflow")
                assert result is not None, f"{op} not registered"
                assert result.backend == "tensorflow"
        except ImportError:
            pytest.skip("TensorFlow not installed")
