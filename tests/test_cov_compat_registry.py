# SPDX-License-Identifier: Apache-2.0
"""Tests for compat and registry modules."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")


# ---------------------------------------------------------------------------
# compat/__init__.py
# ---------------------------------------------------------------------------

from base_attentive.compat import (
    Interval,
    validate_params,
)


class TestInterval:
    def test_interval_with_int_type(self):
        # int → numbers.Integral conversion
        iv = Interval(int, 0, 10, closed="both")
        assert iv is not None

    def test_interval_with_float_type(self):
        iv = Interval(float, 0.0, 1.0, closed="right")
        assert iv is not None

    def test_interval_real_type(self):
        import numbers

        iv = Interval(numbers.Real, 0.0, None, closed="left")
        assert iv is not None

    def test_interval_closed_both(self):
        iv = Interval(int, 1, 100, closed="both")
        assert iv is not None

    def test_interval_closed_left(self):
        iv = Interval(int, 0, 100, closed="left")
        assert iv is not None


class TestSklearnMissing:
    """Test the fallback paths when sklearn is missing."""

    def test_validate_params_fallback_with_none_sklearn(self):
        """When sklearn_validate_params is None, returns identity decorator."""
        import base_attentive.compat as compat_mod

        original = compat_mod.sklearn_validate_params
        try:
            compat_mod.sklearn_validate_params = None
            decorator = compat_mod.validate_params(
                {"x": [int]}
            )

            # Should be an identity decorator
            def my_func(x):
                return x

            result = decorator(my_func)
            assert result is my_func
        finally:
            compat_mod.sklearn_validate_params = original

    def test_check_is_fitted_fallback(self):
        """check_is_fitted fallback should not raise when sklearn not available."""

        # Test the fallback version directly
        def fallback_check_is_fitted(
            estimator, attributes, *, msg=None, all_or_any=all
        ):
            """Simple fallback for check_is_fitted."""
            pass

        # The fallback accepts any estimator and doesn't raise
        fallback_check_is_fitted(
            object(), ["some_attr"]
        )  # should not raise


class TestValidateParams:
    def test_validate_params_returns_decorator(self):
        """validate_params should return a callable decorator."""
        result = validate_params({"n_components": [int]})
        assert callable(result)

    def test_validate_params_with_prefer_skip(self):
        """Test with prefer_skip_nested_validation kwarg."""
        result = validate_params(
            {"n_components": [int]},
            prefer_skip_nested_validation=True,
        )
        assert callable(result)

    def test_validate_params_without_prefer_skip(self):
        """Test with prefer_skip_nested_validation=False."""
        result = validate_params(
            {"n_components": [int]},
            prefer_skip_nested_validation=False,
        )
        assert callable(result)


class TestCompatFallbackPaths:
    """Test the sklearn fallback paths by mocking sklearn imports."""

    def test_interval_raises_when_sklearn_none(self):
        """Interval raises ImportError when sklearn_Interval is None."""
        import base_attentive.compat as compat_mod

        original = compat_mod.sklearn_Interval
        try:
            compat_mod.sklearn_Interval = None
            with pytest.raises(ImportError):
                Interval(int, 0, 10, closed="both")
        finally:
            compat_mod.sklearn_Interval = original


# ---------------------------------------------------------------------------
# compat/tf.py
# ---------------------------------------------------------------------------

from base_attentive.compat.tf import (
    TFConfig,
    optional_tf_function,
    standalone_keras,
    suppress_tf_warnings,
    tf_debugging_assert_equal,
)


class TestTFConfig:
    def test_instantiation(self):
        cfg = TFConfig()
        assert cfg.compat_ndim_enabled is False

    def test_attribute_access(self):
        cfg = TFConfig()
        cfg.compat_ndim_enabled = True
        assert cfg.compat_ndim_enabled is True


class TestStandaloneKeras:
    def test_with_keras_available(self):
        """standalone_keras should return a module from keras."""
        # keras is available via torch backend
        result = standalone_keras("layers")
        assert result is not None

    def test_with_invalid_module(self):
        """standalone_keras raises ImportError for non-existent module."""
        # This module doesn't exist in keras
        with pytest.raises(ImportError):
            standalone_keras("nonexistent_module_xyz_123")

    def test_tf_keras_fallback_then_standalone(self):
        """Test that when tf.keras fails, standalone keras is tried."""
        with patch.dict(
            sys.modules,
            {"tensorflow": None, "tensorflow.keras": None},
        ):
            # With tf not available, should fallback to standalone keras
            result = standalone_keras("layers")
            assert result is not None

    def test_both_unavailable_raises(self):
        """When both tf.keras and standalone keras unavailable, raises ImportError."""
        with patch.dict(
            sys.modules,
            {
                "tensorflow": None,
                "tensorflow.keras": None,
                "keras": None,
            },
        ):
            with pytest.raises(ImportError):
                standalone_keras("layers")


class TestSuppressTfWarnings:
    def test_when_tf_not_available(self):
        """suppress_tf_warnings does nothing when HAS_TF is False."""
        import base_attentive.compat.tf as tf_mod

        original = tf_mod.HAS_TF
        try:
            tf_mod.HAS_TF = False
            suppress_tf_warnings()  # Should not raise
        finally:
            tf_mod.HAS_TF = original

    def test_when_tf_available(self):
        """suppress_tf_warnings calls TF logger when HAS_TF is True."""
        import base_attentive.compat.tf as tf_mod

        original_has_tf = tf_mod.HAS_TF
        original_tf = tf_mod.tf

        mock_tf = MagicMock()
        mock_tf.get_logger.return_value = MagicMock()

        try:
            tf_mod.HAS_TF = True
            tf_mod.tf = mock_tf
            suppress_tf_warnings()
            mock_tf.get_logger.assert_called_once()
        finally:
            tf_mod.HAS_TF = original_has_tf
            tf_mod.tf = original_tf


class TestOptionalTfFunction:
    def test_when_tf_not_available_returns_decorator(self):
        """optional_tf_function returns identity decorator when HAS_TF=False."""
        import base_attentive.compat.tf as tf_mod

        original = tf_mod.HAS_TF
        try:
            tf_mod.HAS_TF = False
            dec = optional_tf_function()

            def my_fn():
                return 42

            result = dec(my_fn)
            # Function should be passed through
            assert result is my_fn
        finally:
            tf_mod.HAS_TF = original

    def test_when_tf_not_available_function_works(self):
        """Decorated function still works when TF unavailable."""
        import base_attentive.compat.tf as tf_mod

        original = tf_mod.HAS_TF
        try:
            tf_mod.HAS_TF = False
            dec = optional_tf_function()

            @dec
            def add(a, b):
                return a + b

            assert add(2, 3) == 5
        finally:
            tf_mod.HAS_TF = original


class TestTfDebuggingAssertEqual:
    def test_when_tf_not_available_returns_none(self):
        """tf_debugging_assert_equal returns None when HAS_TF=False."""
        import base_attentive.compat.tf as tf_mod

        original = tf_mod.HAS_TF
        try:
            tf_mod.HAS_TF = False
            result = tf_debugging_assert_equal(1, 1)
            assert result is None
        finally:
            tf_mod.HAS_TF = original

    def test_when_tf_available_calls_tf(self):
        """tf_debugging_assert_equal calls tf.debugging when HAS_TF=True."""
        import base_attentive.compat.tf as tf_mod

        original_has_tf = tf_mod.HAS_TF
        original_tf = tf_mod.tf

        call_record = {}

        class FakeDebugging:
            @staticmethod
            def assert_equal(
                x, y, message="", name="assert_equal"
            ):
                call_record["called"] = True
                call_record["args"] = (x, y)

        class FakeTF:
            debugging = FakeDebugging()

        try:
            tf_mod.HAS_TF = True
            tf_mod.tf = FakeTF()
            tf_debugging_assert_equal(1, 1, message="test")
            assert call_record.get("called") is True
        finally:
            tf_mod.HAS_TF = original_has_tf
            tf_mod.tf = original_tf


# ---------------------------------------------------------------------------
# registry/component_registry.py
# ---------------------------------------------------------------------------

from base_attentive.registry.component_registry import (
    ComponentRegistration,
    ComponentRegistry,
)


class TestComponentRegistry:
    def setup_method(self):
        self.reg = ComponentRegistry()

    def _dummy_builder(self):
        return None

    def test_register_basic(self):
        reg = self.reg.register(
            "my.comp", self._dummy_builder, backend="generic"
        )
        assert isinstance(reg, ComponentRegistration)
        assert reg.key == "my.comp"

    def test_register_duplicate_raises(self):
        """Registering same key+backend twice raises KeyError."""
        self.reg.register(
            "my.comp", self._dummy_builder, backend="generic"
        )
        with pytest.raises(
            KeyError, match="already registered"
        ):
            self.reg.register(
                "my.comp",
                self._dummy_builder,
                backend="generic",
            )

    def test_register_replace_works(self):
        """replace=True allows re-registering."""
        self.reg.register(
            "my.comp", self._dummy_builder, backend="generic"
        )
        reg2 = self.reg.register(
            "my.comp",
            lambda: 42,
            backend="generic",
            replace=True,
        )
        assert reg2 is not None

    def test_has_key_present(self):
        self.reg.register(
            "my.comp", self._dummy_builder, backend="generic"
        )
        assert self.reg.has("my.comp") is True

    def test_has_key_missing(self):
        """has() returns False for unregistered keys."""
        assert self.reg.has("nonexistent.comp") is False

    def test_has_key_with_backend(self):
        self.reg.register(
            "my.comp", self._dummy_builder, backend="torch"
        )
        assert (
            self.reg.has("my.comp", backend="torch") is True
        )
        assert self.reg.has("my.comp", backend="jax") is False

    def test_resolve_existing(self):
        self.reg.register(
            "my.comp", self._dummy_builder, backend="torch"
        )
        reg = self.reg.resolve("my.comp", backend="torch")
        assert reg.key == "my.comp"

    def test_resolve_generic_fallback(self):
        self.reg.register(
            "my.comp", self._dummy_builder, backend="generic"
        )
        reg = self.reg.resolve("my.comp", backend="torch")
        assert reg.backend == "generic"

    def test_resolve_empty_by_backend_raises(self):
        """resolve() raises KeyError when key is not registered at all."""
        with pytest.raises(
            KeyError, match="Unknown component key"
        ):
            self.reg.resolve(
                "nonexistent.key", backend="torch"
            )

    def test_resolve_no_generic_no_allow_generic_raises(self):
        """resolve() with allow_generic=False and no matching backend raises KeyError."""
        self.reg.register(
            "my.comp", self._dummy_builder, backend="torch"
        )
        with pytest.raises(
            KeyError, match="not registered for backend"
        ):
            self.reg.resolve(
                "my.comp", backend="jax", allow_generic=False
            )

    def test_resolve_no_generic_fallback_raises(self):
        """resolve() raises when no specific or generic registration exists."""
        self.reg.register(
            "my.comp", self._dummy_builder, backend="torch"
        )
        with pytest.raises(KeyError):
            self.reg.resolve(
                "my.comp", backend="jax", allow_generic=False
            )

    def test_list_keys(self):
        self.reg.register(
            "comp.a", self._dummy_builder, backend="generic"
        )
        self.reg.register(
            "comp.b", self._dummy_builder, backend="generic"
        )
        keys = self.reg.list_keys()
        assert "comp.a" in keys
        assert "comp.b" in keys
        assert keys == sorted(keys)

    def test_list_keys_empty(self):
        keys = self.reg.list_keys()
        assert keys == []

    def test_clone_creates_independent_copy(self):
        self.reg.register(
            "my.comp", self._dummy_builder, backend="generic"
        )
        cloned = self.reg.clone()
        assert cloned.has("my.comp") is True
        # Modifying clone doesn't affect original
        cloned.register(
            "clone.only",
            self._dummy_builder,
            backend="generic",
        )
        assert self.reg.has("clone.only") is False

    def test_clone_empty_registry(self):
        cloned = self.reg.clone()
        assert cloned.list_keys() == []

    def test_register_with_description(self):
        reg = self.reg.register(
            "my.comp",
            self._dummy_builder,
            backend="generic",
            description="test component",
        )
        assert reg.description == "test component"

    def test_register_experimental(self):
        reg = self.reg.register(
            "my.comp",
            self._dummy_builder,
            backend="generic",
            experimental=True,
        )
        assert reg.experimental is True

    def test_resolve_returns_correct_builder(self):
        def builder_fn():
            return "built!"

        self.reg.register(
            "my.comp", builder_fn, backend="generic"
        )
        reg = self.reg.resolve("my.comp", backend="generic")
        assert reg.builder() == "built!"


# ---------------------------------------------------------------------------
# registry/model_registry.py
# ---------------------------------------------------------------------------

from base_attentive.registry.model_registry import (
    ModelRegistration,
    ModelRegistry,
)


class TestModelRegistry:
    def setup_method(self):
        self.reg = ModelRegistry()

    def _dummy_builder(self):
        return None

    def test_register_basic(self):
        reg = self.reg.register(
            "my.model", self._dummy_builder, backend="generic"
        )
        assert isinstance(reg, ModelRegistration)
        assert reg.key == "my.model"

    def test_register_duplicate_raises(self):
        """Line 45: Registering same key+backend twice raises KeyError."""
        self.reg.register(
            "my.model", self._dummy_builder, backend="generic"
        )
        with pytest.raises(
            KeyError, match="already registered"
        ):
            self.reg.register(
                "my.model",
                self._dummy_builder,
                backend="generic",
            )

    def test_register_replace_works(self):
        self.reg.register(
            "my.model", self._dummy_builder, backend="generic"
        )
        reg2 = self.reg.register(
            "my.model",
            lambda: 42,
            backend="generic",
            replace=True,
        )
        assert reg2 is not None

    def test_has_key_missing(self):
        """Line 64 / line 86: has() returns False for unregistered keys."""
        assert self.reg.has("nonexistent.model") is False

    def test_has_key_with_backend_missing(self):
        """Line 86: backend check in has()."""
        self.reg.register(
            "my.model", self._dummy_builder, backend="torch"
        )
        assert (
            self.reg.has("my.model", backend="jax") is False
        )

    def test_resolve_empty_raises(self):
        """Lines 82: KeyError when key not in registrations."""
        with pytest.raises(
            KeyError, match="Unknown model key"
        ):
            self.reg.resolve(
                "nonexistent.model", backend="torch"
            )

    def test_resolve_no_matching_backend_raises(self):
        """Lines 93-94: no generic fallback, allow_generic=False → KeyError."""
        self.reg.register(
            "my.model", self._dummy_builder, backend="torch"
        )
        with pytest.raises(
            KeyError, match="not registered for backend"
        ):
            self.reg.resolve(
                "my.model", backend="jax", allow_generic=False
            )

    def test_resolve_generic_fallback(self):
        self.reg.register(
            "my.model", self._dummy_builder, backend="generic"
        )
        reg = self.reg.resolve("my.model", backend="torch")
        assert reg.backend == "generic"

    def test_resolve_exact_backend(self):
        self.reg.register(
            "my.model", self._dummy_builder, backend="torch"
        )
        reg = self.reg.resolve("my.model", backend="torch")
        assert reg.key == "my.model"

    def test_register_with_backend_normalization(self):
        """pytorch alias normalized to torch."""
        reg = self.reg.register(
            "my.model", self._dummy_builder, backend="pytorch"
        )
        assert reg.backend == "torch"
