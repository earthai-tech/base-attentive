"""Tests for the lightweight core utilities."""

from __future__ import annotations

import importlib
import warnings

import pytest


def _import_core_submodule(module_name: str):
    """Import a core submodule while suppressing expected runtime warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return importlib.import_module(module_name)


def test_core_utility_modules_are_importable():
    """Core utility submodules should remain importable independently."""
    checks = _import_core_submodule("base_attentive.core.checks")
    handlers = _import_core_submodule("base_attentive.core.handlers")

    assert callable(checks.validate_nested_param)
    assert callable(handlers.param_deprecated_message)
    assert callable(handlers.delegate_on_error)


class TestValidateNestedParam:
    """Test nested parameter validation helpers."""

    def test_accepts_valid_list_element_types(self):
        """Lists with the expected element type should be returned unchanged."""
        checks = _import_core_submodule("base_attentive.core.checks")
        value = [1, 2, 3]

        assert (
            checks.validate_nested_param(value, list[int], "levels")
            == value
        )

    def test_rejects_non_list_for_list_type(self):
        """A non-list value should raise a clear type error."""
        checks = _import_core_submodule("base_attentive.core.checks")

        with pytest.raises(TypeError, match="levels must be a list"):
            checks.validate_nested_param(
                "not-a-list", list[int], "levels"
            )

    def test_rejects_wrong_list_element_type(self):
        """The failing element index should be included in the error."""
        checks = _import_core_submodule("base_attentive.core.checks")

        with pytest.raises(
            TypeError, match=r"levels\[1\] must be int"
        ):
            checks.validate_nested_param(
                [1, "two", 3], list[int], "levels"
            )

    def test_rejects_wrong_scalar_type(self):
        """Scalar type mismatches should mention the parameter name."""
        checks = _import_core_submodule("base_attentive.core.checks")

        with pytest.raises(
            TypeError, match="forecast_horizon must be int"
        ):
            checks.validate_nested_param(
                "24", int, "forecast_horizon"
            )


class TestParamDeprecatedMessage:
    """Test deprecation warning helpers."""

    def test_warns_for_deprecated_function_keyword(self):
        """A matching deprecated keyword should emit the configured warning."""
        handlers = _import_core_submodule(
            "base_attentive.core.handlers"
        )

        @handlers.param_deprecated_message(
            [
                {
                    "param": "legacy_mode",
                    "condition": bool,
                    "message": "legacy_mode is deprecated",
                }
            ],
            warning_category=FutureWarning,
        )
        def sample_function(*, legacy_mode=False):
            return legacy_mode

        with pytest.warns(
            FutureWarning, match="legacy_mode is deprecated"
        ):
            assert sample_function(legacy_mode=True) is True

    def test_warns_for_deprecated_class_keyword(self):
        """Class decorators should apply the same warning behavior to __init__."""
        handlers = _import_core_submodule(
            "base_attentive.core.handlers"
        )

        @handlers.param_deprecated_message(
            [
                {
                    "param": "legacy_mode",
                    "condition": bool,
                    "message": "legacy_mode is deprecated",
                }
            ],
            warning_category=FutureWarning,
        )
        class Example:
            def __init__(self, *, legacy_mode=False):
                self.legacy_mode = legacy_mode

        with pytest.warns(
            FutureWarning, match="legacy_mode is deprecated"
        ):
            instance = Example(legacy_mode=True)

        assert instance.legacy_mode is True

    def test_skips_warning_when_condition_is_not_met(self):
        """No warning should be emitted when the condition evaluates false."""
        handlers = _import_core_submodule(
            "base_attentive.core.handlers"
        )

        @handlers.param_deprecated_message(
            [
                {
                    "param": "legacy_mode",
                    "condition": bool,
                    "message": "legacy_mode is deprecated",
                }
            ],
            warning_category=FutureWarning,
        )
        def sample_function(*, legacy_mode=False):
            return legacy_mode

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert sample_function(legacy_mode=False) is False

        assert caught == []


class TestDelegateOnError:
    """Test graceful error delegation behavior."""

    def test_delegates_exception_to_handler(self):
        """Exceptions should be converted by the provided handler."""
        handlers = _import_core_submodule(
            "base_attentive.core.handlers"
        )

        @handlers.delegate_on_error(
            lambda error: f"handled {type(error).__name__}: {error}"
        )
        def explode():
            raise RuntimeError("boom")

        assert explode() == "handled RuntimeError: boom"

    def test_reraises_when_no_handler_is_provided(self):
        """Without a handler, the original exception should propagate."""
        handlers = _import_core_submodule(
            "base_attentive.core.handlers"
        )

        @handlers.delegate_on_error()
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()

    def test_returns_success_value_without_using_handler(self):
        """Successful calls should pass through untouched."""
        handlers = _import_core_submodule(
            "base_attentive.core.handlers"
        )

        @handlers.delegate_on_error(lambda error: "unused")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
