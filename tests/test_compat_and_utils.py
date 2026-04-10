"""Test compatibility layers."""

from __future__ import annotations

import pytest


class TestCompatModule:
    """Test compatibility layer."""

    def test_interval_validator(self):
        """Test Interval validator."""
        from base_attentive.compat import Interval

        interval = Interval(int, left=0, right=100, closed="left")
        assert interval is not None
        assert hasattr(interval, "left")
        assert hasattr(interval, "right")

    def test_str_options_validator(self):
        """Test StrOptions validator."""
        from base_attentive.compat import StrOptions

        options = StrOptions({"relu", "sigmoid", "tanh"})
        assert options is not None
        assert hasattr(options, "options")

    def test_validate_params_decorator(self):
        """Test validate_params decorator."""
        from base_attentive.compat import validate_params

        # Decorator should return a callable
        assert callable(validate_params)

        # Test applying it to a function
        @validate_params({})
        def sample_func(x):
            return x

        assert callable(sample_func)


class TestLoggingModule:
    """Test logging utilities."""

    def test_get_logger(self):
        """Test get_logger returns logger."""
        from base_attentive.logging import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_once_per_message_filter(self):
        """Test OncePerMessageFilter."""
        import logging
        from base_attentive.logging import OncePerMessageFilter

        filter_obj = OncePerMessageFilter()
        assert filter_obj is not None

        # Create a dummy log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # First occurrence should pass
        assert filter_obj.filter(record) is True

        # Second occurrence should be filtered
        assert filter_obj.filter(record) is False


class TestAPIMod:
    """Test API module."""

    def test_nnlearner_class(self):
        """Test NNLearner base class."""
        from base_attentive.api import NNLearner

        class TestModel(NNLearner):
            def __init__(self, a=1, b=2):
                self.a = a
                self.b = b

        model = TestModel(a=5, b=10)
        assert model.a == 5
        assert model.b == 10

    def test_nnlearner_get_params(self):
        """Test NNLearner.get_params method."""
        from base_attentive.api import NNLearner

        class TestModel(NNLearner):
            def __init__(self, a=1, b=2):
                self.a = a
                self.b = b

        model = TestModel(a=5, b=10)
        params = model.get_params()
        assert params["a"] == 5
        assert params["b"] == 10

    def test_nnlearner_set_params(self):
        """Test NNLearner.set_params method."""
        from base_attentive.api import NNLearner

        class TestModel(NNLearner):
            def __init__(self, a=1, b=2):
                self.a = a
                self.b = b

        model = TestModel(a=5, b=10)
        model.set_params(a=20, b=30)
        assert model.a == 20
        assert model.b == 30


class TestModelUtils:
    """Test model utility functions."""

    def test_resolve_attention_levels(self):
        """Test resolve_attention_levels function."""
        from base_attentive.models.comp_utils import resolve_attention_levels

        config = resolve_attention_levels()
        assert isinstance(config, dict)
        assert "decoder_attention_stack" in config

    def test_resolve_attention_levels_custom(self):
        """Test resolve_attention_levels with custom config."""
        from base_attentive.models.comp_utils import resolve_attention_levels

        custom = {"decoder_attention_stack": ["custom"], "new_key": "value"}
        config = resolve_attention_levels(custom)
        assert config["decoder_attention_stack"] == ["custom"]
        assert config["new_key"] == "value"

    def test_set_default_params(self):
        """Test set_default_params function."""
        from base_attentive.models.utils import set_default_params

        defaults = {"a": 1, "b": 2}
        user_params = {"a": 10}
        result = set_default_params(user_params, **defaults)
        assert result["a"] == 10  # User value wins
        assert result["b"] == 2  # Default remains


class TestGenericeUtils:
    """Test generic utility functions."""

    def test_select_mode_auto(self):
        """Test select_mode in auto mode."""
        from base_attentive.utils.generic_utils import select_mode

        data = {"test": "value"}
        result = select_mode(data, mode="auto")
        assert result == data
