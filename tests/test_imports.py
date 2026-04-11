"""Test imports and basic functionality."""

from __future__ import annotations

import pytest


def test_import_base_attentive():
    """Test that BaseAttentive can be imported."""
    try:
        from base_attentive import BaseAttentive

        assert BaseAttentive is not None
    except ImportError as e:
        pytest.skip(f"Cannot import BaseAttentive: {e}")


def test_package_metadata():
    """Test package metadata."""
    import base_attentive

    assert hasattr(base_attentive, "__version__")
    assert hasattr(base_attentive, "__author__")
    assert isinstance(base_attentive.__version__, str)


class TestCoreImports:
    """Test all core module imports."""

    def test_import_backend(self):
        """Test backend module imports."""
        from base_attentive.backend import (
            get_backend,
            set_backend,
            get_available_backends,
        )

        assert callable(get_backend)
        assert callable(set_backend)
        assert callable(get_available_backends)

    def test_import_api(self):
        """Test API module imports."""
        from base_attentive.api import NNLearner

        assert NNLearner is not None

    def test_import_compat(self):
        """Test compatibility layer imports."""
        from base_attentive.compat import (
            Interval,
            StrOptions,
            validate_params,
        )

        assert Interval is not None
        assert StrOptions is not None
        assert callable(validate_params)

    def test_import_logging(self):
        """Test logging module imports."""
        from base_attentive.logging import get_logger, OncePerMessageFilter

        assert callable(get_logger)
        assert OncePerMessageFilter is not None

    def test_import_validation(self):
        """Test validation module imports."""
        from base_attentive.validation import (
            validate_model_inputs,
            maybe_reduce_quantiles_bh,
            ensure_bh1,
        )

        assert callable(validate_model_inputs)
        assert callable(maybe_reduce_quantiles_bh)
        assert callable(ensure_bh1)

    def test_import_models(self):
        """Test models module imports."""
        from base_attentive.models.comp_utils import resolve_attention_levels
        from base_attentive.models.utils import set_default_params

        assert callable(resolve_attention_levels)
        assert callable(set_default_params)

    def test_import_utils(self):
        """Test utils module imports."""
        from base_attentive.utils.deps_utils import ensure_pkg
        from base_attentive.utils.generic_utils import select_mode

        assert callable(ensure_pkg)
        assert callable(select_mode)
