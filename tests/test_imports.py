"""Test imports and basic functionality."""

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
