"""Configuration and test utilities."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Add src to path for imports
SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def sample_inputs():
    """Fixture providing sample model inputs."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not installed")
    
    batch_size = 32
    static_dim = 4
    dynamic_dim = 8
    future_dim = 6
    time_steps = 10
    forecast_horizon = 24
    
    static = tf.random.normal([batch_size, static_dim])
    dynamic = tf.random.normal([batch_size, time_steps, dynamic_dim])
    future = tf.random.normal([batch_size, forecast_horizon, future_dim])
    
    return (static, dynamic, future)


@pytest.fixture
def base_attentive_module():
    """Fixture to import base_attentive module."""
    import base_attentive
    return base_attentive


@pytest.fixture
def backend_module():
    """Fixture to import backend module."""
    from base_attentive.backend import (
        get_backend,
        set_backend,
        get_available_backends,
    )
    return {
        "get_backend": get_backend,
        "set_backend": set_backend,
        "get_available_backends": get_available_backends,
    }


@pytest.fixture
def validation_module():
    """Fixture to import validation module."""
    from base_attentive.validation import (
        validate_model_inputs,
        maybe_reduce_quantiles_bh,
        ensure_bh1,
    )
    return {
        "validate_model_inputs": validate_model_inputs,
        "maybe_reduce_quantiles_bh": maybe_reduce_quantiles_bh,
        "ensure_bh1": ensure_bh1,
    }


@pytest.fixture
def compat_module():
    """Fixture to import compat layer."""
    from base_attentive.compat import (
        Interval,
        StrOptions,
        validate_params,
    )
    return {
        "Interval": Interval,
        "StrOptions": StrOptions,
        "validate_params": validate_params,
    }


@pytest.fixture
def logging_module():
    """Fixture to import logging module."""
    from base_attentive.logging import get_logger, OncePerMessageFilter
    return {
        "get_logger": get_logger,
        "OncePerMessageFilter": OncePerMessageFilter,
    }


@pytest.fixture
def api_module():
    """Fixture to import api.property module."""
    from base_attentive.api import NNLearner
    return {"NNLearner": NNLearner}
