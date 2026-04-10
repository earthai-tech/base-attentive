"""Configuration and test utilities."""

import pytest


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
