from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from base_attentive.api.property import NNLearner
from base_attentive.applications.flood import (
    PADRNet,
    PADRNetConfig,
    critical_success_index,
    delta_mass,
    exceedance_probability,
    linear_reservoir_response,
    mass_balance_residual,
    nash_sutcliffe_efficiency,
    true_skill_statistic,
)


def test_padrnet_config_validation_and_update():
    config = PADRNetConfig(input_dim=3, forecast_horizon=2)
    assert config.input_dim == 3
    assert config.with_updates(hidden_dim=32).hidden_dim == 32

    with pytest.raises(ValueError):
        PADRNetConfig(input_dim=0)
    with pytest.raises(ValueError):
        PADRNetConfig(input_dim=3, dropout=1.0)
    with pytest.raises(ValueError):
        PADRNetConfig(
            input_dim=3, hidden_dim=10, num_heads=4
        )


def test_padrnet_factory_validates_params():
    config = PADRNetConfig(input_dim=3)
    with pytest.raises(ValueError):
        PADRNet(config, backend="mxnet")
    with pytest.raises(ValueError):
        PADRNet("bad-config", backend="torch")


def test_padrnet_factory_class_inherits_nnlearner():
    assert issubclass(PADRNet, NNLearner)


def test_flood_metrics_basic_values():
    y_true = np.array([0.0, 0.1, 0.2, 0.0])
    y_pred = np.array([0.0, 0.1, 0.15, 0.06])

    assert nash_sutcliffe_efficiency(
        y_true, y_true
    ) == pytest.approx(1.0)
    assert delta_mass(y_true, y_true) == pytest.approx(0.0)
    assert critical_success_index(
        y_true, y_pred, threshold=0.05
    ) == pytest.approx(2 / 3)
    assert true_skill_statistic(
        y_true, y_pred, threshold=0.05
    ) == pytest.approx(0.5)


def test_flood_physics_helpers_shape_and_range():
    precip = np.zeros((2, 8))
    precip[:, 2:4] = 5.0
    depth = linear_reservoir_response(
        precip, tau=3.0, gain=0.02
    )
    residual = mass_balance_residual(precip, depth, tau=3.0)
    probability = exceedance_probability(
        depth, threshold=0.01
    )

    assert depth.shape == precip.shape
    assert residual.shape == precip.shape
    assert probability.shape == precip.shape
    assert np.all(depth >= 0)
    assert np.all((probability >= 0) & (probability <= 1))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is not installed.",
)
def test_padrnet_torch_output_shapes():
    import torch

    config = PADRNetConfig(
        input_dim=4,
        static_dim=2,
        hidden_dim=16,
        num_heads=4,
        forecast_horizon=3,
    )
    model = PADRNet(config, backend="torch")
    outputs = model(torch.zeros(2, 12, 4), torch.zeros(2, 2))

    assert isinstance(model, NNLearner)
    assert model.get_params(deep=False)["config"] is config
    assert outputs["depth"].shape == (2, 3, 1)
    assert outputs[
        "exceedance_probability"
    ].shape == (2, 3, 1)
    assert outputs["features"].shape == (2, 16)


@pytest.mark.skipif(
    os.environ.get(
        "BASE_ATTENTIVE_RUN_TENSORFLOW_TESTS"
    ) != "1",
    reason="TensorFlow PADR-Net smoke test is opt-in.",
)
def test_padrnet_tensorflow_output_shapes():
    import tensorflow as tf

    config = PADRNetConfig(
        input_dim=4,
        static_dim=2,
        hidden_dim=16,
        num_heads=4,
        forecast_horizon=3,
    )
    model = PADRNet(config, backend="tensorflow")
    outputs = model(tf.zeros((2, 12, 4)), tf.zeros((2, 2)))

    assert isinstance(model, NNLearner)
    assert model.get_params(deep=False)["config"] is config
    assert tuple(outputs["depth"].shape) == (2, 3, 1)
    assert tuple(
        outputs["exceedance_probability"].shape
    ) == (2, 3, 1)
    assert tuple(outputs["features"].shape) == (2, 16)
