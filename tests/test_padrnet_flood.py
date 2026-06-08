from __future__ import annotations

import importlib.util
import subprocess
import sys

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


def _can_import_tensorflow() -> bool:
    if importlib.util.find_spec("tensorflow") is None:
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tensorflow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_padrnet_config_validation_and_update():
    config = PADRNetConfig(input_dim=3, forecast_horizon=2)
    assert config.input_dim == 3
    assert config.with_updates(hidden_dim=32).hidden_dim == 32
    assert config.with_updates(static_dim=2).static_dim == 2

    with pytest.raises(ValueError):
        PADRNetConfig(input_dim=0)
    with pytest.raises(ValueError):
        PADRNetConfig(input_dim=3, dropout=1.0)
    with pytest.raises(ValueError):
        PADRNetConfig(
            input_dim=3, hidden_dim=10, num_heads=4
        )
    with pytest.raises(ValueError):
        config.with_updates(flood_threshold=0.0)


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


def test_flood_metrics_degenerate_cases():
    zeros = np.zeros(4)
    no_events = np.array([0.0, 0.01, 0.02])
    all_events = np.array([0.1, 0.2, 0.3])

    assert nash_sutcliffe_efficiency(zeros, zeros) == 0.0
    assert delta_mass(zeros, zeros) == 0.0
    assert critical_success_index(
        no_events, no_events, threshold=0.05
    ) == 0.0
    assert true_skill_statistic(
        all_events, all_events, threshold=0.05
    ) == pytest.approx(1.0)


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


def test_flood_physics_helper_error_paths():
    precip = np.ones((2, 4))
    depth = np.ones((2, 4))

    assert linear_reservoir_response([]).shape == (0,)

    with pytest.raises(ValueError):
        linear_reservoir_response(precip, tau=0.0)
    with pytest.raises(ValueError):
        linear_reservoir_response(precip, gain=-0.1)
    with pytest.raises(ValueError):
        mass_balance_residual(precip, depth[:, :3])
    with pytest.raises(ValueError):
        mass_balance_residual(precip, depth, tau=0.0)
    with pytest.raises(ValueError):
        exceedance_probability(depth, threshold=0.0)
    with pytest.raises(ValueError):
        exceedance_probability(
            depth, threshold=0.1, scale=0.0
        )


def test_padrnet_torch_without_static_inputs():
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        pytest.skip("PyTorch is not installed.")

    import torch

    config = PADRNetConfig(
        input_dim=3,
        hidden_dim=12,
        num_heads=3,
        forecast_horizon=2,
        flood_threshold=0.1,
    )
    model = PADRNet(config, backend="pytorch")
    outputs = model(torch.ones(2, 5, 3))

    assert isinstance(model, NNLearner)
    assert model.get_params(deep=False)["name"] == "padr_net"
    assert outputs["depth"].shape == (2, 2, 1)
    assert torch.all(outputs["depth"] >= 0)
    assert torch.all(
        outputs["exceedance_probability"] >= 0
    )
    assert torch.all(
        outputs["exceedance_probability"] <= 1
    )


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


def test_padrnet_tensorflow_output_shapes():
    if not _can_import_tensorflow():
        pytest.skip("TensorFlow is not importable.")

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
    assert np.all(outputs["depth"].numpy() >= 0.0)

    cloned = type(model).from_config(model.get_config())
    cloned_outputs = cloned(
        tf.zeros((1, 12, 4)), tf.zeros((1, 2))
    )
    assert isinstance(cloned, NNLearner)
    assert tuple(cloned_outputs["depth"].shape) == (1, 3, 1)
