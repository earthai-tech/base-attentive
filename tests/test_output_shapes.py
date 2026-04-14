from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("batch_size", [2])
def test_point_output_shape(
    point_kwargs: dict[str, object],
    batch_size: int,
):
    from base_attentive import BaseAttentive

    model = BaseAttentive(**point_kwargs)
    dyn = np.random.randn(
        batch_size,
        7,
        int(point_kwargs["dynamic_input_dim"]),
    ).astype("float32")
    fut = np.random.randn(
        batch_size,
        int(point_kwargs["forecast_horizon"]),
        int(point_kwargs["future_input_dim"]),
    ).astype("float32")
    sta = np.random.randn(
        batch_size,
        int(point_kwargs["static_input_dim"]),
    ).astype("float32")

    out = model([sta, dyn, fut], training=False)
    assert tuple(out.shape) == (
        batch_size,
        int(point_kwargs["forecast_horizon"]),
        int(point_kwargs["output_dim"]),
    )


@pytest.mark.parametrize("batch_size", [2])
def test_quantile_output_shape(
    quantile_kwargs: dict[str, object],
    batch_size: int,
):
    from base_attentive import BaseAttentive

    model = BaseAttentive(**quantile_kwargs)
    dyn = np.random.randn(
        batch_size,
        7,
        int(quantile_kwargs["dynamic_input_dim"]),
    ).astype("float32")
    fut = np.random.randn(
        batch_size,
        int(quantile_kwargs["forecast_horizon"]),
        int(quantile_kwargs["future_input_dim"]),
    ).astype("float32")
    sta = np.random.randn(
        batch_size,
        int(quantile_kwargs["static_input_dim"]),
    ).astype("float32")

    out = model([sta, dyn, fut], training=False)
    assert tuple(out.shape) == (
        batch_size,
        int(quantile_kwargs["forecast_horizon"]),
        int(quantile_kwargs["output_dim"]),
        len(quantile_kwargs["quantiles"]),
    )


def test_default_residual_path_is_shape_stable(
    sample_dims: dict[str, int],
):
    from base_attentive import BaseAttentive

    model = BaseAttentive(**sample_dims)
    dyn = np.random.randn(
        2,
        7,
        sample_dims["dynamic_input_dim"],
    ).astype("float32")
    fut = np.random.randn(
        2,
        sample_dims["forecast_horizon"],
        sample_dims["future_input_dim"],
    ).astype("float32")
    sta = np.random.randn(
        2,
        sample_dims["static_input_dim"],
    ).astype("float32")
    out = model([sta, dyn, fut], training=False)
    assert out is not None
