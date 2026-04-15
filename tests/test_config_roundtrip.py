from __future__ import annotations


def test_base_attentive_get_config_roundtrip(
    point_kwargs: dict[str, object],
):
    from base_attentive import BaseAttentive

    model = BaseAttentive(**point_kwargs)
    config = model.get_config()
    clone = BaseAttentive.from_config(config)
    clone_config = clone.get_config()

    assert clone_config["output_dim"] == config["output_dim"]
    assert (
        clone_config["forecast_horizon"]
        == config["forecast_horizon"]
    )
    assert (
        clone_config["hidden_units"] == config["hidden_units"]
    )
    assert (
        clone_config["use_residuals"]
        == config["use_residuals"]
    )


def test_v2_get_config_roundtrip(
    sample_dims: dict[str, int],
):
    from base_attentive.experimental.base_attentive_v2 import (
        BaseAttentiveV2,
    )

    model = BaseAttentiveV2(
        **sample_dims,
        quantiles=(0.1, 0.9),
    )
    config = model.get_config()
    clone = BaseAttentiveV2.from_config(config)
    assert clone.spec.output_dim == model.spec.output_dim
    assert (
        clone.spec.forecast_horizon
        == model.spec.forecast_horizon
    )
    assert tuple(clone.spec.quantiles) == tuple(
        model.spec.quantiles
    )
