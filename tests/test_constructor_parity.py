from __future__ import annotations

import inspect

EXPECTED_LEGACY_PARAMS = {
    "static_input_dim",
    "dynamic_input_dim",
    "future_input_dim",
    "output_dim",
    "forecast_horizon",
    "mode",
    "num_encoder_layers",
    "quantiles",
    "embed_dim",
    "hidden_units",
    "lstm_units",
    "attention_units",
    "num_heads",
    "dropout_rate",
    "max_window_size",
    "memory_size",
    "scales",
    "multi_scale_agg",
    "final_agg",
    "activation",
    "use_residuals",
    "use_vsn",
    "vsn_units",
    "use_batch_norm",
    "apply_dtw",
    "attention_levels",
    "objective",
    "architecture_config",
    "backend_name",
    "component_overrides",
    "verbose",
    "name",
}


def test_base_attentive_public_signature_has_expected_legacy_surface():
    from base_attentive import BaseAttentive

    params = set(
        inspect.signature(BaseAttentive.__init__).parameters
    )
    missing = EXPECTED_LEGACY_PARAMS - params
    assert not missing, (
        f"Missing legacy parameters: {sorted(missing)}"
    )


def test_v2_accepts_spec_based_construction(
    sample_dims: dict[str, int],
):
    from base_attentive.experimental.base_attentive_v2 import (
        BaseAttentiveV2,
    )

    model = BaseAttentiveV2(**sample_dims)
    assert model.spec.output_dim == sample_dims["output_dim"]
    assert (
        model.spec.forecast_horizon
        == sample_dims["forecast_horizon"]
    )


EXPECTED_MODERN_PARAMS = {
    "static_dim",
    "dynamic_dim",
    "future_dim",
    "lookback_window",
    "attention_stack",
    "output_mode",
    "n_quantiles",
}


def test_base_attentive_public_signature_exposes_modern_surface():
    from base_attentive import BaseAttentive

    params = set(
        inspect.signature(BaseAttentive.__init__).parameters
    )
    missing = EXPECTED_MODERN_PARAMS - params
    assert not missing, (
        f"Missing modern parameters: {sorted(missing)}"
    )
