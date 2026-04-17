from __future__ import annotations

import pytest

from base_attentive.config.legacy_adapter import (
    legacy_base_attentive_to_spec,
    normalize_legacy_runtime_spec,
)


def test_legacy_adapter_normalizes_point_mode_to_point_head():
    spec = legacy_base_attentive_to_spec(
        static_input_dim=1,
        dynamic_input_dim=2,
        future_input_dim=0,
        mode="point",
    )
    assert spec.head_type == "point"
    assert spec.runtime.mode is None
    assert spec.extras["legacy_mode"] == "point"


def test_legacy_adapter_normalizes_quantile_mode_when_quantiles_present():
    spec = legacy_base_attentive_to_spec(
        static_input_dim=1,
        dynamic_input_dim=2,
        future_input_dim=0,
        mode="quantile",
        quantiles=(0.1, 0.5, 0.9),
    )
    assert spec.head_type == "quantile"
    assert spec.runtime.mode is None
    assert spec.quantiles == (0.1, 0.5, 0.9)


def test_legacy_adapter_rejects_quantile_mode_without_quantiles():
    with pytest.raises(ValueError, match="requires quantiles"):
        legacy_base_attentive_to_spec(
            static_input_dim=1,
            dynamic_input_dim=2,
            future_input_dim=0,
            mode="quantile",
        )


def test_legacy_adapter_rejects_point_mode_with_quantiles():
    with pytest.raises(ValueError, match="conflicts with quantiles"):
        legacy_base_attentive_to_spec(
            static_input_dim=1,
            dynamic_input_dim=2,
            future_input_dim=0,
            mode="point",
            quantiles=(0.1, 0.9),
        )


def test_legacy_adapter_validate_params_guards_bad_dropout_rate():
    with pytest.raises(ValueError):
        legacy_base_attentive_to_spec(
            static_input_dim=1,
            dynamic_input_dim=2,
            future_input_dim=0,
            dropout_rate=1.5,
        )


def test_legacy_adapter_validate_params_guards_bad_quantile_container_type():
    with pytest.raises(TypeError):
        legacy_base_attentive_to_spec(
            static_input_dim=1,
            dynamic_input_dim=2,
            future_input_dim=0,
            quantiles=0.5,
        )


def test_normalize_legacy_runtime_spec_canonicalizes_hyphenated_mode_and_auto_scales():
    runtime = normalize_legacy_runtime_spec(
        mode="tft_like",
        scales="auto",
    )
    assert runtime.mode == "tft_like"
    assert runtime.scales == "auto"
