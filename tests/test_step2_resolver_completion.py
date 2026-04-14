"""Regression tests for step 2 resolver completion."""

from __future__ import annotations

from base_attentive.experimental import BaseAttentiveV2
from base_attentive.implementations.generic.base_attentive_v2 import (
    _GenericConcatFusion,
    _GenericFlattenPool,
    _GenericLastPool,
    _GenericMeanPool,
)
from base_attentive.implementations.jax import (
    ensure_jax_v2_registered,
)
from base_attentive.implementations.torch import (
    ensure_torch_v2_registered,
)
from base_attentive.registry import ComponentRegistry


def test_generic_pool_and_fusion_builders_return_serializable_layers():
    mean_pool = _GenericMeanPool(
        axis=1, keepdims=False, name="mean"
    )
    last_pool = _GenericLastPool(axis=1, name="last")
    flatten_pool = _GenericFlattenPool(axis=1, name="flatten")
    fusion = _GenericConcatFusion(axis=-1, name="fusion")
    assert mean_pool.get_config()["axis"] == 1
    assert last_pool.get_config()["axis"] == 1
    assert flatten_pool.get_config()["axis"] == 1
    assert fusion.get_config()["axis"] == -1


def test_torch_registration_covers_extended_resolver_surface():
    registry = ComponentRegistry()
    ensure_torch_v2_registered(registry=registry)
    keys = [
        "feature.static_processor",
        "embedding.positional",
        "encoder.dynamic_window",
        "decoder.cross_attention",
        "decoder.hierarchical_attention",
        "decoder.memory_attention",
        "fusion.multi_resolution_attention",
        "pool.final_flatten",
        "head.multi_horizon",
        "head.quantile_distribution",
    ]
    for key in keys:
        assert (
            registry.resolve(key, backend="torch").backend
            == "torch"
        )


def test_jax_registration_covers_extended_resolver_surface():
    registry = ComponentRegistry()
    ensure_jax_v2_registered(registry=registry)
    keys = [
        "feature.static_processor",
        "embedding.positional",
        "encoder.dynamic_window",
        "decoder.cross_attention",
        "decoder.hierarchical_attention",
        "decoder.memory_attention",
        "fusion.multi_resolution_attention",
        "pool.final_flatten",
        "head.multi_horizon",
        "head.quantile_distribution",
    ]
    for key in keys:
        assert (
            registry.resolve(key, backend="jax").backend
            == "jax"
        )


def test_v2_tracks_resolved_components_as_model_attributes():
    model = BaseAttentiveV2(
        static_input_dim=2,
        dynamic_input_dim=3,
        future_input_dim=2,
        output_dim=1,
        forecast_horizon=2,
    )
    assert hasattr(model, "dynamic_projection")
    assert hasattr(model, "decoder_cross_attention")
    assert hasattr(model, "multi_horizon_head")
    assert (
        "dynamic_projection" in model._tracked_component_names
    )
