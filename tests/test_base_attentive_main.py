"""Tests for the public BaseAttentive facade under the V2 architecture."""

from __future__ import annotations

import copy

import pytest

from base_attentive.config.legacy_adapter import legacy_base_attentive_to_spec
from base_attentive.core.base_attentive import BaseAttentive
from base_attentive.experimental.base_attentive_v2 import BaseAttentiveV2


@pytest.fixture
def patched_v2_init(monkeypatch):
    def fake_init(self, **kwargs):
        self.spec = kwargs["spec"]
        self.name = kwargs.get("name")
        self._v2_init_kwargs = kwargs

    monkeypatch.setattr(BaseAttentiveV2, "__init__", fake_init)


def test_base_attentive_initializes_with_legacy_public_attributes(patched_v2_init):
    model = BaseAttentive(
        static_input_dim=2,
        dynamic_input_dim=3,
        future_input_dim=4,
        forecast_horizon=5,
        objective="transformer",
        use_vsn=False,
        attention_levels=["cross", "memory"],
        quantiles=[0.1, 0.5, 0.9],
        architecture_config={"feature_processing": "vsn", "encoder_type": "transformer"},
    )

    assert model.static_input_dim == 2
    assert model.dynamic_input_dim == 3
    assert model.future_input_dim == 4
    assert model.forecast_horizon == 5
    assert model.quantiles == [0.1, 0.5, 0.9]
    assert model.spec.architecture.encoder_type == "transformer"
    assert model.spec.architecture.feature_processing == "dense"
    assert model.spec.attention_levels == ("cross", "memory")


def test_base_attentive_docstring_mentions_v2_facade():
    docstring = BaseAttentive.__doc__
    assert docstring
    assert "resolver-driven V2 model" in docstring


def test_get_config_and_from_config_round_trip_without_mutating_input(patched_v2_init):
    model = BaseAttentive(
        static_input_dim=2,
        dynamic_input_dim=3,
        future_input_dim=1,
        quantiles=[0.1, 0.9],
        architecture_config={"encoder_type": "transformer"},
        component_overrides={"point_head": "head.multi_horizon"},
    )

    config = model.get_config()
    original = copy.deepcopy(config)
    rebuilt = BaseAttentive.from_config(config)

    assert config == original
    assert rebuilt.get_config() == original
    assert rebuilt.quantiles == [0.1, 0.9]


def test_legacy_adapter_matches_facade_spec(patched_v2_init):
    payload = dict(
        static_input_dim=1,
        dynamic_input_dim=2,
        future_input_dim=3,
        forecast_horizon=4,
        objective="hybrid",
        use_vsn=True,
        attention_levels=["cross", "hierarchical"],
    )
    expected = legacy_base_attentive_to_spec(**payload)
    model = BaseAttentive(**payload)
    assert model.spec == expected
