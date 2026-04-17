"""Facade-oriented tests replacing the removed legacy layer internals."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("configured_runtime_backend")

from base_attentive.config.legacy_adapter import (
    normalize_legacy_architecture_spec,
)
from base_attentive.core.base_attentive import BaseAttentive
from base_attentive.experimental.base_attentive_v2 import (
    BaseAttentiveV2,
)


@pytest.fixture
def patched_v2_init(monkeypatch):
    def fake_init(self, **kwargs):
        self.spec = kwargs["spec"]
        self.name = kwargs.get("name")
        self._v2_init_kwargs = kwargs

    monkeypatch.setattr(
        BaseAttentiveV2, "__init__", fake_init
    )


def test_normalize_legacy_architecture_spec_disables_vsn_for_dense_paths():
    spec = normalize_legacy_architecture_spec(
        objective="transformer",
        use_vsn=False,
        attention_levels=["cross", "memory"],
        architecture_config={
            "feature_processing": "vsn",
            "encoder_type": "transformer",
        },
    )
    assert spec.encoder_type == "transformer"
    assert spec.feature_processing == "dense"
    assert spec.decoder_attention_stack == ("cross", "memory")


def test_base_attentive_facade_preserves_legacy_arguments(
    patched_v2_init,
):
    model = BaseAttentive(
        static_input_dim=0,
        dynamic_input_dim=3,
        future_input_dim=2,
        forecast_horizon=2,
        objective="transformer",
        use_vsn=False,
        use_residuals=False,
        num_encoder_layers=2,
        architecture_config={"encoder_type": "transformer"},
    )

    assert model.use_vsn is False
    assert model.use_residuals is False
    assert model.num_encoder_layers == 2
    assert (
        model.spec.architecture.encoder_type == "transformer"
    )
    assert (
        model.spec.architecture.feature_processing == "dense"
    )


def test_base_attentive_facade_quantile_head_type_and_horizon(
    patched_v2_init,
):
    model = BaseAttentive(
        static_input_dim=1,
        dynamic_input_dim=1,
        future_input_dim=1,
        forecast_horizon=4,
        quantiles=[0.1, 0.5, 0.9],
    )

    assert model.spec.head_type == "quantile"
    assert model.spec.forecast_horizon == 4
    assert model.spec.quantiles == (0.1, 0.5, 0.9)
