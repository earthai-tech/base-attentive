from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.usefixtures("configured_runtime_backend")

from base_attentive.compat.versioning import (
    BASE_ATTENTIVE_PARAMETER_RULES,
    DeprecatedParameterWarning,
    UnsupportedCompatibilityWarning,
    n_quantiles_to_quantiles,
    resolve_deprecated_kwargs,
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


def test_resolve_legacy_dimension_aliases_to_modern_names():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_deprecated_kwargs(
            {
                "static_input_dim": 2,
                "dynamic_input_dim": 3,
                "future_input_dim": 4,
            },
            BASE_ATTENTIVE_PARAMETER_RULES,
        )

    assert result["static_dim"] == 2
    assert result["dynamic_dim"] == 3
    assert result["future_dim"] == 4
    assert any(
        isinstance(item.message, DeprecatedParameterWarning)
        for item in caught
    )


def test_new_parameter_names_take_precedence_over_deprecated_aliases():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_deprecated_kwargs(
            {
                "dynamic_dim": 10,
                "dynamic_input_dim": 3,
                "lookback_window": 12,
                "max_window_size": 5,
            },
            BASE_ATTENTIVE_PARAMETER_RULES,
        )

    assert result["dynamic_dim"] == 10
    assert result["lookback_window"] == 12
    assert len(caught) == 2


def test_n_quantiles_helper_generates_evenly_spaced_quantiles():
    assert n_quantiles_to_quantiles(1) == (0.5,)
    assert n_quantiles_to_quantiles(3) == (0.25, 0.5, 0.75)


def test_base_attentive_accepts_modern_parameter_names(
    patched_v2_init,
):
    model = BaseAttentive(
        static_dim=2,
        dynamic_dim=3,
        future_dim=4,
        lookback_window=12,
        attention_stack=["cross", "memory"],
        n_quantiles=3,
    )

    assert model.static_dim == 2
    assert model.dynamic_dim == 3
    assert model.future_dim == 4
    assert model.lookback_window == 12
    assert model.max_window_size == 12
    assert model.quantiles == [0.25, 0.5, 0.75]
    assert tuple(model.spec.attention_levels) == (
        "cross",
        "memory",
    )


def test_output_mode_gaussian_warns_but_does_not_break(
    patched_v2_init,
):
    with pytest.warns(UnsupportedCompatibilityWarning):
        model = BaseAttentive(
            static_dim=1,
            dynamic_dim=2,
            future_dim=3,
            output_mode="gaussian",
        )

    assert model.output_mode == "gaussian"
