# SPDX-License-Identifier: Apache-2.0
"""Tests for pure Python components (no Keras needed).

NOTE: We must apply the Keras patch before importing base_attentive.components
because the components __init__.py imports all submodules including Keras ones.
The conftest.py already adds 'src' to sys.path, so we do NOT re-add it here.
"""

import os
import warnings

import pytest

# Patch missing KERAS_DEPS ops before importing anything from components
import base_attentive as _ba

_orig_ga = _ba._KerasDeps.__getattr__

# Stub class used as a safe base-class fallback when real Keras classes are unavailable.
_KerasStub = type(
    "_KerasStub",
    (object,),
    {"__init__": lambda self, *a, **kw: None},
)

_FALLBACKS = {
    "add_n": lambda tensors, **kw: sum(tensors) if isinstance(tensors, (list, tuple)) else tensors,
    "gather": lambda p, i, axis=None, **kw: p,
    "reduce_logsumexp": lambda x, axis=None, keepdims=False, **kw: x,
    "pow": lambda x, y, **kw: x,
    "rank": lambda x, **kw: len(getattr(x, "shape", [])),
    # Keras class stubs — must be real classes so they can be used as base classes.
    "Loss": _KerasStub,
    "Layer": _KerasStub,
    "Model": _KerasStub,
    # Decorator factory stub — must return a callable that accepts a class.
    "register_keras_serializable": lambda package="Custom", name=None: (lambda cls: cls),
}


def _patched_ga(self, name):
    try:
        return _orig_ga(self, name)
    except (ImportError, AttributeError):
        val = _FALLBACKS.get(name, _KerasStub)
        self._cache[name] = val
        return val


_ba._KerasDeps.__getattr__ = _patched_ga

# Now safe to import components
from base_attentive.components.utils import (
    configure_architecture,
    resolve_attn_levels,
    resolve_fusion_mode,
)
from base_attentive.config.schema import BaseAttentiveSpec
from base_attentive.config.validate import validate_base_attentive_spec


# ---------------------------------------------------------------------------
# resolve_attn_levels
# ---------------------------------------------------------------------------


class TestResolveAttnLevels:
    def test_none_returns_all(self):
        result = resolve_attn_levels(None)
        assert result == ["cross", "hierarchical", "memory"]

    def test_use_all_returns_all(self):
        result = resolve_attn_levels("use_all")
        assert result == ["cross", "hierarchical", "memory"]

    def test_star_returns_all(self):
        result = resolve_attn_levels("*")
        assert result == ["cross", "hierarchical", "memory"]

    def test_cross_string(self):
        assert resolve_attn_levels("cross") == ["cross"]

    def test_cross_att_alias(self):
        assert resolve_attn_levels("cross_att") == ["cross"]

    def test_cross_attention_alias(self):
        assert resolve_attn_levels("cross_attention") == ["cross"]

    def test_hier_att_string(self):
        assert resolve_attn_levels("hier_att") == ["hierarchical"]

    def test_hierarchical_attention_string(self):
        assert resolve_attn_levels("hierarchical_attention") == ["hierarchical"]

    def test_hier_string(self):
        assert resolve_attn_levels("hier") == ["hierarchical"]

    def test_hierarchical_string(self):
        assert resolve_attn_levels("hierarchical") == ["hierarchical"]

    def test_memo_aug_att_string(self):
        assert resolve_attn_levels("memo_aug_att") == ["memory"]

    def test_memory_augmented_attention_string(self):
        assert resolve_attn_levels("memory_augmented_attention") == ["memory"]

    def test_memory_string(self):
        assert resolve_attn_levels("memory") == ["memory"]

    def test_memo_aug_string(self):
        assert resolve_attn_levels("memo_aug") == ["memory"]

    def test_integer_1(self):
        assert resolve_attn_levels(1) == ["cross"]

    def test_integer_2(self):
        assert resolve_attn_levels(2) == ["hierarchical"]

    def test_integer_3(self):
        assert resolve_attn_levels(3) == ["memory"]

    def test_list_inputs(self):
        result = resolve_attn_levels(["cross", "hier_att", "memory"])
        assert result == ["cross", "hierarchical", "memory"]

    def test_list_string_numbers(self):
        result = resolve_attn_levels(["1", "2", "3"])
        assert result == ["cross", "hierarchical", "memory"]

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid attention type"):
            resolve_attn_levels("bad_attention")

    def test_invalid_integer_raises(self):
        with pytest.raises(ValueError, match="Invalid integer"):
            resolve_attn_levels(99)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid type"):
            resolve_attn_levels(3.14)

    def test_list_with_invalid_entry_raises(self):
        with pytest.raises(ValueError, match="Invalid attention type"):
            resolve_attn_levels(["cross", "not_valid"])

    def test_string_number_1(self):
        assert resolve_attn_levels("1") == ["cross"]

    def test_string_number_2(self):
        assert resolve_attn_levels("2") == ["hierarchical"]

    def test_string_number_3(self):
        assert resolve_attn_levels("3") == ["memory"]


# ---------------------------------------------------------------------------
# configure_architecture
# ---------------------------------------------------------------------------


class TestConfigureArchitecture:
    def test_defaults(self):
        cfg = configure_architecture()
        assert cfg["encoder_type"] == "hybrid"
        assert cfg["feature_processing"] == "vsn"
        assert isinstance(cfg["decoder_attention_stack"], list)

    def test_objective_transformer(self):
        cfg = configure_architecture(objective="transformer")
        assert cfg["encoder_type"] == "transformer"

    def test_use_vsn_false(self):
        cfg = configure_architecture(use_vsn=False)
        assert cfg["feature_processing"] == "dense"

    def test_attention_levels_applied(self):
        cfg = configure_architecture(attention_levels="cross")
        assert cfg["decoder_attention_stack"] == ["cross"]

    def test_architecture_config_overrides(self):
        cfg = configure_architecture(architecture_config={"encoder_type": "transformer"})
        assert cfg["encoder_type"] == "transformer"

    def test_deprecated_objective_key_in_config(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = configure_architecture(architecture_config={"objective": "transformer"})
            assert any(issubclass(warning.category, FutureWarning) for warning in w)
        assert cfg["encoder_type"] == "transformer"

    def test_use_vsn_false_config_sets_vsn_reverts_to_dense(self):
        # use_vsn=False but config says vsn → reverts to dense
        cfg = configure_architecture(
            use_vsn=False,
            architecture_config={"feature_processing": "vsn"},
        )
        assert cfg["feature_processing"] == "dense"

    def test_architecture_config_adds_extra_keys(self):
        cfg = configure_architecture(architecture_config={"my_custom": "value"})
        assert cfg["my_custom"] == "value"

    def test_default_attention_levels_none(self):
        cfg = configure_architecture(attention_levels=None)
        assert cfg["decoder_attention_stack"] == ["cross", "hierarchical", "memory"]

    def test_objective_none_defaults_to_hybrid(self):
        cfg = configure_architecture(objective=None)
        assert cfg["encoder_type"] == "hybrid"

    def test_use_vsn_true_default(self):
        cfg = configure_architecture(use_vsn=True)
        assert cfg["feature_processing"] == "vsn"


# ---------------------------------------------------------------------------
# resolve_fusion_mode
# ---------------------------------------------------------------------------


class TestResolveFusionMode:
    def test_none_returns_integrated(self):
        assert resolve_fusion_mode(None) == "integrated"

    def test_integrated(self):
        assert resolve_fusion_mode("integrated") == "integrated"

    def test_disjoint(self):
        assert resolve_fusion_mode("disjoint") == "disjoint"

    def test_independent(self):
        assert resolve_fusion_mode("independent") == "disjoint"

    def test_isolated(self):
        assert resolve_fusion_mode("isolated") == "disjoint"

    def test_invalid_falls_back_to_integrated(self):
        result = resolve_fusion_mode("unknown_mode")
        assert result == "integrated"

    def test_case_insensitive_integrated(self):
        assert resolve_fusion_mode("INTEGRATED") == "integrated"

    def test_case_insensitive_disjoint(self):
        assert resolve_fusion_mode("DISJOINT") == "disjoint"

    def test_independent_upper(self):
        assert resolve_fusion_mode("INDEPENDENT") == "disjoint"

    def test_isolated_upper(self):
        assert resolve_fusion_mode("ISOLATED") == "disjoint"


# ---------------------------------------------------------------------------
# validate_base_attentive_spec
# ---------------------------------------------------------------------------


def _make_spec(**kwargs):
    defaults = dict(
        static_input_dim=0,
        dynamic_input_dim=4,
        future_input_dim=0,
        output_dim=1,
        forecast_horizon=1,
        embed_dim=32,
        hidden_units=64,
        attention_heads=4,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        head_type="point",
    )
    defaults.update(kwargs)
    return BaseAttentiveSpec(**defaults)


class TestValidateBaseAttentiveSpec:
    def test_valid_spec_passes(self):
        spec = _make_spec()
        result = validate_base_attentive_spec(spec)
        assert result is spec

    def test_non_spec_raises_type_error(self):
        with pytest.raises(TypeError, match="BaseAttentiveSpec"):
            validate_base_attentive_spec("not_a_spec")

    def test_non_spec_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="BaseAttentiveSpec"):
            validate_base_attentive_spec({"dynamic_input_dim": 4})

    def test_negative_static_input_dim_raises(self):
        with pytest.raises(ValueError, match="static_input_dim must be >= 0"):
            validate_base_attentive_spec(_make_spec(static_input_dim=-1))

    def test_zero_dynamic_input_dim_raises(self):
        with pytest.raises(ValueError, match="dynamic_input_dim must be > 0"):
            validate_base_attentive_spec(_make_spec(dynamic_input_dim=0))

    def test_negative_dynamic_input_dim_raises(self):
        with pytest.raises(ValueError, match="dynamic_input_dim must be > 0"):
            validate_base_attentive_spec(_make_spec(dynamic_input_dim=-5))

    def test_non_int_dynamic_input_dim_raises(self):
        with pytest.raises(TypeError, match="dynamic_input_dim must be an integer"):
            validate_base_attentive_spec(_make_spec(dynamic_input_dim=4.0))

    def test_dropout_rate_below_zero_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            validate_base_attentive_spec(_make_spec(dropout_rate=-0.1))

    def test_dropout_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            validate_base_attentive_spec(_make_spec(dropout_rate=1.1))

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="layer_norm_epsilon"):
            validate_base_attentive_spec(_make_spec(layer_norm_epsilon=0.0))

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="layer_norm_epsilon"):
            validate_base_attentive_spec(_make_spec(layer_norm_epsilon=-1e-6))

    def test_invalid_head_type_raises(self):
        with pytest.raises(ValueError, match="head_type"):
            validate_base_attentive_spec(_make_spec(head_type="invalid"))

    def test_unknown_head_type_raises(self):
        with pytest.raises(ValueError, match="head_type"):
            validate_base_attentive_spec(_make_spec(head_type="regression"))

    def test_quantile_head_without_quantiles_raises(self):
        with pytest.raises(ValueError, match="quantile head_type requires"):
            validate_base_attentive_spec(_make_spec(head_type="quantile", quantiles=()))

    def test_quantile_head_with_quantiles_passes(self):
        spec = _make_spec(head_type="quantile", quantiles=(0.1, 0.5, 0.9))
        result = validate_base_attentive_spec(spec)
        assert result is spec

    def test_zero_static_input_dim_allowed(self):
        # static_input_dim=0 is allowed (allow_zero=True)
        spec = _make_spec(static_input_dim=0)
        result = validate_base_attentive_spec(spec)
        assert result.static_input_dim == 0

    def test_non_int_embed_dim_raises(self):
        with pytest.raises(TypeError):
            validate_base_attentive_spec(_make_spec(embed_dim="32"))

    def test_negative_output_dim_raises(self):
        with pytest.raises(ValueError, match="output_dim must be > 0"):
            validate_base_attentive_spec(_make_spec(output_dim=-1))

    def test_zero_forecast_horizon_raises(self):
        with pytest.raises(ValueError, match="forecast_horizon must be > 0"):
            validate_base_attentive_spec(_make_spec(forecast_horizon=0))

    def test_zero_attention_heads_raises(self):
        with pytest.raises(ValueError, match="attention_heads must be > 0"):
            validate_base_attentive_spec(_make_spec(attention_heads=0))

    def test_non_int_hidden_units_raises(self):
        with pytest.raises(TypeError, match="hidden_units must be an integer"):
            validate_base_attentive_spec(_make_spec(hidden_units=64.0))

    def test_dropout_rate_zero_passes(self):
        spec = _make_spec(dropout_rate=0.0)
        result = validate_base_attentive_spec(spec)
        assert result.dropout_rate == 0.0

    def test_dropout_rate_one_passes(self):
        spec = _make_spec(dropout_rate=1.0)
        result = validate_base_attentive_spec(spec)
        assert result.dropout_rate == 1.0

    def test_large_valid_spec(self):
        spec = _make_spec(
            static_input_dim=10,
            dynamic_input_dim=20,
            future_input_dim=5,
            output_dim=3,
            forecast_horizon=24,
            embed_dim=128,
            hidden_units=256,
            attention_heads=8,
            dropout_rate=0.2,
            layer_norm_epsilon=1e-5,
        )
        result = validate_base_attentive_spec(spec)
        assert result is spec
