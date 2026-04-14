from __future__ import annotations

import json

import pytest

from base_attentive.config import (
    normalize_base_attentive_spec,
    serialize_base_attentive_spec,
)
from base_attentive.experimental.base_attentive_v2 import (
    BaseAttentiveV2,
)


def test_serialize_base_attentive_spec_normalizes_alias_paths(
    sample_dims,
):
    spec = normalize_base_attentive_spec(
        architecture={
            "sequence_pooling": "pool.last",
            "fusion": "fusion.concat",
        },
        runtime={"final_agg": "average"},
        quantiles=(0.1, 0.5, 0.9),
        **sample_dims,
    )

    payload = serialize_base_attentive_spec(spec)

    assert "sequence_pooling" not in payload["architecture"]
    assert "fusion" not in payload["architecture"]
    assert (
        payload["components"]["sequence_pooling"]
        == "pool.last"
    )
    assert payload["components"]["fusion"] == "fusion.concat"
    assert payload["runtime"]["final_agg"] == "average"
    assert payload["quantiles"] == [0.1, 0.5, 0.9]


def test_v2_from_config_accepts_nested_spec_only(sample_dims):
    model = BaseAttentiveV2(
        **sample_dims,
        quantiles=(0.1, 0.5, 0.9),
        head_type="quantile",
        spec={
            "architecture": {"sequence_pooling": "pool.last"},
        },
    )
    config = model.get_config()

    nested_only = {
        "spec": config["spec"],
        "name": config["name"],
        "trainable": config.get("trainable", True),
        "dtype": config.get("dtype"),
    }

    clone = BaseAttentiveV2.from_config(nested_only)

    assert (
        clone.spec.static_input_dim
        == model.spec.static_input_dim
    )
    assert (
        clone.spec.dynamic_input_dim
        == model.spec.dynamic_input_dim
    )
    assert (
        clone.spec.future_input_dim
        == model.spec.future_input_dim
    )
    assert (
        clone.spec.components.sequence_pooling == "pool.last"
    )
    assert tuple(clone.spec.quantiles) == tuple(
        model.spec.quantiles
    )


@pytest.mark.skipif(
    pytest.importorskip("keras") is None,
    reason="Keras not installed",
)
def test_v2_to_json_roundtrip_preserves_resolver_spec(
    sample_dims,
):
    import keras

    model = BaseAttentiveV2(
        **sample_dims,
        quantiles=(0.1, 0.9),
        head_type="quantile",
        spec={
            "architecture": {
                "sequence_pooling": "pool.last",
                "fusion": "fusion.concat",
            },
            "runtime": {"final_agg": "flatten"},
        },
    )

    payload = json.loads(model.to_json())
    clone = keras.models.model_from_json(model.to_json())

    assert (
        payload["config"]["spec"]["components"][
            "sequence_pooling"
        ]
        == "pool.last"
    )
    assert (
        payload["config"]["spec"]["components"]["fusion"]
        == "fusion.concat"
    )
    assert (
        payload["config"]["spec"]["runtime"]["final_agg"]
        == "flatten"
    )
    assert (
        clone.spec.components.sequence_pooling == "pool.last"
    )
    assert clone.spec.components.fusion == "fusion.concat"
    assert clone.spec.runtime.final_agg == "flatten"
    assert tuple(clone.spec.quantiles) == tuple(
        model.spec.quantiles
    )
