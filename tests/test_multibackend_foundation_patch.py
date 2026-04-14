from __future__ import annotations


def test_import_keras_attr_loads_activations():
    from base_attentive.compat.keras import import_keras_attr

    activations = import_keras_attr("activations")
    assert activations is not None


def test_backend_neutral_aliases_match_legacy_aliases():
    from base_attentive.components import _config as cfg

    assert cfg.concat is cfg.tf_concat
    assert cfg.shape is cfg.tf_shape
    assert cfg.reshape is cfg.tf_reshape
    assert cfg.debugging is cfg.tf_debugging
