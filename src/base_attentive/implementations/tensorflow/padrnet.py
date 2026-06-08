"""TensorFlow implementation of PADR-Net."""

from __future__ import annotations

from typing import Any

from ..._bootstrap import KERAS_DEPS
from ...api.property import NNLearner
from ...applications.flood.config import PADRNetConfig

try:
    import tensorflow as tf

    try:
        layers = tf.keras.layers
        Model = tf.keras.Model
        Sequential = tf.keras.Sequential
        keras_register = (
            tf.keras.utils.register_keras_serializable
        )
    except (ImportError, ModuleNotFoundError):
        from tensorflow.python import keras as _tf_keras
        from tensorflow.python.keras import layers
        from tensorflow.python.keras.utils import (
            generic_utils as _generic_utils,
        )

        keras_register = (
            _generic_utils.register_keras_serializable
        )

        Model = _tf_keras.Model
        Sequential = _tf_keras.Sequential
except ImportError:  # pragma: no cover
    tf = None
    layers = None
    Model = object
    Sequential = None
    keras_register = None


def _ensure_tensorflow() -> None:
    if tf is None or layers is None:
        raise ImportError(
            "TensorFlow is required for TensorFlowPADRNet. "
            "Install it with: pip install tensorflow"
        )


def _register_keras_serializable(name: str):
    if keras_register is None:
        return lambda cls: cls
    return keras_register(
        package=__name__,
        name=name,
    )


def _normalization_layer(name: str):
    layer_norm = getattr(layers, "LayerNormalization", None)
    if layer_norm is not None:
        return layer_norm(epsilon=1e-6, name=name)
    return layers.Lambda(lambda x: x, name=name)


_TF_PADR_BASES = (Model, NNLearner) if tf else (NNLearner,)


@_register_keras_serializable("PADRBlock")
class _PADRBlock(layers.Layer if layers else object):
    def __init__(
        self,
        config: PADRNetConfig,
        *,
        name: str | None = None,
    ):
        _ensure_tensorflow()
        super().__init__(name=name)
        key_dim = max(
            1, config.hidden_dim // config.num_heads
        )
        attention_cls = getattr(
            layers, "MultiHeadAttention", None
        )
        self.uses_mha = attention_cls is not None
        self.attention = (
            attention_cls(
                num_heads=config.num_heads,
                key_dim=key_dim,
                dropout=config.dropout,
                name="attention",
            )
            if self.uses_mha
            else layers.Dense(
                config.hidden_dim,
                activation="gelu",
                name="attention_fallback",
            )
        )
        self.norm_attention = _normalization_layer(
            "attention_norm"
        )
        self.ffn = Sequential(
            [
                layers.Dense(
                    config.hidden_dim * 2,
                    activation="gelu",
                ),
                layers.Dropout(config.dropout),
                layers.Dense(config.hidden_dim),
            ],
            name="ffn",
        )
        self.norm_ffn = _normalization_layer("ffn_norm")

    def call(
        self,
        inputs,
        training: bool = False,
    ):  # noqa: FBT002
        if self.uses_mha:
            attn = self.attention(
                inputs, inputs, training=training
            )
        else:
            attn = self.attention(inputs)
        x = self.norm_attention(inputs + attn)
        ffn = self.ffn(x, training=training)
        return self.norm_ffn(x + ffn)


@_register_keras_serializable("TensorFlowPADRNet")
class TensorFlowPADRNet(*_TF_PADR_BASES):
    """Physics-aware attention model for flood forecasting."""

    def __init__(
        self,
        config: PADRNetConfig,
        *,
        name: str = "padr_net",
        **kwargs: Any,
    ):
        _ensure_tensorflow()
        super().__init__(name=name, **kwargs)
        self.config = config
        self.dynamic_projection = layers.Dense(
            config.hidden_dim,
            activation="gelu",
            name="dynamic_projection",
        )
        self.static_projection = (
            layers.Dense(
                config.hidden_dim,
                activation="gelu",
                name="static_projection",
            )
            if config.static_dim > 0
            else None
        )
        self.blocks = [
            _PADRBlock(config, name=f"padr_block_{i}")
            for i in range(config.num_layers)
        ]
        self.memory_pool = layers.GlobalAveragePooling1D(
            name="memory_pool"
        )
        self.depth_head = Sequential(
            [
                layers.Dense(
                    config.hidden_dim,
                    activation="gelu",
                ),
                layers.Dropout(config.dropout),
                layers.Dense(config.forecast_horizon),
                layers.Activation("softplus"),
            ],
            name="depth_head",
        )

    def call(  # noqa: D102
        self,
        inputs,
        static_inputs=None,
        training: bool = False,
    ):
        x = self.dynamic_projection(inputs)
        if (
            self.static_projection is not None
            and static_inputs is not None
        ):
            static = self.static_projection(static_inputs)
            static = KERAS_DEPS.expand_dims(static, axis=1)
            x = x + static
        for block in self.blocks:
            x = block(x, training=training)

        features = self.memory_pool(x)
        depth = self.depth_head(features, training=training)
        depth = KERAS_DEPS.expand_dims(depth, axis=-1)
        scale = max(
            0.004, 0.08 * self.config.flood_threshold
        )
        logits = (
            depth - self.config.flood_threshold
        ) / scale
        exceedance = KERAS_DEPS.sigmoid(logits)
        return {
            "depth": depth,
            "exceedance_probability": exceedance,
            "features": features,
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "padrnet_config": self.config.__dict__,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        padrnet_config = config.pop("padrnet_config")
        return cls(PADRNetConfig(**padrnet_config), **config)


__all__ = ["TensorFlowPADRNet"]
