# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Backend implementations for TensorFlow, JAX, and PyTorch."""

from __future__ import annotations

import importlib

from .base import Backend

__all__ = [
    "TensorFlowBackend",
    "JaxBackend",
    "TorchBackend",
    "PyTorchBackend",
]

_MULTI_BACKEND_BLOCKERS = (
    "BaseAttentive still contains TensorFlow-oriented compatibility paths.",
    "The compat.tf helpers are still TensorFlow-specific.",
    "Some runtime shape/assert checks still assume TensorFlow graph semantics.",
)


def _import_module(module_name: str):
    """Import a module by name."""
    return importlib.import_module(module_name)


class TensorFlowBackend(Backend):
    """TensorFlow-backed runtime."""

    name = "tensorflow"
    framework = "tensorflow"
    required_modules = ("tensorflow",)
    uses_keras_runtime = True
    supports_base_attentive = True

    def _initialize_imports(self):
        tf = _import_module("tensorflow")

        try:
            keras = _import_module("keras")
        except ImportError:
            keras = tf.keras

        layers = keras.layers

        self.tf = tf
        self.keras = keras
        self.layers = layers
        self.Tensor = getattr(tf, "Tensor", object)
        self.Layer = getattr(layers, "Layer", None)
        self.Model = getattr(keras, "Model", None)
        self.Sequential = getattr(keras, "Sequential", None)
        self.Dense = getattr(layers, "Dense", None)
        self.LSTM = getattr(layers, "LSTM", None)
        self.MultiHeadAttention = getattr(
            layers, "MultiHeadAttention", None
        )
        self.LayerNormalization = getattr(
            layers, "LayerNormalization", None
        )
        self.Dropout = getattr(layers, "Dropout", None)
        self.BatchNormalization = getattr(
            layers, "BatchNormalization", None
        )


class JaxBackend(Backend):
    """Keras-on-JAX runtime descriptor."""

    name = "jax"
    framework = "jax"
    required_modules = ("keras", "jax")
    uses_keras_runtime = True
    experimental = True
    supports_base_attentive = False
    blockers = _MULTI_BACKEND_BLOCKERS

    def _initialize_imports(self):
        keras = _import_module("keras")
        jax = _import_module("jax")

        self.keras = keras
        self.jax = jax
        self.layers = getattr(keras, "layers", None)
        self.Tensor = object
        self.Layer = getattr(self.layers, "Layer", None)
        self.Model = getattr(keras, "Model", None)
        self.Sequential = getattr(keras, "Sequential", None)
        self.Dense = getattr(self.layers, "Dense", None)
        self.LSTM = getattr(self.layers, "LSTM", None)
        self.MultiHeadAttention = getattr(
            self.layers, "MultiHeadAttention", None
        )
        self.LayerNormalization = getattr(
            self.layers, "LayerNormalization", None
        )
        self.Dropout = getattr(self.layers, "Dropout", None)
        self.BatchNormalization = getattr(
            self.layers, "BatchNormalization", None
        )


class TorchBackend(Backend):
    """Keras-on-Torch runtime descriptor."""

    name = "torch"
    framework = "torch"
    required_modules = ("keras", "torch")
    uses_keras_runtime = True
    experimental = True
    supports_base_attentive = False
    blockers = _MULTI_BACKEND_BLOCKERS

    def _initialize_imports(self):
        keras = _import_module("keras")
        torch = _import_module("torch")

        self.keras = keras
        self.torch = torch
        self.layers = getattr(keras, "layers", None)
        self.Tensor = getattr(torch, "Tensor", object)
        self.Layer = getattr(self.layers, "Layer", None)
        self.Model = getattr(keras, "Model", None)
        self.Sequential = getattr(keras, "Sequential", None)
        self.Dense = getattr(self.layers, "Dense", None)
        self.LSTM = getattr(self.layers, "LSTM", None)
        self.MultiHeadAttention = getattr(
            self.layers, "MultiHeadAttention", None
        )
        self.LayerNormalization = getattr(
            self.layers, "LayerNormalization", None
        )
        self.Dropout = getattr(self.layers, "Dropout", None)
        self.BatchNormalization = getattr(
            self.layers, "BatchNormalization", None
        )


class PyTorchBackend(TorchBackend):
    """Backward-compatible alias for the Torch runtime."""

    name = "pytorch"
