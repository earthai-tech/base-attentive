"""Test backend abstraction module without importing heavy runtimes."""

from __future__ import annotations

import sys
import types

import pytest


def _fake_keras_module(runtime: str = "jax"):
    keras = types.ModuleType("keras")
    keras.backend = types.SimpleNamespace(backend=lambda: runtime)
    keras.layers = types.SimpleNamespace(
        Layer=object,
        Dense=object,
        LSTM=object,
        MultiHeadAttention=object,
        LayerNormalization=object,
        Dropout=object,
        BatchNormalization=object,
    )
    keras.Model = object
    keras.Sequential = object
    return keras


class TestBackendModule:
    """Test backend abstraction system."""

    def test_get_available_backends_uses_lightweight_detection(self, monkeypatch):
        """Availability probing should not need to import heavy runtimes."""
        from base_attentive import backend as backend_module

        monkeypatch.setattr(
            backend_module,
            "_has_module",
            lambda name: name in {"keras", "jax", "torch"},
        )

        backends = backend_module.get_available_backends()

        assert backends == ["jax", "torch"]

    def test_get_backend_invalid(self):
        """Unknown backend names should still fail fast."""
        from base_attentive.backend import get_backend

        with pytest.raises(ValueError):
            get_backend("invalid_backend_xyz")

    def test_get_backend_respects_keras_backend_env(self, monkeypatch):
        """The default lookup should fall back to KERAS_BACKEND."""
        from base_attentive import backend as backend_module

        fake_keras = _fake_keras_module("jax")
        fake_jax = types.ModuleType("jax")

        monkeypatch.delenv("BASE_ATTENTIVE_BACKEND", raising=False)
        monkeypatch.setenv("KERAS_BACKEND", "jax")
        monkeypatch.setattr(
            backend_module,
            "_has_module",
            lambda name: name in {"keras", "jax"},
        )
        monkeypatch.setattr(
            backend_module,
            "_import_module",
            lambda name: {
                "keras": fake_keras,
                "jax": fake_jax,
            }[name],
        )

        backend = backend_module.get_backend()

        assert backend.name == "jax"
        assert backend.framework == "jax"

    def test_set_backend_normalizes_torch_alias_and_syncs_env(self, monkeypatch):
        """Setting the backend should normalize aliases and sync env vars."""
        from base_attentive import backend as backend_module

        fake_keras = _fake_keras_module("torch")
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = object

        monkeypatch.setattr(
            backend_module,
            "_has_module",
            lambda name: name in {"keras", "torch"},
        )
        monkeypatch.setattr(
            backend_module,
            "_import_module",
            lambda name: {
                "keras": fake_keras,
                "torch": fake_torch,
            }[name],
        )

        backend = backend_module.set_backend("pytorch")

        assert backend.name == "torch"
        assert backend.framework == "torch"
        assert backend_module.get_backend().name == "torch"

    def test_get_backend_capabilities_reports_experimental_runtimes(self, monkeypatch):
        """Capability reports should document JAX/Torch as provisional."""
        from base_attentive import backend as backend_module

        monkeypatch.setattr(
            backend_module,
            "_has_module",
            lambda name: name in {"keras", "jax"},
        )

        capabilities = backend_module.get_backend_capabilities("jax")

        assert capabilities["name"] == "jax"
        assert capabilities["experimental"] is True
        assert capabilities["supports_base_attentive"] is False
        assert capabilities["blockers"]

    def test_set_backend_warns_when_keras_runtime_is_already_loaded(self, monkeypatch):
        """Switching after Keras is loaded should emit a restart warning."""
        from base_attentive import backend as backend_module

        fake_keras = _fake_keras_module("tensorflow")
        fake_jax = types.ModuleType("jax")

        monkeypatch.setattr(
            backend_module,
            "_has_module",
            lambda name: name in {"keras", "jax"},
        )
        monkeypatch.setattr(
            backend_module,
            "_import_module",
            lambda name: {
                "keras": fake_keras,
                "jax": fake_jax,
            }[name],
        )
        monkeypatch.setitem(sys.modules, "keras", fake_keras)

        with pytest.warns(RuntimeWarning, match="Restart Python"):
            backend = backend_module.set_backend("jax")

        assert backend.name == "jax"
