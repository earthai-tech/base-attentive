from __future__ import annotations

import importlib
import sys


class _ModuleGuard:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.saved = None

    def __enter__(self):
        self.saved = {
            name: module
            for name, module in list(sys.modules.items())
            if name == self.prefix
            or name.startswith(f"{self.prefix}.")
        }
        for name in list(self.saved):
            sys.modules.pop(name, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name in list(sys.modules):
            if name == self.prefix or name.startswith(
                f"{self.prefix}."
            ):
                sys.modules.pop(name, None)
        sys.modules.update(self.saved or {})


def test_root_import_is_lazy():
    with _ModuleGuard("base_attentive"):
        module = importlib.import_module("base_attentive")

        assert "base_attentive._bootstrap" not in sys.modules
        assert (
            "base_attentive.keras_runtime" not in sys.modules
        )
        assert "base_attentive.backend" not in sys.modules
        assert module.__version__


def test_runtime_attr_bootstraps_without_backend_import():
    with _ModuleGuard("base_attentive"):
        module = importlib.import_module("base_attentive")
        assert module.KERAS_BACKEND in {
            "tensorflow",
            "jax",
            "torch",
        }
        assert "base_attentive._bootstrap" in sys.modules
        assert "base_attentive.backend" not in sys.modules
