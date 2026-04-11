"""Additional tests that raise coverage on the unit-testable runtime surface."""

from __future__ import annotations

import importlib
import sys
import types
import warnings

import numpy as np
import pytest


def _reload(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class _FakeTensorLike:
    def __init__(self, value=None, item_error: bool = False):
        self.value = value
        self._item_error = item_error

    def item(self):
        if self._item_error:
            raise TypeError("boom")
        return self.value


class _FakeIntervalWithInclusive:
    def __init__(self, type_, left, right, *, closed="right", inclusive=None):
        self.type_ = type_
        self.left = left
        self.right = right
        self.closed = closed
        self.inclusive = inclusive


class _FakeIntervalWithoutInclusive:
    def __init__(self, type_, left, right, *, closed="right"):
        self.type_ = type_
        self.left = left
        self.right = right
        self.closed = closed


class _FakeKerasValidationOps:
    float32 = np.float32

    @staticmethod
    def convert_to_tensor(value):
        if isinstance(value, str) and value == "__boom__":
            raise TypeError("boom")
        return np.asarray(value)

    @staticmethod
    def reduce_mean(value, axis):
        return np.mean(np.asarray(value), axis=axis)

    @staticmethod
    def reduce_sum(value, axis):
        return np.sum(np.asarray(value), axis=axis)

    @staticmethod
    def expand_dims(value, axis=-1):
        return np.expand_dims(np.asarray(value), axis=axis)

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)


def test_top_level_runtime_helpers_cover_scalar_and_lazy_import_paths(monkeypatch):
    """The top-level runtime helpers should gracefully handle fallbacks."""
    import base_attentive as package

    assert package._normalize_configured_backend(None) == "tensorflow"
    assert package._normalize_configured_backend(" pytorch ") == "torch"
    assert package._safe_import("math").sqrt(9) == 3
    assert package._safe_import("definitely_missing_package_xyz") is None

    assert package._resolve_scalar(3) == 3
    assert package._resolve_scalar(_FakeTensorLike(5)) == 5
    assert package._resolve_scalar(_FakeTensorLike(7, item_error=True)) == 7

    fake_tf = types.SimpleNamespace(get_static_value=lambda value: 11)
    monkeypatch.setattr(
        package,
        "_safe_import",
        lambda name: fake_tf if name == "tensorflow" else None,
    )
    assert package._get_static_value(object()) == 11

    broken_tf = types.SimpleNamespace(
        get_static_value=lambda value: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        package,
        "_safe_import",
        lambda name: broken_tf if name == "tensorflow" else None,
    )
    assert package._get_static_value(object()) is None

    decorator = package._KerasAutographExperimental.do_not_convert()
    assert decorator("sentinel") == "sentinel"
    assert package._KerasAutographExperimental.do_not_convert("value") == "value"

    assert package._KerasDebuggingNamespace.assert_equal(4, 4) is None
    with pytest.raises(AssertionError, match="mismatch"):
        package._KerasDebuggingNamespace.assert_equal(1, 2, message="mismatch")

    fake_linalg = types.SimpleNamespace(band_part=lambda x, n1, n2: ("band", x, n1, n2))
    monkeypatch.setattr(
        package,
        "_safe_import",
        lambda name: types.SimpleNamespace(linalg=fake_linalg)
        if name == "tensorflow"
        else None,
    )
    assert package._KerasLinalgNamespace.band_part("x", 1, 0) == ("band", "x", 1, 0)

    monkeypatch.setattr(package, "_safe_import", lambda name: None)
    with pytest.raises(ImportError, match="only available with TensorFlow"):
        package._KerasLinalgNamespace.band_part("x", 1, 0)


def test_top_level_keras_deps_resolve_symbols_from_keras_and_tensorflow(monkeypatch):
    """The runtime namespace resolver should search keras, then TensorFlow."""
    import base_attentive as package

    fake_reduction = types.SimpleNamespace(AUTO="auto", SUM="sum", NONE="none")

    fake_keras = types.SimpleNamespace(
        __name__="keras",
        activations=types.SimpleNamespace(relu="relu"),
        random=types.SimpleNamespace(normal="normal"),
        ops=types.SimpleNamespace(concatenate="concat-op"),
        saving=types.SimpleNamespace(register_keras_serializable="register-op"),
        losses=types.SimpleNamespace(get="loss-get", Reduction=fake_reduction),
    )
    fake_tf = types.SimpleNamespace(
        zeros="tf-zeros",
        Tensor="TensorType",
        TensorShape="TensorShapeType",
        Assert="tf-assert",
        debugging=types.SimpleNamespace(assert_equal="tf-debug"),
        linalg="tf-linalg",
        keras=types.SimpleNamespace(utils=types.SimpleNamespace(custom="custom-util")),
    )

    monkeypatch.setattr(package, "KERAS_BACKEND", "tensorflow")
    monkeypatch.setattr(
        package,
        "_safe_import",
        lambda name: {
            "keras": fake_keras,
            "tensorflow": fake_tf,
            "tensorflow.keras": fake_keras,
        }.get(name),
    )

    deps = package._KerasDeps()
    assert deps.autograph.experimental.do_not_convert("ok") == "ok"
    assert deps.debugging == fake_tf.debugging
    assert deps.bool is np.bool_
    assert deps.float32 is np.float32
    assert deps.int32 is np.int32
    assert deps.Assert == "tf-assert"
    assert deps.Tensor == "TensorType"
    assert deps.TensorShape == "TensorShapeType"
    assert deps.Reduction is fake_reduction
    assert deps.get_static_value is package._get_static_value
    assert deps.linalg == "tf-linalg"
    assert deps.register_keras_serializable == "register-op"
    assert deps.get == "loss-get"
    assert deps.activations is fake_keras.activations
    assert deps.random is fake_keras.random
    assert deps.concat == "concat-op"
    assert deps.zeros == "tf-zeros"
    assert deps.zeros == "tf-zeros"  # cache hit

    monkeypatch.setattr(package, "_safe_import", lambda name: None)
    fallback_deps = package._KerasDeps()
    assert fallback_deps.Reduction.AUTO == "auto"
    assert fallback_deps.Assert(True) is True
    assert fallback_deps.Tensor is object
    assert fallback_deps.TensorShape is tuple

    with pytest.raises(ImportError, match="Cannot import missing_symbol"):
        _ = fallback_deps.missing_symbol

    assert "Keras is required for models" in package.dependency_message("models")


def test_top_level_lazy_export_handles_success_and_failure(monkeypatch):
    """The package-level ``BaseAttentive`` export should remain lazy."""
    import base_attentive as package

    package.__dict__.pop("BaseAttentive", None)
    monkeypatch.setitem(
        sys.modules,
        "base_attentive.core",
        types.SimpleNamespace(BaseAttentive="BaseAttentiveClass"),
    )
    assert package.__getattr__("BaseAttentive") == "BaseAttentiveClass"

    package.__dict__.pop("BaseAttentive", None)
    monkeypatch.setitem(sys.modules, "base_attentive.core", types.ModuleType("core"))
    with pytest.raises(AttributeError):
        package.__getattr__("BaseAttentive")

    with pytest.raises(AttributeError):
        package.__getattr__("missing")


def test_api_property_handles_param_discovery_and_nested_updates():
    """NNLearner should support nested params and fail on invalid constructors."""
    from base_attentive.api.property import NNLearner

    class Child(NNLearner):
        def __init__(self, value=1):
            self.value = value

    class Parent(NNLearner):
        def __init__(self, child=None, marker="x", cls=Child):
            self.child = Child() if child is None else child
            self.marker = marker
            self.cls = cls

    class Empty(NNLearner):
        pass

    class Bad(NNLearner):
        def __init__(self, *args):
            self.args = args

    parent = Parent()
    params = parent.get_params()
    assert params["child__value"] == 1
    assert params["cls"] is Child
    assert Empty._get_param_names() == []

    parent.set_params(marker="y", child__value=9)
    assert parent.marker == "y"
    assert parent.child.value == 9

    with pytest.raises(ValueError, match="Invalid parameter"):
        parent.set_params(unknown=1)

    with pytest.raises(RuntimeError, match="should not have variable positional"):
        Bad._get_param_names()


def test_models_and_generic_utils_cover_alias_and_error_branches():
    """The model helper modules should normalize aliases and defaults."""
    from base_attentive.models import resolve_attention_levels as resolve_from_package
    from base_attentive.models import set_default_params as set_defaults_from_package
    from base_attentive.models.comp_utils import resolve_attention_levels
    from base_attentive.models.utils import set_default_params
    from base_attentive.utils.generic_utils import select_mode

    assert resolve_from_package()["attention_heads"] == 4
    assert set_defaults_from_package({"a": 1}, b=2) == {"a": 1, "b": 2}
    assert resolve_attention_levels("cross_attention") == ["cross"]
    assert resolve_attention_levels(["hier_att", "memory_augmented_attention"]) == [
        "hierarchical",
        "memory",
    ]
    with pytest.raises(ValueError, match="Unknown attention level"):
        resolve_attention_levels("unknown")
    with pytest.raises(TypeError, match="must be a dict, string, or sequence"):
        resolve_attention_levels(42)
    with pytest.raises(TypeError, match="must contain only string values"):
        resolve_attention_levels(["cross", 1])

    quantiles, scales, return_sequences = set_default_params([0.1, 0.9], None, "mean")
    assert quantiles == [0.1, 0.9]
    assert scales == [1]
    assert return_sequences is True

    assert select_mode(None, default="fallback") == "fallback"
    assert select_mode("tft-like", canonical=["hybrid", "transformer"]) == "tft_like"
    assert select_mode("transformer", canonical=["hybrid", "transformer"]) == "transformer"


def test_dependency_decorator_warns_or_ignores_missing_packages():
    """Dependency guards should support raise, warn, and ignore modes."""
    from base_attentive.utils.deps_utils import ensure_pkg

    @ensure_pkg("definitely_missing_package_xyz", error="warn", extra="Install it.")
    def warned():
        return "warned"

    @ensure_pkg("definitely_missing_package_xyz", error="ignore")
    def ignored():
        return "ignored"

    with pytest.warns(UserWarning, match="Install it."):
        assert warned() == "warned"

    assert ignored() == "ignored"

    @ensure_pkg("definitely_missing_package_xyz", error="raise")
    def raised():
        return "raised"

    with pytest.raises(ImportError, match="required but not installed"):
        raised()


def test_validation_module_covers_verbose_and_reduction_branches(monkeypatch, caplog):
    """Validation helpers should cover runtime logging and reduction variants."""
    import base_attentive.validation as validation_module

    monkeypatch.setattr(validation_module, "KERAS_BACKEND", "tensorflow")
    monkeypatch.setattr(validation_module, "KERAS_DEPS", _FakeKerasValidationOps())

    assert validation_module._has_runtime() is True
    assert validation_module._normalize_inputs((1, 2, 3, 4)) == [1, 2, 3]

    caplog.clear()
    with caplog.at_level("INFO"):
        static, dynamic, future = validation_module.validate_model_inputs(
            [np.ones((2, 1)), np.ones((2, 3, 1)), np.ones((2, 2, 1))],
            verbose=1,
        )
    assert static.shape == (2, 1)
    assert dynamic.shape == (2, 3, 1)
    assert future.shape == (2, 2, 1)
    assert any("Validating input tensors" in message for message in caplog.messages)

    static, dynamic, future = validation_module.validate_model_inputs(
        ["__boom__", None, np.ones((1, 2, 1))],
        error="warn",
    )
    assert static == "__boom__"
    assert dynamic is None
    assert future.shape == (1, 2, 1)

    rank4 = np.ones((2, 3, 4, 1), dtype=np.float32)
    reduced_rank4 = validation_module.maybe_reduce_quantiles_bh(
        rank4,
        reduction=lambda value, axis: np.max(value, axis=axis),
    )
    assert reduced_rank4.shape == (2, 3, 1)

    rank3 = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    reduced_rank3 = validation_module.maybe_reduce_quantiles_bh(rank3, reduction="sum")
    assert reduced_rank3.shape == (2, 2)

    ensured = validation_module.ensure_bh1(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        reduce_axis=2,
        reduction="sum",
        dtype=np.float32,
    )
    assert ensured.shape == (2, 3)
    assert ensured.dtype == np.float32


def test_compat_module_covers_interval_variants_and_validate_params(monkeypatch):
    """The sklearn compat layer should handle old and new call signatures."""
    import base_attentive.compat as compat_module

    monkeypatch.setattr(compat_module, "sklearn_Interval", None)
    with pytest.raises(ImportError, match="does not have Interval support"):
        compat_module.Interval(int, 0, 1)

    monkeypatch.setattr(compat_module, "sklearn_Interval", _FakeIntervalWithInclusive)
    interval = compat_module.Interval(int, 0, 10, inclusive=True)
    assert interval.type_.__name__ == "Integral"
    assert interval.inclusive is True

    monkeypatch.setattr(compat_module, "sklearn_Interval", _FakeIntervalWithoutInclusive)
    interval = compat_module.Interval(float, 0.0, 1.0, closed="both")
    assert interval.closed == "both"

    monkeypatch.setattr(compat_module, "sklearn_validate_params", None)
    decorator = compat_module.validate_params({})
    assert decorator(lambda x: x)(1) == 1

    calls = {}

    def fake_validate_with_prefer(params, *args, prefer_skip_nested_validation=None, **kwargs):
        calls["prefer"] = prefer_skip_nested_validation

        def decorator(func):
            return func

        return decorator

    monkeypatch.setattr(compat_module, "sklearn_validate_params", fake_validate_with_prefer)
    compat_module.validate_params({}, prefer_skip_nested_validation=False)
    assert calls["prefer"] is False

    def fake_validate_without_prefer(params, *args, **kwargs):
        calls["kwargs"] = kwargs.copy()

        def decorator(func):
            return func

        return decorator

    monkeypatch.setattr(compat_module, "sklearn_validate_params", fake_validate_without_prefer)
    compat_module.validate_params({}, prefer_skip_nested_validation=False)
    assert "prefer_skip_nested_validation" not in calls["kwargs"]


def test_types_module_exports_aliases():
    """Type aliases should stay importable for callers and docs."""
    from base_attentive.compat.types import DatasetLike, PathLike, TensorLike

    assert TensorLike is object or TensorLike is not None
    assert DatasetLike is object or DatasetLike is not None
    assert PathLike is not None


def test_version_check_utilities_cover_metadata_and_error_paths(monkeypatch):
    """Version helpers should parse runtime versions across fallbacks."""
    import base_attentive.backend.version_check as version_check

    assert version_check.parse_version("3.0.0rc1") == (3, 0, 0)
    assert version_check.version_at_least("2.15.0", "2.10.0") is True
    assert version_check.get_backend_version("unknown") is None

    monkeypatch.setattr(version_check.importlib.util, "find_spec", lambda name: None)
    assert version_check.get_backend_version("tensorflow") is None

    monkeypatch.setattr(version_check.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "tensorflow", types.SimpleNamespace(__version__="9.9.9"))
    assert version_check.get_backend_version("tensorflow") == "9.9.9"
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)

    calls = []

    def fake_metadata_version(name):
        calls.append(name)
        if name == "tensorflow":
            raise version_check.importlib.metadata.PackageNotFoundError
        if name == "tensorflow-cpu":
            return "2.16.1"
        raise AssertionError(name)

    monkeypatch.setattr(version_check.importlib.metadata, "version", fake_metadata_version)
    assert version_check.get_backend_version("tensorflow") == "2.16.1"
    assert calls == ["tensorflow", "tensorflow-cpu"]

    monkeypatch.setattr(
        version_check.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert version_check.get_backend_version("jax") is None

    assert version_check.check_tensorflow_compatibility(None) == (
        False,
        "TensorFlow not installed",
    )
    assert version_check.check_tensorflow_compatibility("2.9.0")[0] is False
    assert version_check.check_tensorflow_compatibility("2.10.0")[0] is True
    assert version_check.check_torch_compatibility(None) == (False, "PyTorch not installed")
    assert version_check.check_torch_compatibility("1.13.0")[0] is False
    assert version_check.check_torch_compatibility("2.0.1+cu118")[0] is True


def test_detector_and_backend_runtime_cover_selection_and_install_paths(monkeypatch):
    """Backend selection should cover environment, fallback, and install paths."""
    import subprocess

    import base_attentive.backend as backend_module
    import base_attentive.backend.detector as detector
    import base_attentive.backend.implementations as implementations

    class AvailableBackend:
        supports_base_attentive = True
        experimental = False
        framework = "available"
        uses_keras_runtime = True
        blockers = ()

        def __init__(self, load_runtime=False):
            self.load_runtime = load_runtime

        def is_available(self):
            return True

        def get_capabilities(self):
            return {"name": "available", "available": True}

    class ExperimentalBackend(AvailableBackend):
        supports_base_attentive = False
        experimental = True
        blockers = ("x",)

    class MissingBackend(AvailableBackend):
        def __init__(self, load_runtime=False):
            raise ImportError("missing")

    class ExplodingBackend(AvailableBackend):
        def __init__(self, load_runtime=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        detector,
        "_BACKENDS",
        {
            "tensorflow": AvailableBackend,
            "jax": ExperimentalBackend,
            "torch": ExplodingBackend,
            "pytorch": ExperimentalBackend,
        },
    )
    monkeypatch.setattr(detector, "_CANONICAL_BACKENDS", ("tensorflow", "jax", "torch"))
    monkeypatch.setattr(detector, "get_backend_version", lambda name: f"{name}-1.0")

    detected = detector.detect_available_backends()
    assert detected["tensorflow"]["available"] is True
    assert detected["torch"]["error"] == "boom"

    monkeypatch.setenv("KERAS_BACKEND", "jax")
    assert detector.normalize_backend_name("keras") == "jax"
    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "jax")
    assert detector.select_best_backend(prefer="tensorflow") == "jax"
    monkeypatch.delenv("BASE_ATTENTIVE_BACKEND", raising=False)
    assert detector.select_best_backend(prefer="tensorflow") == "tensorflow"
    assert detector.select_best_backend(prefer="torch", require_supported=False) == "tensorflow"

    monkeypatch.setattr(
        detector,
        "detect_available_backends",
        lambda: {
            "tensorflow": {"available": False, "supported": False},
            "jax": {"available": True, "supported": False},
            "torch": {"available": False, "supported": False},
        },
    )
    assert detector.select_best_backend(require_supported=False) == "jax"

    monkeypatch.setattr(detector, "select_best_backend", lambda require_supported: "tensorflow" if require_supported else None)
    assert detector.ensure_default_backend() == "tensorflow"

    monkeypatch.setattr(detector, "select_best_backend", lambda require_supported: None if require_supported else "jax")
    assert detector.ensure_default_backend() == "jax"

    monkeypatch.setattr(detector, "select_best_backend", lambda require_supported: None)
    with pytest.raises(RuntimeError, match="No compatible backend installed"):
        detector.ensure_default_backend(auto_install=False)

    installs = []
    monkeypatch.setattr(detector.subprocess, "check_call", lambda cmd, stdout=None, stderr=None: installs.append(cmd) or 0)
    assert detector.ensure_default_backend(auto_install=True, install_tensorflow=False) == "jax"
    assert installs[0][-1] == "jax"

    monkeypatch.setattr(
        detector.subprocess,
        "check_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "pip")),
    )
    with pytest.raises(RuntimeError, match="Failed to install"):
        detector.ensure_default_backend(auto_install=True)

    monkeypatch.setattr(
        detector,
        "_BACKENDS",
        {
            "tensorflow": AvailableBackend,
            "jax": MissingBackend,
            "torch": AvailableBackend,
            "pytorch": ExperimentalBackend,
        },
    )
    assert detector.get_available_backends() == ["tensorflow", "torch"]

    fake_keras = types.SimpleNamespace(
        layers=types.SimpleNamespace(
            Layer="Layer",
            Dense="Dense",
            LSTM="LSTM",
            MultiHeadAttention="MHA",
            LayerNormalization="Norm",
            Dropout="Dropout",
            BatchNormalization="BatchNorm",
        ),
        Model="Model",
        Sequential="Sequential",
    )
    fake_tf = types.SimpleNamespace(Tensor="Tensor", keras=fake_keras)
    fake_torch = types.SimpleNamespace(Tensor="TorchTensor")
    monkeypatch.setattr(
        implementations,
        "_import_module",
        lambda name: {
            "tensorflow": fake_tf,
            "keras": fake_keras,
            "jax": types.SimpleNamespace(),
            "torch": fake_torch,
        }[name],
    )

    tf_backend = object.__new__(implementations.TensorFlowBackend)
    tf_backend._initialize_imports()
    assert tf_backend.Dense == "Dense"
    jax_backend = object.__new__(implementations.JaxBackend)
    jax_backend._initialize_imports()
    assert jax_backend.Layer == "Layer"
    torch_backend = object.__new__(implementations.TorchBackend)
    torch_backend._initialize_imports()
    assert torch_backend.Tensor == "TorchTensor"

    monkeypatch.setattr(
        backend_module,
        "_BACKENDS",
        {
            "tensorflow": AvailableBackend,
            "jax": ExperimentalBackend,
            "torch": MissingBackend,
            "pytorch": ExperimentalBackend,
        },
    )
    monkeypatch.setattr(backend_module, "normalize_backend_name", lambda name: str(name).strip().lower())
    monkeypatch.setattr(backend_module, "get_available_backends", lambda: ["jax"])
    monkeypatch.setattr(backend_module, "select_best_backend", lambda require_supported=True: "tensorflow")
    monkeypatch.setattr(backend_module, "ensure_default_backend", lambda auto_install=True: "tensorflow")
    monkeypatch.setattr(backend_module, "get_backend_version", lambda name: f"{name}-version")
    monkeypatch.setattr(backend_module, "check_tensorflow_compatibility", lambda: (False, "tf warn"))
    monkeypatch.setattr(backend_module, "_CURRENT_BACKEND", AvailableBackend())
    monkeypatch.delenv("BASE_ATTENTIVE_BACKEND", raising=False)
    monkeypatch.delenv("KERAS_BACKEND", raising=False)

    assert backend_module.get_backend() is backend_module._CURRENT_BACKEND

    backend_module._CURRENT_BACKEND = None
    assert isinstance(backend_module.get_backend(), AvailableBackend)
    with pytest.raises(ValueError, match="Unknown backend"):
        backend_module.get_backend("invalid")
    with pytest.raises(ValueError, match="Backend 'torch' is not available"):
        backend_module.get_backend("torch")

    backend_module._CURRENT_BACKEND = None
    monkeypatch.setattr(
        backend_module,
        "get_backend",
        lambda name=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    caps = backend_module.get_backend_capabilities()
    assert caps["available"] is True
    assert caps["version"] == "tensorflow-version"

    with pytest.raises(ValueError, match="Unknown backend"):
        backend_module.get_backend_capabilities("invalid")

    monkeypatch.setattr(
        backend_module,
        "_BACKENDS",
        {"tensorflow": ExplodingBackend, "jax": ExperimentalBackend, "torch": MissingBackend, "pytorch": ExperimentalBackend},
    )
    caps = backend_module.get_backend_capabilities("tensorflow")
    assert caps["available"] is False
    assert caps["error"] == "boom"

    monkeypatch.setattr(
        backend_module,
        "_BACKENDS",
        {"tensorflow": AvailableBackend, "jax": ExperimentalBackend, "torch": MissingBackend, "pytorch": ExperimentalBackend},
    )
    import base_attentive.backend.base as backend_base

    monkeypatch.setattr(
        backend_base,
        "_read_loaded_keras_backend",
        lambda: "jax",
    )
    monkeypatch.setattr(backend_module, "get_backend", lambda name=None: AvailableBackend())
    with pytest.warns(RuntimeWarning):
        backend = backend_module.set_backend("tensorflow")
    assert isinstance(backend, AvailableBackend)
    assert backend_module._CURRENT_BACKEND is backend
    assert backend_module.os.environ["BASE_ATTENTIVE_BACKEND"] == "tensorflow"

    monkeypatch.delenv("BASE_ATTENTIVE_BACKEND", raising=False)
    monkeypatch.setattr(backend_module, "select_best_backend", lambda require_supported=True: "jax" if require_supported else None)
    backend_module._auto_initialize()
    assert backend_module.os.environ["BASE_ATTENTIVE_BACKEND"] == "jax"
