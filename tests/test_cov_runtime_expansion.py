"""Coverage expansion tests for runtime-facing helpers."""

from __future__ import annotations

import os
import types
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("BASE_ATTENTIVE_BACKEND", "torch")


def test_api_docs_components_cover_trim_and_nested_access():
    from base_attentive.api.docs import DocstringComponents

    docs = DocstringComponents({"core": "\ntrimmed text\n    "})
    assert docs.core == "trimmed text"

    nested = DocstringComponents.from_nested_components(
        child=docs
    )
    assert nested.child is docs

    with pytest.raises(AttributeError):
        _ = docs.missing


def test_backend_base_helpers_cover_override_and_capabilities(
    monkeypatch,
):
    import base_attentive.backend.base as base

    helper_mod = types.SimpleNamespace(
        _has_module=lambda name: name == "ok.module",
        _import_module=lambda name: {"name": name},
    )
    monkeypatch.setitem(
        base.sys.modules, "base_attentive.backend", helper_mod
    )
    assert base._has_module("ok.module") is True
    assert base._import_module("demo.module") == {
        "name": "demo.module"
    }

    monkeypatch.delitem(
        base.sys.modules,
        "base_attentive.backend",
        raising=False,
    )
    monkeypatch.setattr(
        base.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert base._has_module("broken.module") is False

    fake_keras = types.SimpleNamespace(
        backend=types.SimpleNamespace(
            backend=lambda: "pytorch"
        )
    )
    monkeypatch.setitem(base.sys.modules, "keras", fake_keras)
    assert base._read_loaded_keras_backend() == "torch"

    class DemoBackend(base.Backend):
        name = "demo"
        framework = "demo"
        required_modules = ("demo.module",)
        experimental = True
        supports_base_attentive = True
        supports_base_attentive_v2 = True
        blockers = ("missing docs",)
        v2_blockers = ("missing runtime",)

    monkeypatch.setattr(
        base, "_has_module", lambda module_name: True
    )
    backend = DemoBackend(load_runtime=False)
    caps = backend.get_capabilities()

    assert caps["name"] == "demo"
    assert caps["framework"] == "demo"
    assert caps["experimental"] is True
    assert caps["supports_base_attentive"] is True
    assert caps["supports_base_attentive_v2"] is True
    assert caps["blockers"] == ["missing docs"]
    assert caps["v2_blockers"] == ["missing runtime"]


def test_detector_paths_cover_selection_install_and_errors(
    monkeypatch,
):
    import base_attentive.backend.detector as detector

    with pytest.raises(RuntimeError, match="Unknown backend"):
        detector._backend_install_target("unknown")

    assert detector.normalize_backend_name(None) == "tensorflow"
    assert detector.normalize_backend_name("  ") == "tensorflow"

    monkeypatch.setenv("KERAS_BACKEND", "torch")
    assert detector.normalize_backend_name("keras") == "torch"

    monkeypatch.setattr(
        detector.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(ImportError("boom")),
    )
    assert detector._has_module("missing") is False

    monkeypatch.setattr(
        detector,
        "subprocess",
        types.SimpleNamespace(
            DEVNULL=object(),
            CalledProcessError=RuntimeError,
            check_call=lambda *args, **kwargs: (
                (_ for _ in ()).throw(RuntimeError("pip failed"))
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="Failed to install"):
        detector.install_backend_runtime("torch")

    class _Backend:
        supports_base_attentive = True
        supports_base_attentive_v2 = True
        experimental = False

        def __init__(self, load_runtime=False):
            del load_runtime

        def is_available(self):
            return True

    class _BrokenBackend:
        def __init__(self, load_runtime=False):
            del load_runtime
            raise RuntimeError("broken backend")

    monkeypatch.setattr(
        detector,
        "_CANONICAL_BACKENDS",
        ("tensorflow", "jax"),
    )
    monkeypatch.setattr(
        detector,
        "_BACKENDS",
        {
            "tensorflow": _Backend,
            "jax": _BrokenBackend,
            "torch": _Backend,
            "pytorch": _Backend,
        },
    )
    monkeypatch.setattr(
        detector,
        "get_backend_version",
        lambda name: f"{name}-1.0",
    )

    detected = detector.detect_available_backends()
    assert detected["tensorflow"]["available"] is True
    assert detected["jax"]["available"] is False
    assert "error" in detected["jax"]

    monkeypatch.setattr(
        detector,
        "detect_available_backends",
        lambda: {
            "tensorflow": {
                "available": False,
                "supported": True,
            },
            "jax": {
                "available": True,
                "supported": False,
            },
            "torch": {
                "available": True,
                "supported": True,
            },
        },
    )
    monkeypatch.delenv(
        "BASE_ATTENTIVE_BACKEND", raising=False
    )
    assert detector.select_best_backend(prefer="torch") == "torch"
    assert (
        detector.select_best_backend(require_supported=False)
        == "jax"
    )

    monkeypatch.setattr(
        detector,
        "select_best_backend",
        lambda require_supported=True, prefer=None: None,
    )
    with pytest.raises(RuntimeError, match="No compatible backend"):
        detector.ensure_default_backend(auto_install=False)

    installs: list[str] = []
    monkeypatch.setattr(
        detector,
        "select_best_backend",
        lambda require_supported=True, prefer=None: None,
    )
    monkeypatch.setattr(
        detector,
        "install_backend_runtime",
        lambda name: installs.append(name),
    )
    assert (
        detector.ensure_default_backend(
            auto_install=True, install_tensorflow=False
        )
        == "jax"
    )
    assert installs == ["jax"]

    class _ImportBackend:
        def __init__(self, load_runtime=False):
            del load_runtime
            raise ImportError("missing runtime")

    monkeypatch.setattr(
        detector,
        "_BACKENDS",
        {"tensorflow": _ImportBackend, "jax": _Backend},
    )
    monkeypatch.setattr(
        detector, "_CANONICAL_BACKENDS", ("tensorflow", "jax")
    )
    assert detector.get_available_backends() == ["jax"]


def test_backend_module_covers_core_api_and_lazy_exports(
    monkeypatch,
):
    import base_attentive.backend as backend

    sentinel = object()
    monkeypatch.setattr(backend, "_CURRENT_BACKEND", sentinel)
    monkeypatch.delenv(
        "BASE_ATTENTIVE_BACKEND", raising=False
    )
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    assert backend.get_backend() is sentinel

    class _Backend:
        name = "torch"
        framework = "torch"
        uses_keras_runtime = True
        experimental = False
        supports_base_attentive = True
        supports_base_attentive_v2 = True
        blockers = ("x",)
        v2_blockers = ("y",)

        def __init__(self, load_runtime=True):
            self.load_runtime = load_runtime

        def is_available(self):
            return True

        def get_capabilities(self):
            return {"name": "torch"}

    installs: list[str] = []
    created: list[bool] = []

    class _FlakyBackend(_Backend):
        def __init__(self, load_runtime=True):
            created.append(load_runtime)
            if len(created) == 1:
                raise ImportError("install me")
            super().__init__(load_runtime=load_runtime)

    fake_detector = types.SimpleNamespace(
        ensure_default_backend=lambda **kwargs: "torch",
        get_available_backends=lambda: ["jax"],
        backend_install_command=lambda name: f"pip install {name}",
        install_backend_runtime=lambda name: installs.append(name),
    )

    def _fake_module(name: str):
        if name == "base_attentive.backend.detector":
            return fake_detector
        if name == "base_attentive.backend.base":
            return types.SimpleNamespace(
                _read_loaded_keras_backend=lambda: "jax"
            )
        if name == "base_attentive.backend.torch_utils":
            return types.SimpleNamespace(
                TorchDeviceManager="manager",
                get_torch_device=lambda: "cpu",
                get_torch_version=lambda: "2.5.0",
                torch_is_available=lambda: True,
            )
        if name == "base_attentive.backend.version_check":
            return types.SimpleNamespace(
                get_backend_version=lambda backend_name: f"{backend_name}-2.0",
                check_tensorflow_compatibility=lambda: (True, None),
                check_torch_compatibility=lambda: (True, None),
                parse_version=lambda value: value,
                version_at_least=lambda current, minimum: True,
            )
        return types.SimpleNamespace()

    monkeypatch.setattr(backend, "_module", _fake_module)
    monkeypatch.setattr(
        backend,
        "_backend_classes",
        lambda: {
            "tensorflow": _Backend,
            "jax": _Backend,
            "torch": _FlakyBackend,
            "pytorch": _Backend,
        },
    )
    monkeypatch.setattr(
        backend, "normalize_backend_name", lambda name: name
    )
    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "auto")
    monkeypatch.setenv("BASE_ATTENTIVE_AUTO_INSTALL", "1")
    monkeypatch.setattr(backend, "_CURRENT_BACKEND", None)

    active = backend.get_backend()
    assert isinstance(active, _FlakyBackend)
    assert installs == ["torch"]

    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unknown backend"):
        backend.get_backend()

    monkeypatch.setattr(
        backend,
        "_backend_classes",
        lambda: {"torch": _Backend},
    )
    monkeypatch.setattr(
        backend,
        "get_backend",
        lambda name=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "torch")
    caps = backend.get_backend_capabilities()
    assert caps["name"] == "torch"
    assert caps["available"] is True

    class _BrokenBackend(_Backend):
        def __init__(self, load_runtime=False):
            del load_runtime
            raise RuntimeError("broken")

    monkeypatch.setattr(
        backend,
        "_backend_classes",
        lambda: {"torch": _BrokenBackend},
    )
    caps = backend.get_backend_capabilities("torch")
    assert caps["available"] is False
    assert caps["error"] == "broken"

    caught = []
    monkeypatch.setattr(
        backend,
        "check_tensorflow_compatibility",
        lambda: (False, "tf warning"),
    )
    monkeypatch.setattr(
        backend.warnings,
        "warn",
        lambda message, *args, **kwargs: caught.append(message),
    )
    monkeypatch.setattr(
        backend, "get_backend", lambda name=None: {"name": name}
    )
    result = backend.set_backend("tensorflow")
    assert result == {"name": "tensorflow"}
    assert os.environ["BASE_ATTENTIVE_BACKEND"] == "tensorflow"
    assert os.environ["KERAS_BACKEND"] == "tensorflow"
    assert any("tf warning" in msg for msg in caught)
    assert any("already loaded" in msg for msg in caught)

    monkeypatch.delenv(
        "BASE_ATTENTIVE_BACKEND", raising=False
    )
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        backend._auto_initialize()

    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "auto")
    monkeypatch.setattr(
        backend, "ensure_default_backend", lambda **kwargs: "jax"
    )
    monkeypatch.setattr(
        backend, "set_backend", lambda name: f"set:{name}"
    )
    assert backend._auto_initialize() == "set:jax"

    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "torch")
    assert backend._auto_initialize() == "set:torch"

    monkeypatch.setattr(
        backend,
        "_module",
        lambda name: types.SimpleNamespace(TorchDeviceManager="ok"),
    )
    assert backend.__getattr__("TorchDeviceManager") == "ok"
    with pytest.raises(AttributeError):
        backend.__getattr__("missing")


def test_keras_runtime_helpers_cover_fallbacks(monkeypatch):
    import base_attentive._keras_runtime as runtime

    monkeypatch.setattr(
        runtime, "_bootstrap_module", lambda: SimpleNamespace()
    )

    calls = []

    class _Compat(runtime._CompatLayer):
        def build(self, input_shape=None):
            calls.append(("build", input_shape))
            if input_shape is not None:
                raise TypeError("retry without args")
            self.built = True

        def call(self, value, training=False):
            calls.append(("call", training))
            return value

    layer = _Compat(name="demo")
    out = layer(np.ones((2, 3)), training=True)
    assert out.shape == (2, 3)
    assert calls == [
        ("build", (2, 3)),
        ("build", None),
        ("call", True),
    ]

    init_calls = []

    def _initializer(shape, dtype=None):
        init_calls.append((shape, dtype))
        if dtype is not None:
            raise TypeError("shape only")
        return np.ones(shape)

    weight = runtime._CompatLayer().add_weight(
        name="w",
        shape=(2, 2),
        dtype=np.float32,
        initializer=_initializer,
    )
    assert weight.shape == (2, 2)
    assert init_calls == [((2, 2), np.float32), ((2, 2), None)]

    monkeypatch.setattr(
        runtime, "_keras_deps", lambda: (_ for _ in ()).throw(ImportError())
    )
    assert runtime.resolve_keras_dep("Layer", fallback="x") == "x"
    with pytest.raises(ImportError):
        runtime.resolve_keras_dep("Layer")

    monkeypatch.setattr(
        runtime,
        "resolve_keras_dep",
        lambda name, fallback=runtime._MISSING: object
        if name == "Layer"
        else "model",
    )
    assert runtime.get_layer_class() is runtime._CompatLayer
    assert runtime.get_model_class() == "model"

    called = []

    def _register(package, name=None):
        called.append((package, name))
        return "decorator"

    monkeypatch.setattr(
        runtime,
        "resolve_keras_dep",
        lambda name, fallback=runtime._MISSING: _register,
    )
    assert runtime.register_keras_serializable("Pkg", name="Thing") == "decorator"
    assert called == [("Pkg", "Thing")]


def test_versioning_rules_cover_transforms_and_warnings():
    from base_attentive.compat.versioning import (
        DeprecatedParameterWarning,
        ParameterRule,
        RemovedParameterWarning,
        UnsupportedCompatibilityWarning,
        apply_parameter_compatibility,
        n_quantiles_to_quantiles,
        resolve_deprecated_config,
        resolve_deprecated_kwargs,
    )

    assert n_quantiles_to_quantiles(1) == (0.5,)
    assert n_quantiles_to_quantiles(3) == (0.25, 0.5, 0.75)
    with pytest.raises(TypeError, match="must be an integer"):
        n_quantiles_to_quantiles("x")
    with pytest.raises(ValueError, match=">= 1"):
        n_quantiles_to_quantiles(0)

    rules = (
        ParameterRule(
            old_name="old",
            new_name="new",
            behavior="rename",
            transform=lambda value: value + 1,
        ),
        ParameterRule(
            old_name="drop_me",
            behavior="removed",
        ),
        ParameterRule(
            old_name="noop_flag",
            behavior="noop",
            implemented=False,
        ),
    )
    with pytest.warns(DeprecatedParameterWarning):
        resolved = resolve_deprecated_kwargs(
            {"old": 2},
            rules[:1],
            component_name="Thing",
        )
    assert resolved == {"new": 3}

    with pytest.warns(DeprecatedParameterWarning):
        resolved = resolve_deprecated_kwargs(
            {"old": 1, "new": 9},
            (
                ParameterRule(
                    old_name="old",
                    new_name="new",
                    behavior="rename",
                    precedence="new",
                ),
            ),
            component_name="Thing",
        )
    assert resolved == {"new": 9}

    with pytest.warns(RemovedParameterWarning):
        resolved = resolve_deprecated_config(
            {"drop_me": True},
            rules[1:2],
            component_name="Cfg",
        )
    assert resolved == {}

    with pytest.warns(UnsupportedCompatibilityWarning):
        resolved = resolve_deprecated_kwargs(
            {"noop_flag": True},
            rules[2:],
            component_name="Thing",
        )
    assert resolved == {}

    with pytest.raises(ValueError, match="Unsupported compatibility"):
        resolve_deprecated_kwargs(
            {"x": 1},
            (ParameterRule(old_name="x", behavior="weird"),),
        )

    @apply_parameter_compatibility(
        (
            ParameterRule(
                old_name="legacy",
                new_name="modern",
                behavior="rename",
            ),
        ),
        component_name="Decorated",
    )
    def _init(self, **kwargs):
        self.kwargs = kwargs

    holder = SimpleNamespace()
    with pytest.warns(DeprecatedParameterWarning):
        _init(holder, legacy=5)
    assert holder.kwargs == {"modern": 5}


def test_api_property_and_experimental_helpers_cover_extra_branches():
    from base_attentive.api.property import NNLearner
    import base_attentive.experimental as experimental_pkg

    class _RaisingLen:
        def __len__(self):
            raise RuntimeError("boom")

    class _ArrayLike:
        shape = [2, 3]
        dtype = "float32"

    class _BrokenShape:
        def __iter__(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "<broken-shape>"

    class _ShapeBroken:
        dtype = None
        shape = _BrokenShape()

    class _Mini(NNLearner):
        _repr_width = 14
        _repr_max_items = 2
        _repr_max_value_chars = 8

        def __init__(self, value=None):
            self.value = value

    assert NNLearner._safe_len(_RaisingLen()) is None
    assert NNLearner._container_summary(_RaisingLen()) == "_RaisingLen(...)"
    summary = NNLearner._array_summary(_ArrayLike())
    assert "shape=(2, 3)" in summary
    assert "dtype=float32" in summary
    assert NNLearner._safe_shape(_ShapeBroken()) == "<broken-shape>"
    assert NNLearner._safe_dtype(_ShapeBroken()) == "?"
    assert NNLearner._truncate_text(
        "abc def ghi", max_chars=7
    ) == "abc ..."
    assert NNLearner._indent_block("a\nb", spaces=2) == "  a\n  b"

    seq_text = _Mini._format_sequence(
        (1,),
        depth=0,
        indent=0,
        visited=set(),
        cfg=_Mini._repr_config(),
    )
    assert seq_text.endswith(",)")

    dict_text = _Mini._format_dict(
        {"a": 1, "b": 2, "c": 3},
        depth=0,
        indent=0,
        visited=set(),
        cfg=_Mini._repr_config(),
    )
    assert "..." in dict_text or "\n" in dict_text

    class _BrokenLearner(NNLearner):
        def __init__(self):
            pass

        def get_params(self, deep=False):
            del deep
            raise RuntimeError("boom")

    assert "BrokenLearner" in _BrokenLearner()._repr_text()

    monkeypatch = pytest.MonkeyPatch()
    try:
        fake_module = SimpleNamespace(BaseAttentiveV2="sentinel")
        monkeypatch.setattr(
            experimental_pkg.importlib,
            "import_module",
            lambda name: fake_module,
        )
        experimental_pkg.__dict__.pop("BaseAttentiveV2", None)
        assert experimental_pkg.__getattr__("BaseAttentiveV2") == "sentinel"
        assert "BaseAttentiveV2" in experimental_pkg.__dir__()
        with pytest.raises(AttributeError):
            experimental_pkg.__getattr__("missing")
    finally:
        monkeypatch.undo()


def test_api_property_repr_and_param_paths_cover_more_branches():
    from base_attentive.api.property import NNLearner

    class _Child(NNLearner):
        def __init__(self, width=1):
            self.width = width

    class _Verbose(NNLearner):
        _repr_width = 18
        _repr_max_items = 2

        def __init__(
            self,
            child=None,
            values=None,
            func=len,
            klass=dict,
        ):
            self.child = _Child(3) if child is None else child
            self.values = (
                [1, 2, 3, 4] if values is None else values
            )
            self.func = func
            self.klass = klass

    cfg = _Verbose._repr_config()

    assert NNLearner._is_array_like("x") is False
    assert NNLearner._is_array_like([1, 2]) is False
    assert NNLearner._is_learner_like(_Child) is False
    assert NNLearner._callable_name(len) == "len"
    assert NNLearner._callable_name(_Child) == "_Child"

    items, truncated = NNLearner._iter_items_limited(
        range(5), max_items=2
    )
    assert items == [0, 1]
    assert truncated is True

    assert (
        _Verbose._format_value(
            _Child,
            depth=0,
            indent=0,
            visited=set(),
            cfg=cfg,
        )
        == "_Child"
    )
    assert (
        _Verbose._format_value(
            lambda x: x,
            depth=0,
            indent=0,
            visited=set(),
            cfg=cfg,
        )
        == "<lambda>"
    )

    set_text = _Verbose._format_value(
        {3, 1, 2},
        depth=0,
        indent=0,
        visited=set(),
        cfg=cfg,
    )
    assert set_text.startswith("{")

    cyclic = []
    cyclic.append(cyclic)
    assert (
        _Verbose._format_value(
            cyclic,
            depth=0,
            indent=0,
            visited=set(),
            cfg=cfg,
        )
        is not None
    )

    class _BrokenNames(_Verbose):
        def get_params(self, deep=False):
            del deep
            return {
                "child": self.child,
                "values": self.values,
                "func": self.func,
                "klass": self.klass,
            }

        @classmethod
        def _get_param_names(cls):
            raise RuntimeError("boom")

    learner_text = _BrokenNames._format_learner(
        _BrokenNames(),
        depth=0,
        indent=0,
        visited=set(),
        cfg=cfg,
    )
    assert "child=" in learner_text

    verbose = _Verbose(values={"a": [1, 2, 3, 4]})
    params_shallow = verbose.get_params(deep=False)
    params_deep = verbose.get_params(deep=True)
    assert "child" in params_shallow
    assert "child__width" not in params_shallow
    assert params_deep["child__width"] == 3

    assert verbose.set_params() is verbose
    verbose.set_params(child__width=9)
    assert verbose.child.width == 9

    text = str(verbose)
    assert text.startswith("_Verbose:")
    assert "values:" in text


def test_backend_context_helpers_cover_current_and_operations(
    monkeypatch,
):
    import base_attentive.resolver.backend_context as bc
    from base_attentive.registry.capabilities import (
        BackendCapabilityReport,
    )

    assert bc._safe_import("definitely_missing_xyz") is None
    assert bc._first_available(None, "x", "y") == "x"
    assert bc._resolve_native_runtime(
        SimpleNamespace(tf="tf"),
        "tensorflow",
    ) == "tf"
    assert bc._resolve_native_runtime(
        SimpleNamespace(jax="jax"),
        "jax",
    ) == "jax"
    assert bc._resolve_native_runtime(
        SimpleNamespace(torch="torch"),
        "torch",
    ) == "torch"

    class _ImportAttr:
        @property
        def broken(self):
            raise ImportError("boom")

    assert bc._resolve_optional_attr(None, "x") is None
    assert (
        bc._resolve_optional_attr(_ImportAttr(), "broken") is None
    )

    report = BackendCapabilityReport(
        name="torch",
        framework="torch",
        available=True,
        uses_keras_runtime=True,
        experimental=False,
        supports_base_attentive=True,
        supports_base_attentive_v2=True,
    )

    fake_debug = SimpleNamespace(
        assert_equal=lambda actual, expected, message="": (
            actual,
            expected,
            message,
        )
    )
    fake_ops = SimpleNamespace(
        convert_to_tensor=lambda value, dtype=None: (
            "tensor",
            value,
            dtype,
        ),
        concatenate=lambda values, axis=-1: (
            "concat",
            values,
            axis,
        ),
        shape=lambda value: ("shape", value),
        reshape=lambda value, shape: (
            "reshape",
            value,
            shape,
        ),
        expand_dims=lambda value, axis=-1: (
            "expand",
            value,
            axis,
        ),
        tile=lambda value, reps: ("tile", value, reps),
        mean=lambda value, axis=None, keepdims=False: (
            "mean",
            value,
            axis,
            keepdims,
        ),
        ones=lambda shape, dtype=None: np.ones(shape, dtype=dtype),
    )
    fake_keras = SimpleNamespace(
        ops=fake_ops,
        random="keras-random",
    )
    fake_runtime = SimpleNamespace(
        keras=fake_keras,
        torch=SimpleNamespace(
            linalg="native-linalg",
            debugging=fake_debug,
        ),
        layers=SimpleNamespace(Dense="rt-dense"),
        Tensor="TensorType",
        Layer="LayerType",
        Model="ModelType",
        Sequential="SeqType",
    )
    fake_deps = SimpleNamespace(
        Dense="dep-dense",
        Layer="dep-layer",
        Model="dep-model",
        Sequential="dep-seq",
        random="dep-random",
        linalg="dep-linalg",
        debugging=fake_debug,
        concatenate=lambda values, axis=-1: (
            "dep-concat",
            values,
            axis,
        ),
        shape=lambda value: ("dep-shape", value),
        reshape=lambda value, shape: (
            "dep-reshape",
            value,
            shape,
        ),
        expand_dims=lambda value, axis=-1: (
            "dep-expand",
            value,
            axis,
        ),
        tile=lambda value, reps: ("dep-tile", value, reps),
        reduce_mean=lambda value, axis=None, keepdims=False: (
            "dep-mean",
            value,
            axis,
            keepdims,
        ),
        ones=lambda shape, dtype=None: np.ones(shape, dtype=dtype),
    )
    monkeypatch.setattr(bc, "KERAS_BACKEND", "torch")
    monkeypatch.setattr(bc, "KERAS_DEPS", fake_deps)
    monkeypatch.setattr(
        bc,
        "get_backend_capability_report",
        lambda name: report,
    )
    monkeypatch.setattr(
        bc, "get_backend", lambda name: fake_runtime
    )
    monkeypatch.setattr(bc, "_safe_import", lambda name: None)

    ctx = bc.BackendContext.current("torch")
    assert ctx.name == "torch"
    assert ctx.framework == "torch"
    assert ctx.layers.Dense == "rt-dense"
    assert ctx.native.linalg == "native-linalg"
    assert ctx.Tensor == "TensorType"
    assert ctx.Layer == "LayerType"
    assert ctx.Model == "ModelType"
    assert ctx.Sequential == "SeqType"

    ctx.require("layers.Dense", "Model")
    assert ctx._resolve_attr_path("layers.Dense") == "rt-dense"
    assert ctx.convert_to_tensor(1, dtype="float32") == (
        "tensor",
        1,
        "float32",
    )
    assert ctx.concatenate([1, 2], axis=0) == (
        "concat",
        [1, 2],
        0,
    )
    assert ctx.concat([1, 2], axis=1) == (
        "concat",
        [1, 2],
        1,
    )
    assert ctx.shape("x") == ("shape", "x")
    assert ctx.reshape("x", (2, 3)) == (
        "reshape",
        "x",
        (2, 3),
    )
    assert ctx.expand_dims("x", axis=2) == (
        "expand",
        "x",
        2,
    )
    assert ctx.tile("x", [1, 2]) == ("tile", "x", [1, 2])
    assert ctx.mean("x", axis=1, keepdims=True) == (
        "mean",
        "x",
        1,
        True,
    )
    assert ctx.ones((2, 2), dtype=np.float32).shape == (2, 2)
    np.testing.assert_array_equal(
        ctx.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    assert ctx.assert_equal(1, 1, message="ok") == (
        1,
        1,
        "ok",
    )

    missing = bc.BackendContext(
        name="demo",
        framework="demo",
        capability_report=report,
        runtime=None,
        keras=None,
        ops=SimpleNamespace(),
        layers=SimpleNamespace(),
        random=None,
        linalg=None,
        debugging=None,
        keras_deps=SimpleNamespace(),
    )
    with pytest.raises(ImportError, match="missing required"):
        missing.require("layers.Dense")
    with pytest.raises(ImportError, match="convert_to_tensor"):
        missing.convert_to_tensor(1)
    with pytest.raises(ImportError, match="concatenate"):
        missing.concatenate([1, 2])
    with pytest.raises(ImportError, match="shape is unavailable"):
        missing.shape(1)
    with pytest.raises(ImportError, match="reshape is unavailable"):
        missing.reshape(1, (1,))
    with pytest.raises(
        ImportError, match="expand_dims is unavailable"
    ):
        missing.expand_dims(1)
    with pytest.raises(ImportError, match="tile is unavailable"):
        missing.tile(1, [1])
    with pytest.raises(ImportError, match="mean is unavailable"):
        missing.mean(1)
    with pytest.raises(ImportError, match="ones is unavailable"):
        missing.ones((1,))
    with pytest.raises(ImportError, match="zeros is unavailable"):
        missing.zeros((1,))
    with pytest.raises(
        ImportError, match="debugging.assert_equal"
    ):
        missing.assert_equal(1, 1)
