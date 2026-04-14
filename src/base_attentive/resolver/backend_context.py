"""Backend context objects for resolver-driven assembly."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .. import KERAS_BACKEND, KERAS_DEPS
from ..backend import get_backend, normalize_backend_name
from ..registry import (
    BackendCapabilityReport,
    get_backend_capability_report,
)

_LAYER_ATTRS = (
    "Add",
    "BatchNormalization",
    "Concatenate",
    "Dense",
    "Dropout",
    "Embedding",
    "Flatten",
    "Input",
    "LSTM",
    "Layer",
    "LayerNormalization",
    "Model",
    "MultiHeadAttention",
    "Sequential",
    "Softmax",
    "TimeDistributed",
)


def _safe_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _resolve_optional_attr(
    namespace: Any, name: str
) -> Any | None:
    if namespace is None:
        return None
    try:
        return getattr(namespace, name)
    except (AttributeError, ImportError):
        return None


def _first_available(*values: Any) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_native_runtime(
    runtime: Any | None,
    framework: str,
) -> Any | None:
    if runtime is None:
        return None

    normalized_framework = normalize_backend_name(framework)
    if normalized_framework == "tensorflow":
        return getattr(runtime, "tf", None)
    if normalized_framework == "jax":
        return getattr(runtime, "jax", None)
    if normalized_framework == "torch":
        return getattr(runtime, "torch", None)
    return None


def _build_layers_namespace(
    runtime: Any | None,
    keras_deps: Any,
) -> SimpleNamespace:
    runtime_layers = getattr(runtime, "layers", None)
    values: dict[str, Any] = {}

    for attr in _LAYER_ATTRS:
        values[attr] = _first_available(
            _resolve_optional_attr(runtime_layers, attr),
            _resolve_optional_attr(runtime, attr),
            _resolve_optional_attr(keras_deps, attr),
        )

    return SimpleNamespace(**values)


@dataclass(frozen=True)
class BackendContext:
    """Runtime context used by the resolver and builders."""

    name: str
    framework: str
    capability_report: BackendCapabilityReport
    runtime: Any | None
    keras: Any | None
    ops: Any
    layers: Any
    random: Any | None
    linalg: Any | None
    debugging: Any | None
    native: Any | None = None
    Tensor: Any = object
    Layer: Any = object
    Model: Any = object
    Sequential: Any = object
    uses_keras_runtime: bool = True
    experimental: bool = False
    keras_deps: Any = KERAS_DEPS

    @classmethod
    def current(
        cls, name: str | None = None
    ) -> "BackendContext":
        normalized_name = normalize_backend_name(
            name or KERAS_BACKEND
        )
        capability_report = get_backend_capability_report(
            normalized_name
        )

        runtime = None
        try:
            runtime = get_backend(normalized_name)
        except Exception:
            runtime = None

        keras_runtime = _first_available(
            getattr(runtime, "keras", None),
            _safe_import("keras"),
        )
        native_runtime = _resolve_native_runtime(
            runtime,
            capability_report.framework,
        )
        ops = _first_available(
            _resolve_optional_attr(keras_runtime, "ops"),
            KERAS_DEPS,
        )
        layers = _build_layers_namespace(runtime, KERAS_DEPS)
        random = _first_available(
            _resolve_optional_attr(keras_runtime, "random"),
            _resolve_optional_attr(native_runtime, "random"),
            _resolve_optional_attr(KERAS_DEPS, "random"),
        )
        linalg = _first_available(
            _resolve_optional_attr(native_runtime, "linalg"),
            _resolve_optional_attr(KERAS_DEPS, "linalg"),
        )
        debugging = _first_available(
            _resolve_optional_attr(
                native_runtime, "debugging"
            ),
            _resolve_optional_attr(KERAS_DEPS, "debugging"),
        )

        return cls(
            name=normalized_name,
            framework=capability_report.framework,
            capability_report=capability_report,
            runtime=runtime,
            keras=keras_runtime,
            ops=ops,
            layers=layers,
            random=random,
            linalg=linalg,
            debugging=debugging,
            native=native_runtime,
            Tensor=_first_available(
                getattr(runtime, "Tensor", None),
                _resolve_optional_attr(KERAS_DEPS, "Tensor"),
                object,
            ),
            Layer=_first_available(
                getattr(runtime, "Layer", None),
                _resolve_optional_attr(layers, "Layer"),
                object,
            ),
            Model=_first_available(
                getattr(runtime, "Model", None),
                _resolve_optional_attr(layers, "Model"),
                object,
            ),
            Sequential=_first_available(
                getattr(runtime, "Sequential", None),
                _resolve_optional_attr(layers, "Sequential"),
                object,
            ),
            uses_keras_runtime=bool(
                capability_report.uses_keras_runtime
            ),
            experimental=bool(capability_report.experimental),
            keras_deps=KERAS_DEPS,
        )

    def require(self, *attrs: str) -> None:
        missing = [
            attr
            for attr in attrs
            if self._resolve_attr_path(attr) is None
        ]
        if missing:
            joined = ", ".join(missing)
            raise ImportError(
                f"Backend {self.name!r} is missing required "
                f"runtime surfaces: {joined}."
            )

    def _resolve_attr_path(self, path: str) -> Any | None:
        current: Any = self
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    def convert_to_tensor(
        self,
        value: Any,
        dtype: Any | None = None,
    ) -> Any:
        fn = _first_available(
            _resolve_optional_attr(
                self.ops, "convert_to_tensor"
            ),
            _resolve_optional_attr(self.ops, "constant"),
        )
        if fn is None:
            raise ImportError(
                "convert_to_tensor is unavailable for the active "
                f"backend {self.name!r}."
            )
        if dtype is None:
            return fn(value)
        return fn(value, dtype=dtype)

    def concatenate(self, values: Any, axis: int = -1) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "concatenate"),
            _resolve_optional_attr(self.ops, "concat"),
            _resolve_optional_attr(
                self.keras_deps, "concatenate"
            ),
            _resolve_optional_attr(self.keras_deps, "concat"),
        )
        if fn is None:
            raise ImportError(
                "concatenate/concat is unavailable for the active "
                f"backend {self.name!r}."
            )
        return fn(values, axis=axis)

    def concat(self, values: Any, axis: int = -1) -> Any:
        return self.concatenate(values, axis=axis)

    def shape(self, value: Any) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "shape"),
            _resolve_optional_attr(self.keras_deps, "shape"),
        )
        if fn is None:
            raise ImportError(
                f"shape is unavailable for backend {self.name!r}."
            )
        return fn(value)

    def reshape(self, value: Any, shape: Any) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "reshape"),
            _resolve_optional_attr(
                self.keras_deps, "reshape"
            ),
        )
        if fn is None:
            raise ImportError(
                f"reshape is unavailable for backend {self.name!r}."
            )
        return fn(value, shape)

    def expand_dims(self, value: Any, axis: int = -1) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "expand_dims"),
            _resolve_optional_attr(
                self.keras_deps, "expand_dims"
            ),
        )
        if fn is None:
            raise ImportError(
                "expand_dims is unavailable for the active "
                f"backend {self.name!r}."
            )
        return fn(value, axis=axis)

    def tile(self, value: Any, reps: Any) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "tile"),
            _resolve_optional_attr(self.keras_deps, "tile"),
        )
        if fn is None:
            raise ImportError(
                f"tile is unavailable for backend {self.name!r}."
            )
        return fn(value, reps)

    def mean(
        self,
        value: Any,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "mean"),
            _resolve_optional_attr(self.ops, "reduce_mean"),
            _resolve_optional_attr(self.keras_deps, "mean"),
            _resolve_optional_attr(
                self.keras_deps, "reduce_mean"
            ),
        )
        if fn is None:
            raise ImportError(
                f"mean is unavailable for backend {self.name!r}."
            )
        return fn(value, axis=axis, keepdims=keepdims)

    def ones(
        self,
        shape: Any,
        dtype: Any | None = None,
    ) -> Any:
        fn = _first_available(
            _resolve_optional_attr(self.ops, "ones"),
            _resolve_optional_attr(self.keras_deps, "ones"),
        )
        if fn is None:
            raise ImportError(
                f"ones is unavailable for backend {self.name!r}."
            )
        if dtype is None:
            return fn(shape)
        return fn(shape, dtype=dtype)

    def zeros(
        self,
        shape: Any,
        dtype: Any | None = None,
    ) -> Any:
        fn = _resolve_optional_attr(self.ops, "zeros")
        if fn is not None:
            if dtype is None:
                return fn(shape)
            return fn(shape, dtype=dtype)

        ones_fn = _resolve_optional_attr(self.ops, "ones")
        if ones_fn is None:
            raise ImportError(
                f"zeros is unavailable for backend {self.name!r}."
            )
        if dtype is None:
            return ones_fn(shape) * 0
        return ones_fn(shape, dtype=dtype) * 0

    def assert_equal(
        self,
        actual: Any,
        expected: Any,
        *,
        message: str = "",
    ) -> Any:
        fn = _resolve_optional_attr(
            self.debugging, "assert_equal"
        )
        if fn is None:
            raise ImportError(
                "debugging.assert_equal is unavailable for the "
                f"active backend {self.name!r}."
            )
        return fn(actual, expected, message=message)


__all__ = ["BackendContext"]
