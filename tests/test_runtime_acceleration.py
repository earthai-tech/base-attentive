"""Tests for runtime acceleration helpers."""

from __future__ import annotations

import sys
import types

import pytest


class _FakeTensorFlow(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.function_calls = []

    def function(self, **kwargs):
        self.function_calls.append(kwargs)

        def decorator(func):
            def wrapped(inputs):
                return func(inputs)

            wrapped._tf_function_kwargs = kwargs
            return wrapped

        return decorator


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, training=False):
        self.calls.append((inputs, training))
        return {"inputs": inputs, "training": training}


def test_make_fast_predict_fn_wraps_model_with_tf_function(
    monkeypatch,
):
    """The helper should build a traced inference callable."""
    from base_attentive import runtime

    fake_tf = _FakeTensorFlow()
    fake_model = _FakeModel()

    monkeypatch.setattr(
        runtime, "KERAS_BACKEND", "tensorflow"
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    fast_predict = runtime.make_fast_predict_fn(
        fake_model,
        jit_compile=False,
        reduce_retracing=False,
        warmup_inputs=("warmup",),
    )

    assert fake_tf.function_calls == [
        {"jit_compile": False, "reduce_retracing": False}
    ]
    assert fake_model.calls == [(("warmup",), False)]

    output = fast_predict(["batch"])

    assert output == {"inputs": ["batch"], "training": False}
    assert fake_model.calls[-1] == (["batch"], False)


def test_make_fast_predict_fn_requires_tensorflow_backend(
    monkeypatch,
):
    """The helper should fail fast on unsupported backends."""
    from base_attentive import runtime

    monkeypatch.setattr(runtime, "KERAS_BACKEND", "torch")

    with pytest.raises(
        RuntimeError, match="requires the TensorFlow backend"
    ):
        runtime.make_fast_predict_fn(_FakeModel())


def test_make_fast_predict_fn_raises_helpful_error_when_tf_missing(
    monkeypatch,
):
    """Missing TensorFlow should produce a clear import error."""
    from base_attentive import runtime

    monkeypatch.setattr(
        runtime, "KERAS_BACKEND", "tensorflow"
    )
    monkeypatch.delitem(
        sys.modules, "tensorflow", raising=False
    )

    def _missing_tensorflow(name):
        if name == "tensorflow":
            raise ImportError("tensorflow missing")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        _missing_tensorflow,
    )

    with pytest.raises(
        ImportError, match="TensorFlow is required"
    ):
        runtime.make_fast_predict_fn(_FakeModel())
