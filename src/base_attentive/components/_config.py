# SPDX-License-Identifier: BSD-3-Clause

# BSD-3-Clause license applies to this file.
# See: third_party/licenses/fusionlab-learn_BSD-3-Clause.txt

"""Centralized Keras symbols and backend-neutral ops for components.

This module is intentionally backend-neutral. Component code should
import neutral aliases such as ``shape`` and ``concat`` instead of
TensorFlow-shaped helper names.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np

from ..keras_runtime import (
    KERAS_BACKEND,
    KERAS_DEPS,
    dependency_message,
    get_layer_class,
)
from ..compat.keras import import_keras_attr
from ..logging import get_logger


def _resolve_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return None
    return getattr(value, "value", None)


def _dep(name: str, fallback: Any = None) -> Any:
    try:
        return getattr(KERAS_DEPS, name)
    except ImportError:
        if fallback is not None:
            return fallback
        raise


def _constant(value, dtype=None):
    return np.asarray(value, dtype=dtype)


def _convert_to_tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype)


def _zeros(shape, dtype=None):
    normalized_shape = shape
    if not isinstance(shape, tuple):
        normalized_shape = tuple(np.asarray(shape).tolist())
    return np.zeros(normalized_shape, dtype=dtype)


def _assert_equal(actual, expected, message="", name=None):
    del name
    actual_value = _resolve_scalar(actual)
    expected_value = _resolve_scalar(expected)
    if actual_value is not None and expected_value is not None:
        if actual_value != expected_value:
            raise AssertionError(
                message or f"{actual_value} != {expected_value}"
            )
    return None


def _shape(value):
    return np.asarray(value).shape


def _reshape(value, new_shape):
    return np.reshape(value, new_shape)


def _concat(values, axis=0):
    return np.concatenate(values, axis=axis)


def _stack(values, axis=0):
    return np.stack(values, axis=axis)


def _unstack(values, axis=0):
    arr = np.asarray(values)
    return [
        np.take(arr, i, axis=axis)
        for i in range(arr.shape[axis])
    ]


def _expand_dims(value, axis=0):
    return np.expand_dims(value, axis=axis)


def _tile(value, multiples):
    return np.tile(value, multiples)


def _repeat(value, repeats, axis=None):
    return np.repeat(value, repeats, axis=axis)


def _cast(value, dtype, **kwargs):  # noqa: ARG001
    return np.asarray(value, dtype=dtype)


def _add_n(values):
    arrays = [np.asarray(value) for value in values]
    return sum(arrays[1:], arrays[0]) if arrays else 0


def _cond(pred, true_fn, false_fn):
    return true_fn() if pred else false_fn()


def _get_static_value(value):
    return _resolve_scalar(value)


def _softplus(value):
    arr = np.asarray(value)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0)


def _reduce_logsumexp(value, axis=None, keepdims=False):
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    arr = np.asarray(value)
    max_value = np.max(arr, axis=axis, keepdims=True)
    stable = np.log(
        np.sum(
            np.exp(arr - max_value), axis=axis, keepdims=True
        )
    )
    result = stable + max_value
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


try:
    activations = KERAS_DEPS.activations
except (ImportError, AttributeError) as exc:
    try:
        activations = import_keras_attr("activations")
    except Exception as keras_exc:
        raise ImportError(str(exc)) from keras_exc


def _instantiate_layer(target, *args, **kwargs):
    try:
        parameters = inspect.signature(target).parameters
    except Exception:
        parameters = {}
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not accepts_var_kwargs and parameters:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }
    return target(*args, **kwargs)


_RAW_LSTM = _dep("LSTM")
_RAW_LAYERNORM = _dep("LayerNormalization")
_RAW_TIMEDISTRIBUTED = _dep("TimeDistributed")
_RAW_MHA = _dep("MultiHeadAttention")
Model = _dep("Model")
_RAW_BATCHNORM = _dep("BatchNormalization")
_RAW_INPUT = _dep("Input")
_RAW_SOFTMAX = _dep("Softmax")
_RAW_FLATTEN = _dep("Flatten")
_RAW_DROPOUT = _dep("Dropout")
_RAW_DENSE = _dep("Dense")
_RAW_EMBEDDING = _dep("Embedding")
_RAW_CONCATENATE = _dep("Concatenate")
Layer = _dep("Layer")


def LSTM(*args, **kwargs):
    return _instantiate_layer(_RAW_LSTM, *args, **kwargs)


def LayerNormalization(*args, **kwargs):
    return _instantiate_layer(_RAW_LAYERNORM, *args, **kwargs)


def TimeDistributed(*args, **kwargs):
    return _instantiate_layer(
        _RAW_TIMEDISTRIBUTED, *args, **kwargs
    )


def MultiHeadAttention(*args, **kwargs):
    return _instantiate_layer(_RAW_MHA, *args, **kwargs)


def BatchNormalization(*args, **kwargs):
    return _instantiate_layer(_RAW_BATCHNORM, *args, **kwargs)


def Input(*args, **kwargs):
    return _instantiate_layer(_RAW_INPUT, *args, **kwargs)


def Softmax(*args, **kwargs):
    return _instantiate_layer(_RAW_SOFTMAX, *args, **kwargs)


def Flatten(*args, **kwargs):
    return _instantiate_layer(_RAW_FLATTEN, *args, **kwargs)


def Dropout(*args, **kwargs):
    return _instantiate_layer(_RAW_DROPOUT, *args, **kwargs)


def Dense(*args, **kwargs):
    return _instantiate_layer(_RAW_DENSE, *args, **kwargs)


def Embedding(*args, **kwargs):
    return _instantiate_layer(_RAW_EMBEDDING, *args, **kwargs)


def Concatenate(*args, **kwargs):
    return _instantiate_layer(
        _RAW_CONCATENATE, *args, **kwargs
    )


if Layer is object:
    Layer = get_layer_class()
Loss = _dep("Loss")
Tensor = _dep("Tensor", np.ndarray)
Sequential = _dep("Sequential")
Constant = _dep("Constant", _constant)
TensorShape = _dep("TensorShape", tuple)
Reduction = _dep("Reduction")

register_keras_serializable = _dep(
    "register_keras_serializable"
)
get_loss = _dep("get")

# Backend-neutral op aliases.
assert_op = _dep(
    "Assert",
    lambda condition,
    data=None,
    summarize=None,
    name=None: condition,
)
concat = _dep("concat", _concat)
shape = _dep("shape", _shape)
reshape = _dep("reshape", _reshape)
repeat = _dep("repeat", _repeat)
add = _dep("add", np.add)
cast = _dep("cast", _cast)
maximum = _dep("maximum", np.maximum)
reduce_mean = _dep("reduce_mean", np.mean)
add_n = _dep("add_n", _add_n)
float32 = _dep("float32", np.float32)
constant = _dep("constant", _constant)
convert_to_tensor = _dep(
    "convert_to_tensor", _convert_to_tensor
)
zeros = _dep("zeros", _zeros)
square = _dep("square", np.square)
transpose = _dep("transpose", np.transpose)
logical_and = _dep("logical_and", np.logical_and)
logical_not = _dep("logical_not", np.logical_not)
logical_or = _dep("logical_or", np.logical_or)
get_static_value = _dep("get_static_value", _get_static_value)
reduce_sum = _dep("reduce_sum", np.sum)
stack = _dep("stack", _stack)
unstack = _dep("unstack", _unstack)
expand_dims = _dep("expand_dims", _expand_dims)
tile = _dep("tile", _tile)
where = _dep("where", np.where)
range = _dep("range", np.arange)
rank = _dep("rank", np.ndim)
split = _dep("split", np.split)
multiply = _dep("multiply", np.multiply)
cond = _dep("cond", _cond)
equal = _dep("equal", np.equal)
int32 = _dep("int32", np.int32)
debugging = _dep("debugging")
autograph = _dep("autograph")
assert_equal = getattr(debugging, "assert_equal", _assert_equal)
do_not_convert = getattr(
    getattr(autograph, "experimental", object()),
    "do_not_convert",
    lambda func=None, **kwargs: (
        (lambda inner: inner)
        if func is None
        else func
    ),
)
pad = _dep("pad", np.pad)
ones_like = _dep("ones_like", np.ones_like)
bool_dtype = _dep("bool", np.bool_)
newaxis = _dep("newaxis", None)
abs_op = _dep("abs", np.abs)
power = _dep("pow", np.power)
pow_op = power
sin = _dep("sin", np.sin)
cos = _dep("cos", np.cos)
exp = _dep("exp", np.exp)
log = _dep("log", np.log)
sigmoid = _dep(
    "sigmoid", lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x)))
)
cumsum = _dep("cumsum", np.cumsum)
gather = _dep("gather", np.take)
random = _dep("random", np.random)
softplus = _dep("softplus", _softplus)
reduce_logsumexp = _dep("reduce_logsumexp", _reduce_logsumexp)
sqrt = _dep("sqrt", np.sqrt)
erf = _dep("erf", np.vectorize(math.erf))
ones = _dep("ones", np.ones)
linalg = _dep("linalg", np.linalg)
floordiv = _dep("floordiv", np.floor_divide)
greater = _dep("greater", np.greater)
reduce_max = _dep("reduce_max", np.max)
subtract = _dep("subtract", np.subtract)

_logger = get_logger(__name__)
DEP_MSG = dependency_message("nn.components")

__all__ = [
    "activations",
    "LSTM",
    "LayerNormalization",
    "TimeDistributed",
    "MultiHeadAttention",
    "Model",
    "BatchNormalization",
    "Input",
    "Softmax",
    "Flatten",
    "Dropout",
    "Dense",
    "Embedding",
    "Concatenate",
    "Layer",
    "Loss",
    "Tensor",
    "Sequential",
    "TensorShape",
    "Constant",
    "Reduction",
    "register_keras_serializable",
    "get_loss",
    "assert_op",
    "concat",
    "shape",
    "reshape",
    "repeat",
    "add",
    "cast",
    "maximum",
    "reduce_mean",
    "add_n",
    "float32",
    "constant",
    "convert_to_tensor",
    "zeros",
    "assert_equal",
    "square",
    "transpose",
    "logical_and",
    "logical_not",
    "logical_or",
    "get_static_value",
    "reduce_sum",
    "stack",
    "unstack",
    "expand_dims",
    "tile",
    "where",
    "range",
    "rank",
    "split",
    "multiply",
    "cond",
    "equal",
    "int32",
    "debugging",
    "autograph",
    "do_not_convert",
    "pad",
    "ones_like",
    "bool_dtype",
    "newaxis",
    "abs_op",
    "power",
    "pow_op",
    "sin",
    "cos",
    "exp",
    "log",
    "sigmoid",
    "cumsum",
    "gather",
    "random",
    "softplus",
    "reduce_logsumexp",
    "sqrt",
    "erf",
    "ones",
    "linalg",
    "floordiv",
    "greater",
    "reduce_max",
    "subtract",
    "_logger",
    "DEP_MSG",
    "KERAS_DEPS",
    "KERAS_BACKEND",
]
