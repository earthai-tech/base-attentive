"""Lightweight NumPy-backed fallbacks for Keras-style symbols.

These helpers are intentionally small and only aim to support the test and
graceful-degradation paths exercised by ``base_attentive`` when a full Keras
runtime is unavailable.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


def _to_numpy_dtype(dtype):
    if isinstance(dtype, DTypeProxy):
        return dtype.as_numpy_dtype
    if hasattr(dtype, "as_numpy_dtype"):
        return dtype.as_numpy_dtype
    if dtype is None:
        return np.float32
    return np.dtype(dtype).type


def _normalize_shape(shape) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, np.ndarray):
        return tuple(int(v) for v in shape.tolist())
    if isinstance(shape, (list, tuple)):
        return tuple(int(v) for v in shape)
    return (int(shape),)


def _ensure_array(value, dtype=None):
    np_dtype = _to_numpy_dtype(dtype) if dtype is not None else None
    return np.asarray(value, dtype=np_dtype)


class DTypeProxy:
    """TensorFlow-like dtype proxy exposing ``as_numpy_dtype``."""

    def __init__(self, dtype):
        self.as_numpy_dtype = np.dtype(dtype).type

    def __call__(self, value):
        return self.as_numpy_dtype(value)

    def __repr__(self):
        return repr(self.as_numpy_dtype)


float32 = DTypeProxy(np.float32)
int32 = DTypeProxy(np.int32)
bool = DTypeProxy(np.bool_)


class TensorShape(tuple):
    """Small TensorShape-compatible tuple wrapper."""

    def __new__(cls, value=()):
        if value is None:
            value = ()
        if isinstance(value, TensorShape):
            return value
        return super().__new__(cls, tuple(value))

    @property
    def rank(self):
        return len(self)

    def as_list(self):
        return list(self)

    def concatenate(self, values):
        return TensorShape(tuple(self) + tuple(values))


class Variable:
    """Minimal variable wrapper supporting array coercion and ``assign``."""

    __array_priority__ = 1000

    def __init__(self, value):
        self.value = np.asarray(value)

    def assign(self, value):
        self.value = np.asarray(value, dtype=self.value.dtype)
        return self

    def __array__(self, dtype=None):
        if dtype is None:
            return self.value
        return self.value.astype(dtype)

    def __getitem__(self, item):
        return self.value[item]


Tensor = np.ndarray


def _infer_input_shape(value):
    if isinstance(value, (list, tuple)):
        return [_infer_input_shape(item) for item in value]
    return TensorShape(getattr(np.asarray(value), "shape", ()))


def _initialize_value(initializer, shape, dtype):
    np_dtype = _to_numpy_dtype(dtype)
    shape = _normalize_shape(shape)

    if isinstance(initializer, Constant):
        value = initializer(shape=shape, dtype=np_dtype)
    elif isinstance(initializer, str):
        if initializer == "zeros":
            value = np.zeros(shape, dtype=np_dtype)
        elif initializer == "ones":
            value = np.ones(shape, dtype=np_dtype)
        else:
            value = np.zeros(shape, dtype=np_dtype)
    elif callable(initializer):
        try:
            value = initializer(shape, np_dtype)
        except TypeError:
            value = initializer(shape)
        value = np.asarray(value, dtype=np_dtype)
        if value.shape == () and shape:
            value = np.full(shape, value.item(), dtype=np_dtype)
    elif initializer is None:
        value = np.zeros(shape, dtype=np_dtype)
    else:
        value = np.asarray(initializer, dtype=np_dtype)
        if value.shape == () and shape:
            value = np.full(shape, value.item(), dtype=np_dtype)

    if value.shape != shape and shape:
        value = np.broadcast_to(value, shape).copy()
    return Variable(value)


class Layer:
    """Minimal Keras-like layer base class."""

    def __init__(
        self,
        *args,
        name=None,
        dtype=float32,
        trainable=True,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.dtype = dtype
        self.trainable = trainable
        self.built = False
        self._init_kwargs = dict(kwargs)

    def add_weight(
        self,
        name=None,
        shape=None,
        initializer="zeros",
        trainable=True,
        dtype=float32,
        **kwargs,
    ):
        return _initialize_value(initializer, shape, dtype)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        return inputs

    def __call__(self, *args, **kwargs):
        if not self.built and args:
            try:
                self.build(_infer_input_shape(args[0]))
            except Exception:
                self.built = True
        return self.call(*args, **kwargs)

    def get_config(self):
        return {"name": self.name}


class Loss(Layer):
    def __init__(self, reduction="auto", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduction = reduction

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config


class Constant:
    def __init__(self, value):
        self.value = value

    def __call__(self, shape=None, dtype=None):
        array = np.asarray(self.value, dtype=_to_numpy_dtype(dtype))
        if shape:
            array = np.broadcast_to(
                array, _normalize_shape(shape)
            ).copy()
        return array


class Dense(Layer):
    def __init__(
        self, units, activation=None, use_bias=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def call(self, inputs, training=False):
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim == 0:
            x = x.reshape(1)
        out_shape = x.shape[:-1] + (self.units,)
        base = np.mean(x, axis=-1, keepdims=True)
        out = np.broadcast_to(base, out_shape).astype(
            np.float32, copy=True
        )
        return self.activation(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"units": self.units, "use_bias": self.use_bias}
        )
        return config


class Dropout(Layer):
    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=False):
        return np.asarray(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=False):
        x = np.asarray(inputs, dtype=np.float32)
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class BatchNormalization(LayerNormalization):
    pass


class Softmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, training=False):
        x = np.asarray(inputs, dtype=np.float32)
        shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs, training=False):
        x = np.asarray(inputs)
        return np.zeros(
            x.shape + (self.output_dim,), dtype=np.float32
        )


class Flatten(Layer):
    def call(self, inputs, training=False):
        x = np.asarray(inputs)
        return (
            x.reshape((x.shape[0], -1))
            if x.ndim > 1
            else x.reshape((1, -1))
        )


class Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, training=False):
        return np.concatenate(
            [np.asarray(x) for x in inputs], axis=self.axis
        )


class TimeDistributed(Layer):
    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, training=False):
        x = np.asarray(inputs)
        if x.ndim < 3:
            return self.layer(x)
        outputs = [self.layer(x[:, i, :]) for i in range(x.shape[1])]
        return np.stack(outputs, axis=1)


class Sequential(Layer):
    def __init__(self, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.layers = list(layers or [])

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            try:
                x = layer(x, training=training)
            except TypeError:
                x = layer(x)
        return x


class LSTM(Layer):
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.return_sequences = return_sequences
        self.proj = Dense(units)

    def call(self, inputs, training=False):
        x = np.asarray(inputs, dtype=np.float32)
        if self.return_sequences:
            return self.proj(x)
        if x.ndim >= 3:
            x = np.mean(x, axis=-2)
        return self.proj(x)


class MultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.dropout = dropout

    def call(self, *args, **kwargs):
        return_attention_scores = kwargs.pop(
            "return_attention_scores", False
        )
        query = kwargs.get("query")
        value = kwargs.get("value")
        key = kwargs.get("key")

        if query is None and args:
            query = args[0]
        if value is None:
            if len(args) >= 2:
                value = args[1]
            else:
                value = query
        if key is None:
            if len(args) >= 3:
                key = args[2]
            else:
                key = value

        query = np.asarray(query, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        output = query
        if value.shape == query.shape:
            output = 0.5 * (query + value)

        if not return_attention_scores:
            return output

        if query.ndim >= 3 and value.ndim >= 3:
            batch, tq, tv = (
                query.shape[0],
                query.shape[-2],
                value.shape[-2],
            )
        else:
            batch, tq, tv = 1, 1, 1
        scores = np.zeros(
            (batch, self.num_heads, tq, tv), dtype=np.float32
        )
        return output, scores


class Model(Layer):
    pass


def Input(shape=None, dtype=float32, **kwargs):
    return np.zeros(
        _normalize_shape(shape), dtype=_to_numpy_dtype(dtype)
    )


def register_keras_serializable(package="Custom", name=None):
    def decorator(cls):
        return cls

    return decorator


def get(obj):
    return obj


class _Activations:
    def get(self, activation):
        if activation is None or activation == "linear":
            return lambda x: np.asarray(x)
        if callable(activation):
            return activation
        name = str(activation).lower()
        if name == "relu":
            return lambda x: np.maximum(np.asarray(x), 0.0)
        if name == "sigmoid":
            return lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x)))
        if name == "tanh":
            return lambda x: np.tanh(np.asarray(x))
        if name == "elu":
            return lambda x: np.where(
                np.asarray(x) > 0,
                np.asarray(x),
                np.exp(np.asarray(x)) - 1.0,
            )
        if name == "selu":
            scale = 1.0507009873554805
            alpha = 1.6732632423543772
            return lambda x: scale * np.where(
                np.asarray(x) > 0,
                np.asarray(x),
                alpha * (np.exp(np.asarray(x)) - 1.0),
            )
        if name == "gelu":
            return (
                lambda x: 0.5
                * np.asarray(x)
                * (1.0 + erf(np.asarray(x) / math.sqrt(2.0)))
            )
        raise ValueError(f"Unknown activation '{activation}'.")

    def serialize(self, activation):
        if isinstance(activation, str):
            return activation
        return getattr(
            activation, "__name__", activation.__class__.__name__
        )


activations = _Activations()


class _RandomNamespace:
    @staticmethod
    def uniform(shape, minval=0.0, maxval=1.0, dtype=float32):
        return np.random.uniform(
            minval, maxval, size=_normalize_shape(shape)
        ).astype(_to_numpy_dtype(dtype))

    @staticmethod
    def normal(shape, mean=0.0, stddev=1.0, dtype=float32):
        return np.random.normal(
            mean, stddev, size=_normalize_shape(shape)
        ).astype(_to_numpy_dtype(dtype))


random = _RandomNamespace()


class _LinalgNamespace:
    @staticmethod
    def band_part(x, num_lower, num_upper):
        arr = np.asarray(x)
        rows, cols = arr.shape[-2], arr.shape[-1]
        mask = np.zeros((rows, cols), dtype=bool)
        for i in range(rows):
            for j in range(cols):
                lower_ok = num_lower < 0 or i - j <= num_lower
                upper_ok = num_upper < 0 or j - i <= num_upper
                mask[i, j] = lower_ok and upper_ok
        return np.where(mask, arr, np.zeros_like(arr))


linalg = _LinalgNamespace()


Reduction = SimpleNamespace(AUTO="auto", SUM="sum", NONE="none")


def Assert(condition, data=None, summarize=None, name=None):
    return condition


class _AutographExperimental:
    @staticmethod
    def do_not_convert(func=None, **kwargs):
        if func is None:

            def decorator(inner):
                return inner

            return decorator
        return func


autograph = SimpleNamespace(experimental=_AutographExperimental())
debugging = SimpleNamespace(
    assert_equal=lambda actual, expected, message="", name=None: None
)


def constant(value, dtype=None):
    return _ensure_array(value, dtype)


def cast(value, dtype, **kwargs):
    return np.asarray(value, dtype=_to_numpy_dtype(dtype))


def shape(value):
    return np.asarray(np.shape(value), dtype=np.int32)


def reshape(value, new_shape):
    return np.reshape(
        np.asarray(value),
        tuple(int(v) for v in np.asarray(new_shape).tolist()),
    )


def repeat(value, repeats, axis=None):
    return np.repeat(np.asarray(value), repeats, axis=axis)


def add(x, y):
    return np.asarray(x) + np.asarray(y)


def maximum(x, y):
    return np.maximum(np.asarray(x), np.asarray(y))


def mean(x, axis=None, keepdims=False):
    return np.mean(np.asarray(x), axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
    return mean(x, axis=axis, keepdims=keepdims)


def add_n(values, **kwargs):
    arrays = [np.asarray(v) for v in values]
    result = np.zeros_like(arrays[0], dtype=np.result_type(*arrays))
    for array in arrays:
        result = result + array
    return result


def square(x):
    return np.square(np.asarray(x))


def transpose(x, perm=None):
    return np.transpose(np.asarray(x), axes=perm)


def logical_and(x, y):
    return np.logical_and(np.asarray(x), np.asarray(y))


def logical_not(x):
    return np.logical_not(np.asarray(x))


def logical_or(x, y):
    return np.logical_or(np.asarray(x), np.asarray(y))


def get_static_value(value):
    if isinstance(value, (int, float, bool, str)):
        return value
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return None


def reduce_sum(x, axis=None, keepdims=False):
    return np.sum(np.asarray(x), axis=axis, keepdims=keepdims)


def stack(values, axis=0):
    return np.stack([np.asarray(v) for v in values], axis=axis)


def unstack(value, axis=0):
    array = np.asarray(value)
    return [
        np.take(array, i, axis=axis) for i in range(array.shape[axis])
    ]


def expand_dims(value, axis=-1, **kwargs):
    return np.expand_dims(np.asarray(value), axis=axis)


def tile(value, reps):
    return np.tile(
        np.asarray(value),
        tuple(int(v) for v in np.asarray(reps).tolist()),
    )


def where(condition, x=None, y=None):
    if x is None and y is None:
        return np.argwhere(np.asarray(condition))
    return np.where(condition, x, y)


def range(start, limit=None, delta=1, dtype=None):
    if limit is None:
        start, limit = 0, start
    np_dtype = _to_numpy_dtype(dtype) if dtype is not None else None
    return np.arange(start, limit, delta, dtype=np_dtype)


def rank(value, **kwargs):
    return np.ndim(value)


def split(value, num_or_size_splits, axis=0):
    array = np.asarray(value)
    if isinstance(num_or_size_splits, int):
        return np.array_split(array, num_or_size_splits, axis=axis)
    indices = np.cumsum(num_or_size_splits)[:-1]
    return np.split(array, indices, axis=axis)


def multiply(x, y):
    return np.asarray(x) * np.asarray(y)


def cond(pred, true_fn, false_fn):
    return true_fn() if pred else false_fn()


def equal(x, y):
    return np.equal(np.asarray(x), np.asarray(y))


def pad(x, paddings, mode="constant", constant_values=0):
    return np.pad(
        np.asarray(x),
        paddings,
        mode=mode,
        constant_values=constant_values,
    )


def ones_like(x, dtype=None):
    return np.ones_like(
        np.asarray(x), dtype=_to_numpy_dtype(dtype) if dtype else None
    )


def abs(x):
    return np.abs(np.asarray(x))


def pow(x, y, **kwargs):
    return np.power(np.asarray(x), y)


def sin(x):
    return np.sin(np.asarray(x))


def cos(x):
    return np.cos(np.asarray(x))


def exp(x):
    return np.exp(np.asarray(x))


def log(x):
    return np.log(np.asarray(x))


def sigmoid(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def cumsum(x, axis=0):
    return np.cumsum(np.asarray(x), axis=axis)


def gather(params, indices, axis=0, batch_dims=0, **kwargs):
    params = np.asarray(params)
    indices = np.asarray(indices)
    if batch_dims == 1 and params.ndim >= 2:
        gathered = []
        flat_indices = indices.reshape(indices.shape[0], -1)
        for batch_idx, batch_indices in enumerate(flat_indices):
            taken = np.take(
                params[batch_idx],
                batch_indices,
                axis=axis - 1 if axis > 0 else axis,
            )
            gathered.append(taken)
        gathered = np.asarray(gathered)
        trailing = (
            params.shape[axis + 1 :] if axis + 1 < params.ndim else ()
        )
        return gathered.reshape(
            (params.shape[0],) + indices.shape[1:] + trailing
        )
    return np.take(params, indices, axis=axis)


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def reduce_logsumexp(x, axis=None, keepdims=False):
    array = np.asarray(x)
    max_x = np.max(array, axis=axis, keepdims=True)
    result = (
        np.log(
            np.sum(np.exp(array - max_x), axis=axis, keepdims=True)
        )
        + max_x
    )
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def sqrt(x):
    return np.sqrt(np.asarray(x))


def erf(x):
    vectorized = np.vectorize(math.erf)
    return vectorized(np.asarray(x))


def ones(shape, dtype=float32):
    return np.ones(
        _normalize_shape(shape), dtype=_to_numpy_dtype(dtype)
    )


def floordiv(x, y):
    return np.floor_divide(np.asarray(x), y)


def greater(x, y):
    return np.greater(np.asarray(x), np.asarray(y))


concatenate = concat = lambda values, axis=0: np.concatenate(
    [np.asarray(v) for v in values], axis=axis
)


layers = SimpleNamespace(
    Layer=Layer,
    Dense=Dense,
    Dropout=Dropout,
    LayerNormalization=LayerNormalization,
    BatchNormalization=BatchNormalization,
    Softmax=Softmax,
    Embedding=Embedding,
    Flatten=Flatten,
    Concatenate=Concatenate,
    MultiHeadAttention=MultiHeadAttention,
    LSTM=LSTM,
    TimeDistributed=TimeDistributed,
)

losses = SimpleNamespace(Loss=Loss, Reduction=Reduction, get=get)
saving = SimpleNamespace(
    register_keras_serializable=register_keras_serializable
)
utils = SimpleNamespace(
    register_keras_serializable=register_keras_serializable
)
ops = SimpleNamespace(
    concat=concat,
    concatenate=concatenate,
    shape=shape,
    reshape=reshape,
    repeat=repeat,
    add=add,
    cast=cast,
    maximum=maximum,
    mean=mean,
    reduce_mean=reduce_mean,
    add_n=add_n,
    constant=constant,
    square=square,
    transpose=transpose,
    logical_and=logical_and,
    logical_not=logical_not,
    logical_or=logical_or,
    get_static_value=get_static_value,
    reduce_sum=reduce_sum,
    stack=stack,
    unstack=unstack,
    expand_dims=expand_dims,
    tile=tile,
    where=where,
    arange=range,
    range=range,
    rank=rank,
    split=split,
    multiply=multiply,
    equal=equal,
    pad=pad,
    ones_like=ones_like,
    abs=abs,
    pow=pow,
    sin=sin,
    cos=cos,
    exp=exp,
    log=log,
    sigmoid=sigmoid,
    cumsum=cumsum,
    gather=gather,
    softplus=softplus,
    reduce_logsumexp=reduce_logsumexp,
    sqrt=sqrt,
    erf=erf,
    ones=ones,
    floor_divide=floordiv,
    floordiv=floordiv,
    greater=greater,
)
