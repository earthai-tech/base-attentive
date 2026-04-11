"""Minimal docstring component utilities used by BaseAttentive."""

from __future__ import annotations

import re

__all__ = [
    "DocstringComponents",
    "_halnet_core_params",
]


class DocstringComponents:
    """Store named docstring fragments with dot-style access."""

    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict: dict[str, str], strip_whitespace: bool = True):
        if strip_whitespace:
            entries = {}
            for key, value in comp_dict.items():
                match = re.match(self.regexp, value)
                entries[key] = value if match is None else match.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries

    def __getattr__(self, attr: str) -> str:
        if attr in self.entries:
            return self.entries[attr]
        raise AttributeError(attr)

    @classmethod
    def from_nested_components(
        cls, **kwargs: "DocstringComponents"
    ) -> "DocstringComponents":
        return cls(kwargs, strip_whitespace=False)


_halnet_core_params = dict(
    static_input_dim="""
static_input_dim : int
    Dimensionality of the static (time-invariant) input features.
    These are features that do not change over time for a given
    sample, such as a sensor location, soil type, or product
    category. If 0, no static features are used.
""",
    dynamic_input_dim="""
dynamic_input_dim : int
    Dimensionality of the dynamic (time-varying) input features
    known over the historical lookback window. This usually includes
    lagged target values and past covariates such as rainfall,
    temperature, or sales history.
""",
    future_input_dim="""
future_input_dim : int
    Dimensionality of the time-varying features whose values are
    known in advance for the forecast period, such as holidays,
    planned promotions, or calendar indicators. If 0, no future
    features are used.
""",
    embed_dim="""
embed_dim : int, default 32
    Base dimensionality of the internal feature space. Static,
    dynamic, and future inputs are projected into this shared
    representation before they are processed by recurrent and
    attention layers.
""",
    hidden_units="""
hidden_units : int, default 64
    Number of hidden units used in the Gated Residual Networks
    throughout the architecture. Increasing this value raises the
    capacity of the model's non-linear transformations.
""",
    lstm_units="""
lstm_units : int, default 64
    Number of hidden units in each LSTM inside the
    :class:`~base_attentive.components.temporal.MultiScaleLSTM`
    encoder block.
""",
    attention_units="""
attention_units : int, default 32
    Output dimensionality used by the attention mechanisms. This
    corresponds to the model dimension used for cross-attention,
    hierarchical attention, and memory-augmented attention. It
    should be divisible by `num_heads`.
""",
    num_heads="""
num_heads : int, default 4
    Number of attention heads in each multi-head attention layer.
    Multiple heads allow the model to attend to different
    representation subspaces in parallel.
""",
    dropout_rate="""
dropout_rate : float, default 0.1
    Dropout rate applied inside several layers such as GRNs and
    attention blocks to reduce overfitting. Must be between 0 and 1.
""",
    max_window_size="""
max_window_size : int, default 10
    Number of past time steps in the lookback window. This should
    match the temporal span used when preparing the historical
    dynamic inputs.
""",
    memory_size="""
memory_size : int, default 100
    Number of memory slots used by the
    :class:`~base_attentive.components.attention.MemoryAugmentedAttention`
    layer for modeling longer-range dependencies.
""",
    scales="""
scales : list of int, optional
    Scale factors used by
    :class:`~base_attentive.components.temporal.MultiScaleLSTM`.
    Each scale `s` processes every `s`-th time step. If `None` or
    ``'auto'``, the default scale list is `[1]`.
""",
    multi_scale_agg="""
multi_scale_agg : {'last', 'average', 'concat', ...}, default 'last'
    Strategy used to combine the outputs from the multi-scale LSTM
    encoder. ``'concat'`` preserves a richer sequence view for later
    attention layers, while ``'last'`` and related reductions produce
    a compact context vector.
""",
    final_agg="""
final_agg : {'last', 'average', 'flatten'}, default 'last'
    Aggregation strategy used to collapse the final temporal feature
    map into a single feature vector before decoding.
""",
    activation="""
activation : str, default 'relu'
    Activation function used in Dense layers and Gated Residual
    Networks throughout the model.
""",
    use_residuals="""
use_residuals : bool, default True
    If `True`, enables residual add-and-normalize connections after
    key sublayers. These shortcuts improve optimization and help
    preserve gradient flow in deeper models.
""",
    use_vsn="""
use_vsn : bool, default True
    If `True`, Variable Selection Networks are used for input feature
    processing. If `False`, simpler Dense projections are used instead.
""",
    vsn_units="""
vsn_units : int, optional
    Number of units used inside the internal GRNs of the Variable
    Selection Networks. If `None`, the model falls back to a value
    derived from `hidden_units`.
""",
)
