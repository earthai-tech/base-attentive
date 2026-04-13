"""PyTorch-optimized V2 component builders.

This module provides PyTorch-specific implementations of BaseAttentiveV2
components with optimizations including:
- Native PyTorch nn.Module layers
- Torch's optimized attention mechanisms
- Automatic mixed precision (AMP) support via context
- Efficient tensor operations using torch ops
- CUDA/device-agnostic implementations
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


def _ensure_torch():
    """Ensure PyTorch is available."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for torch backend implementations. "
            "Install it with: pip install torch"
        )


class _TorchTemporalSelfAttentionEncoder(nn.Module if nn else object):
    """PyTorch-optimized temporal encoder with multi-head attention.

    Advantages over generic version:
    - Uses native torch.nn.MultiheadAttention (highly optimized C++ backend)
    - Support for CUDA/device acceleration
    - Automatic mixed precision compatible
    - Better gradient computation for PyTorch
    """

    def __init__(
        self,
        *,
        units: int,
        hidden_units: int,
        num_heads: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        name: str | None = None,
        **kwargs,
    ):
        _ensure_torch()
        super().__init__()

        # Ensure units is divisible by num_heads for attention
        if units % num_heads != 0:
            raise ValueError(
                f"units ({units}) must be divisible by num_heads ({num_heads})"
            )

        self.units = units
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_epsilon

        # PyTorch's MultiheadAttention (highly optimized)
        self.attention = nn.MultiheadAttention(
            embed_dim=units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,  # Expects (batch, seq_len, embed_dim)
        )

        self.norm1 = nn.LayerNorm(units, eps=layer_norm_epsilon)

        # FFN layers
        activation_fn = getattr(F, activation, None)
        if activation_fn is None:
            # Try as module
            activation_module = getattr(
                nn, activation.capitalize(), None
            )
            if activation_module is None:
                raise ValueError(f"Unknown activation: {activation}")

        self.ffn_hidden = nn.Linear(units, hidden_units)
        self.activation_fn = (
            (lambda x: activation_fn(x))
            if activation_fn
            else activation_module()
        )
        self.ffn_output = nn.Linear(hidden_units, units)

        self.dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        )
        self.norm2 = nn.LayerNorm(units, eps=layer_norm_epsilon)

    def forward(
        self,
        inputs: torch.Tensor,
        training: bool = False,  # noqa: FBT002
    ) -> torch.Tensor:
        """Forward pass with residual connections and layer normalization."""
        # Set training mode
        was_training = self.training
        self.train(training)

        try:
            # Self-attention with residual
            attn_output, _ = self.attention(inputs, inputs, inputs)
            x = self.norm1(inputs + attn_output)

            # FFN with residual
            ffn_output = self.ffn_hidden(x)
            if isinstance(self.activation_fn, nn.Module):
                ffn_output = self.activation_fn(ffn_output)
            else:
                ffn_output = self.activation_fn(ffn_output)

            if self.dropout is not None:
                ffn_output = self.dropout(ffn_output)

            ffn_output = self.ffn_output(ffn_output)
            return self.norm2(x + ffn_output)
        finally:
            self.train(was_training)


def _build_torch_dense_projection(
    *,
    units: int,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
) -> nn.Linear:
    """Build a PyTorch Linear projection layer.

    Arguments:
        units: Output dimension
        activation: Activation function name (ignored; torch uses forward hooks)
        name: Layer name (torch doesn't use this directly)
        **kwargs: Additional kwargs

    Returns:
        nn.Linear layer (user can wrap with activation if needed)
    """
    _ensure_torch()
    # Note: PyTorch nn.Linear doesn't have built-in activation
    # We return just the Linear layer; activation can be applied in forward()
    return nn.Linear(kwargs.get("in_features", 32), units)


def _build_torch_temporal_self_attention_encoder(
    *,
    units: int,
    hidden_units: int,
    num_heads: int,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    layer_norm_epsilon: float = 1e-6,
    name: str | None = None,
    **kwargs,
) -> _TorchTemporalSelfAttentionEncoder:
    """Build a PyTorch-optimized temporal self-attention encoder.

    Advantages:
    - Native PyTorch MultiheadAttention (C++ backend acceleration)
    - Automatic device placement support
    - Native CUDA support

    Arguments:
        units: Model dimension
        hidden_units: FFN hidden dimension
        num_heads: Number of attention heads
        activation: FFN activation function
        dropout_rate: Dropout rate for attention and FFN
        layer_norm_epsilon: Layer normalization epsilon
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        _TorchTemporalSelfAttentionEncoder instance
    """
    _ensure_torch()
    return _TorchTemporalSelfAttentionEncoder(
        units=units,
        hidden_units=hidden_units,
        num_heads=num_heads,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        name=name,
        **kwargs,
    )


def _build_torch_mean_pool(
    *,
    axis: int | None = None,
    keepdims: bool = False,
    name: str | None = None,
    **kwargs,
):
    """Build a pooling layer using PyTorch's functional API.

    Arguments:
        axis: Axis along which to compute the mean (PyTorch uses dim)
        keepdims: Whether to keep reduced dimensions
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        Callable that computes mean along axis
    """
    _ensure_torch()

    def mean_pool_fn(x):
        # Convert axis to PyTorch dim (PyTorch uses negative indexing from end)
        if axis is None:
            dim = None
        else:
            dim = axis if axis >= 0 else len(x.shape) + axis
        return torch.mean(x, dim=dim, keepdim=keepdims)

    return mean_pool_fn


def _build_torch_last_pool(
    *,
    name: str | None = None,
    **kwargs,
):
    """Build a layer that extracts the last timestep.

    Arguments:
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        Callable that extracts the last timestep
    """
    _ensure_torch()

    def last_pool_fn(x):
        return x[
            :, -1:, :
        ]  # Keep sequence dimension for compatibility

    return last_pool_fn


def _build_torch_concat_fusion(
    *,
    axis: int = -1,
    name: str | None = None,
    **kwargs,
):
    """Build a concatenation layer using PyTorch's cat operation.

    Arguments:
        axis: Axis along which to concatenate (PyTorch uses dim)
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        Callable that concatenates inputs along axis
    """
    _ensure_torch()

    def concat_fn(inputs):
        return torch.cat(inputs, dim=axis)

    return concat_fn


def _build_torch_point_forecast_head(
    *,
    output_dim: int = 1,
    forecast_horizon: int = 1,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
) -> nn.Linear:
    """Build a PyTorch-optimized point forecast output head.

    Arguments:
        output_dim: Output dimension (e.g., 1 for univariate)
        forecast_horizon: Number of forecasting steps
        activation: Output activation (not typically used)
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        nn.Linear for point forecasting
    """
    _ensure_torch()
    total_output = output_dim * forecast_horizon
    return nn.Linear(kwargs.get("in_features", 64), total_output)


def _build_torch_quantile_head(
    *,
    output_dim: int = 1,
    forecast_horizon: int = 1,
    quantiles: tuple[float, ...] | None = None,
    name: str | None = None,
    **kwargs,
) -> nn.Linear:
    """Build a PyTorch-optimized quantile forecast output head.

    Arguments:
        output_dim: Output dimension
        forecast_horizon: Number of forecasting steps
        quantiles: Tuple of quantile levels
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        nn.Linear for quantile forecasting
    """
    _ensure_torch()
    if not quantiles:
        quantiles = (0.1, 0.5, 0.9)
    num_quantiles = len(quantiles)
    total_output = output_dim * forecast_horizon * num_quantiles
    return nn.Linear(kwargs.get("in_features", 64), total_output)


def ensure_torch_v2_registered(registry=None) -> None:
    """Register all PyTorch-optimized V2 components.

    This registers PyTorch-specific implementations that will be preferred
    over generic implementations when the PyTorch backend is active.

    Arguments:
        registry: ComponentRegistry instance. If None, uses DEFAULT_COMPONENT_REGISTRY.
    """
    _ensure_torch()

    if registry is None:
        from ...registry import DEFAULT_COMPONENT_REGISTRY

        registry = DEFAULT_COMPONENT_REGISTRY

    # Register PyTorch-optimized projections
    registry.register(
        "projection.dense",
        _build_torch_dense_projection,
        backend="torch",
    )
    registry.register(
        "projection.static",
        _build_torch_dense_projection,
        backend="torch",
    )
    registry.register(
        "projection.dynamic",
        _build_torch_dense_projection,
        backend="torch",
    )
    registry.register(
        "projection.future",
        _build_torch_dense_projection,
        backend="torch",
    )
    registry.register(
        "projection.hidden",
        _build_torch_dense_projection,
        backend="torch",
    )

    # Register PyTorch-optimized encoders
    registry.register(
        "encoder.temporal_self_attention",
        _build_torch_temporal_self_attention_encoder,
        backend="torch",
    )

    # Register PyTorch-optimized pooling
    registry.register(
        "pool.mean",
        _build_torch_mean_pool,
        backend="torch",
    )
    registry.register(
        "pool.last",
        _build_torch_last_pool,
        backend="torch",
    )

    # Register PyTorch-optimized fusion
    registry.register(
        "fusion.concat",
        _build_torch_concat_fusion,
        backend="torch",
    )

    # Register PyTorch-optimized heads
    registry.register(
        "head.point_forecast",
        _build_torch_point_forecast_head,
        backend="torch",
    )
    registry.register(
        "head.quantile",
        _build_torch_quantile_head,
        backend="torch",
    )


__all__ = [
    "ensure_torch_v2_registered",
    "_TorchTemporalSelfAttentionEncoder",
    "_build_torch_dense_projection",
    "_build_torch_temporal_self_attention_encoder",
    "_build_torch_mean_pool",
    "_build_torch_last_pool",
    "_build_torch_concat_fusion",
    "_build_torch_point_forecast_head",
    "_build_torch_quantile_head",
]
