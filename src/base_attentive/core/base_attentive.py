# SPDX-License-Identifier: Apache-2.0
# BaseAttentive — https://github.com/earthai-tech/base-attentive
# Copyright (c) 2026-present
# Author: LKouadio <etanoyau@gmail.com>

"""
Base class for advanced, attentive sequence-to-sequence models.

This module provides the BaseAttentive class, a foundational blueprint for
building powerful, data-driven, sequence-to-sequence time series forecasting
models with advanced attention mechanisms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None
    keras = None
    layers = None


class BaseAttentive:
    """
    BaseAttentive: A foundational blueprint for sequence-to-sequence
    time series forecasting models with attention mechanisms.

    This is a stub/template class. Full implementation requires:
    - TensorFlow/Keras with attention layer implementations
    - Custom attention components (cross, hierarchical, memory-augmented)
    - VSN and GRN layers for feature processing
    - LSTM and transformer encoder components

    For the full implementation, refer to the geoprior-v3 repository.

    Parameters
    ----------
    static_input_dim : int
        Dimension of static input features
    dynamic_input_dim : int
        Dimension of dynamic input features
    future_input_dim : int
        Dimension of future input features
    output_dim : int, default=1
        Dimension of output targets
    forecast_horizon : int, default=1
        Number of steps to forecast ahead
    mode : str, default='pihal_like'
        Operating mode: 'pihal_like' or 'tft_like'
    embed_dim : int, default=32
        Embedding dimension
    hidden_units : int, default=64
        Hidden units for dense layers
    lstm_units : int, default=64
        LSTM hidden units
    attention_units : int, default=32
        Attention head dimension
    num_heads : int, default=4
        Number of attention heads
    dropout_rate : float, default=0.1
        Dropout rate for regularization
    max_window_size : int, default=10
        Maximum window size for DTW
    memory_size : int, default=100
        Memory size for memory-augmented attention
    scales : list, optional
        (Reserved) Scales for multi-scale processing
    multi_scale_agg : str, default='last'
        Aggregation mode for multi-scale outputs
    final_agg : str, default='last'
        Final aggregation mode
    activation : str, default='relu'
        Activation function
    use_residuals : bool, default=True
        Whether to use residual connections
    use_vsn : bool, default=True
        Whether to use Variable Selection Networks
    vsn_units : int, optional
        VSN hidden units
    use_batch_norm : bool, default=False
        Whether to use batch normalization
    apply_dtw : bool, default=True
        Whether to apply Dynamic Time Warping
    quantiles : list, optional
        Quantile levels for quantile regression
    objective : str, default='hybrid'
        Architecture objective: 'hybrid' or 'transformer'
    architecture_config : dict, optional
        Advanced architecture configuration
    verbose : int, default=0
        Verbosity level
    name : str, default='BaseAttentiveModel'
        Model name
    **kwargs
        Additional keyword arguments

    Attributes
    ----------
    config : dict
        Model configuration dictionary
    trainable_params : int
        Number of trainable parameters

    Examples
    --------
    Create and use a BaseAttentive model:

    >>> from base_attentive import BaseAttentive
    >>> model = BaseAttentive(
    ...     static_input_dim=4,
    ...     dynamic_input_dim=8,
    ...     future_input_dim=6,
    ...     output_dim=2,
    ...     forecast_horizon=24,
    ...     quantiles=[0.1, 0.5, 0.9]
    ... )

    Notes
    -----
    This is a stub implementation. For production use, we recommend:
    1. Installing geoprior-v3 for the full BaseAttentive implementation
    2. Contributing to the base-attentive standalone package
    3. Using HALNet or PIHALNet models built on BaseAttentive

    The full architecture supports:
    - Hybrid encoder (Multi-scale LSTM + Attention)
    - Transformer encoder (Pure attention)
    - Multiple attention mechanisms (cross, hierarchical, memory)
    - Variable Selection Networks for feature importance
    - Uncertainty quantification via quantile modeling
    - Dynamic Time Warping for temporal alignment

    References
    ----------
    .. [1] Vaswani et al. "Attention Is All You Need" (2017)
    .. [2] Lim et al. "Temporal Fusion Transformers" (2021)
    .. [3] Bahdanau et al. "Neural Machine Translation" (2015)
    """

    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        mode: Optional[str] = None,
        num_encoder_layers: int = 2,
        quantiles: Optional[List[float]] = None,
        embed_dim: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        attention_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        max_window_size: int = 10,
        memory_size: int = 100,
        scales: Optional[List[int]] = None,
        multi_scale_agg: str = "last",
        final_agg: str = "last",
        activation: str = "relu",
        use_residuals: bool = True,
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        use_batch_norm: bool = False,
        apply_dtw: bool = True,
        attention_levels: Optional[Union[str, List[str]]] = None,
        objective: str = "hybrid",
        architecture_config: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        name: str = "BaseAttentiveModel",
        **kwargs,
    ):
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required to use BaseAttentive. "
                "Install it with: pip install tensorflow>=2.12.0"
            )

        # Store configuration
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.mode = mode or "pihal_like"
        self.num_encoder_layers = num_encoder_layers
        self.quantiles = quantiles
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.scales = scales or [1, 3]
        self.multi_scale_agg = multi_scale_agg
        self.final_agg = final_agg
        self.activation = activation
        self.use_residuals = use_residuals
        self.use_vsn = use_vsn
        self.vsn_units = vsn_units or hidden_units
        self.use_batch_norm = use_batch_norm
        self.apply_dtw = apply_dtw
        self.attention_levels = attention_levels
        self.objective = objective
        self.architecture_config = architecture_config or {}
        self.verbose = verbose
        self.name = name
        self.kwargs = kwargs

        # Note: Full implementation requires building Keras model
        self._model = None

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"BaseAttentive(\n"
            f"  static_dim={self.static_input_dim},\n"
            f"  dynamic_dim={self.dynamic_input_dim},\n"
            f"  future_dim={self.future_input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  forecast_horizon={self.forecast_horizon},\n"
            f"  attention_units={self.attention_units},\n"
            f"  objective='{self.objective}'\n"
            f")"
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.

        Returns
        -------
        dict
            Complete model configuration
        """
        return {
            "static_input_dim": self.static_input_dim,
            "dynamic_input_dim": self.dynamic_input_dim,
            "future_input_dim": self.future_input_dim,
            "output_dim": self.output_dim,
            "forecast_horizon": self.forecast_horizon,
            "mode": self.mode,
            "num_encoder_layers": self.num_encoder_layers,
            "quantiles": self.quantiles,
            "embed_dim": self.embed_dim,
            "hidden_units": self.hidden_units,
            "lstm_units": self.lstm_units,
            "attention_units": self.attention_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "max_window_size": self.max_window_size,
            "memory_size": self.memory_size,
            "scales": self.scales,
            "multi_scale_agg": self.multi_scale_agg,
            "final_agg": self.final_agg,
            "activation": self.activation,
            "use_residuals": self.use_residuals,
            "use_vsn": self.use_vsn,
            "vsn_units": self.vsn_units,
            "use_batch_norm": self.use_batch_norm,
            "apply_dtw": self.apply_dtw,
            "attention_levels": self.attention_levels,
            "objective": self.objective,
            "architecture_config": self.architecture_config,
            "verbose": self.verbose,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseAttentive":
        """
        Create a model from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from get_config()

        Returns
        -------
        BaseAttentive
            New model instance
        """
        arch_config = config.pop("architecture_config", None)
        return cls(**config, architecture_config=arch_config)

    def summary(self) -> None:
        """Print model summary."""
        print(self)
        print(f"\nConfiguration:")
        for key, value in self.get_config().items():
            if value is not None:
                print(f"  {key}: {value}")


__all__ = ["BaseAttentive"]
