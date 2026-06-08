"""PyTorch implementation of PADR-Net."""

from __future__ import annotations

from typing import Any

from ...api.property import NNLearner
from ...applications.flood.config import PADRNetConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for TorchPADRNet. "
            "Install it with: pip install torch"
        )


class _PADRBlock(nn.Module if nn else object):
    def __init__(self, config: PADRNetConfig):
        _ensure_torch()
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(
                config.hidden_dim, config.hidden_dim * 2
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_dim * 2, config.hidden_dim
            ),
        )
        self.norm_ffn = nn.LayerNorm(config.hidden_dim)

    def forward(self, inputs):
        attn, _weights = self.attention(
            inputs,
            inputs,
            inputs,
            need_weights=False,
        )
        x = self.norm_attention(inputs + attn)
        return self.norm_ffn(x + self.ffn(x))


_TORCH_PADR_BASES = (
    (nn.Module, NNLearner) if nn else (NNLearner,)
)


class TorchPADRNet(*_TORCH_PADR_BASES):
    """Physics-aware attention model for flood forecasting."""

    def __init__(
        self,
        config: PADRNetConfig,
        *,
        name: str = "padr_net",
        **kwargs: Any,
    ):
        del kwargs
        _ensure_torch()
        super().__init__()
        self.config = config
        self.name = name
        self.dynamic_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.static_projection = (
            nn.Sequential(
                nn.Linear(
                    config.static_dim, config.hidden_dim
                ),
                nn.GELU(),
            )
            if config.static_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                _PADRBlock(config)
                for _ in range(config.num_layers)
            ]
        )
        self.depth_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_dim, config.forecast_horizon
            ),
            nn.Softplus(),
        )

    def forward(self, inputs, static_inputs=None):
        x = self.dynamic_projection(inputs)
        if (
            self.static_projection is not None
            and static_inputs is not None
        ):
            static = self.static_projection(
                static_inputs
            ).unsqueeze(1)
            x = x + static
        for block in self.blocks:
            x = block(x)
        features = x.mean(dim=1)
        depth = self.depth_head(features).unsqueeze(-1)
        threshold = torch.as_tensor(
            self.config.flood_threshold,
            dtype=depth.dtype,
            device=depth.device,
        )
        scale = torch.as_tensor(
            max(0.004, 0.08 * self.config.flood_threshold),
            dtype=depth.dtype,
            device=depth.device,
        )
        exceedance = torch.sigmoid(
            (depth - threshold) / scale
        )
        return {
            "depth": depth,
            "exceedance_probability": exceedance,
            "features": features,
        }


__all__ = ["TorchPADRNet"]
