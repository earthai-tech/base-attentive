"""Legacy ``BaseAttentive`` configuration adapters."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from ..compat.sklearn import Interval, StrOptions, validate_params
from .defaults import (
    DEFAULT_BASE_ATTENTIVE_RUNTIME,
    DEFAULT_LEGACY_ARCHITECTURE,
)
from .schema import (
    BaseAttentiveArchitectureSpec,
    BaseAttentiveComponentSpec,
    BaseAttentiveRuntimeSpec,
    BaseAttentiveSpec,
)

_ALLOWED_ATTENTION_LEVELS = {
    "cross",
    "hierarchical",
    "memory",
}
_ALLOWED_RUNTIME_MODES = {
    "tft",
    "pihal",
    "tft_like",
    "pihal_like",
    "tft-like",
    "pihal-like",
    "point",
    "quantile",
}
_LEGACY_BASE_ATTENTIVE_PARAM_RULES = {
    "static_input_dim": [Interval(int, 0, None, closed="left")],
    "dynamic_input_dim": [Interval(int, 1, None, closed="left")],
    "future_input_dim": [Interval(int, 0, None, closed="left")],
    "output_dim": [Interval(int, 1, None, closed="left")],
    "forecast_horizon": [Interval(int, 1, None, closed="left")],
    "mode": [StrOptions(_ALLOWED_RUNTIME_MODES), None],
    "num_encoder_layers": [Interval(int, 1, None, closed="left")],
    "quantiles": [list, tuple, None],
    "embed_dim": [Interval(int, 1, None, closed="left")],
    "hidden_units": [Interval(int, 1, None, closed="left")],
    "lstm_units": [int, tuple],
    "attention_units": [Interval(int, 1, None, closed="left")],
    "num_heads": [Interval(int, 1, None, closed="left")],
    "dropout_rate": [Interval(float, 0.0, 1.0, closed="both")],
    "max_window_size": [Interval(int, 1, None, closed="left")],
    "memory_size": [Interval(int, 1, None, closed="left")],
    "scales": [list, tuple, StrOptions({"auto"}), None],
    "multi_scale_agg": [str],
    "final_agg": [str],
    "activation": [str],
    "use_residuals": [bool],
    "use_vsn": [bool],
    "vsn_units": [Interval(int, 1, None, closed="left"), None],
    "use_batch_norm": [bool],
    "apply_dtw": [bool],
    "attention_levels": [str, list, tuple, None],
    "objective": [str, None],
    "architecture_config": [BaseAttentiveArchitectureSpec, dict, None],
    "backend_name": [str, None],
    "component_overrides": [BaseAttentiveComponentSpec, dict, None],
    "verbose": [Interval(int, 0, None, closed="left")],
    "extras": [dict, None],
}


def _coerce_mapping(
    value: Any, *, name: str
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(
        f"{name} must be a mapping or dataclass instance."
    )


def _normalize_encoder_type(value: str | None) -> str:
    if value is None:
        return DEFAULT_LEGACY_ARCHITECTURE["encoder_type"]

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"hybrid", "pihal", "pihal_like"}:
        return "hybrid"
    if normalized in {"transformer", "tft", "tft_like"}:
        return "transformer"
    raise ValueError(
        "encoder_type/objective must resolve to 'hybrid' or "
        f"'transformer', got {value!r}."
    )


def _normalize_attention_stack(
    value: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_LEGACY_ARCHITECTURE[
            "decoder_attention_stack"
        ]

    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"all", "default", "full"}:
            return DEFAULT_LEGACY_ARCHITECTURE[
                "decoder_attention_stack"
            ]
        parts = raw.replace("+", ",").replace("|", ",")
        values = [
            item.strip() for item in parts.split(",") if item
        ]
    else:
        values = [str(item).strip().lower() for item in value]

    normalized: list[str] = []
    for item in values:
        if item not in _ALLOWED_ATTENTION_LEVELS:
            raise ValueError(
                "attention_levels entries must be drawn from "
                f"{sorted(_ALLOWED_ATTENTION_LEVELS)!r}; "
                f"got {item!r}."
            )
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _normalize_legacy_mode(
    value: str | None,
) -> tuple[str | None, str | None]:
    """Normalize legacy ``mode`` into runtime and head intent.

    The legacy constructor sometimes used ``mode='point'`` or
    ``mode='quantile'`` to express output-head intent. In the V2 spec,
    ``runtime.mode`` is reserved for runtime/decoder style and point vs.
    quantile is encoded through ``head_type`` plus ``quantiles``.
    """
    if value is None:
        return None, None

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"point", "quantile"}:
        return None, normalized
    if normalized in {"tft", "pihal", "tft_like", "pihal_like"}:
        return normalized, None
    raise ValueError(
        "mode must resolve to one of 'tft', 'pihal', 'tft_like', "
        f"'pihal_like', 'point', or 'quantile'; got {value!r}."
    )


def _resolve_legacy_head_type(
    *,
    mode: str | None,
    quantiles: list[float] | tuple[float, ...] | None,
) -> tuple[str | None, str]:
    runtime_mode, legacy_head_mode = _normalize_legacy_mode(mode)
    quantiles_tuple = tuple(quantiles or ())
    inferred_head_type = "quantile" if quantiles_tuple else "point"

    if legacy_head_mode == "point" and quantiles_tuple:
        raise ValueError(
            "legacy mode='point' conflicts with quantiles. "
            "Remove mode or omit quantiles for a point head."
        )
    if legacy_head_mode == "quantile" and not quantiles_tuple:
        raise ValueError(
            "legacy mode='quantile' requires quantiles. "
            "Provide quantiles or remove mode."
        )

    return runtime_mode, legacy_head_mode or inferred_head_type


def normalize_legacy_architecture_spec(
    *,
    objective: str | None = None,
    use_vsn: bool = True,
    attention_levels: str | list[str] | None = None,
    architecture_config: (
        BaseAttentiveArchitectureSpec
        | Mapping[str, Any]
        | None
    ) = None,
) -> BaseAttentiveArchitectureSpec:
    """Build a structured architecture spec from legacy inputs."""
    data = dict(DEFAULT_LEGACY_ARCHITECTURE)
    data.update(
        _coerce_mapping(
            architecture_config,
            name="architecture_config",
        )
    )

    objective_override = objective
    if objective_override is None and "objective" in data:
        objective_override = data.pop("objective")

    data["encoder_type"] = _normalize_encoder_type(
        data.get("encoder_type") or objective_override
    )

    if (
        not use_vsn
        and data.get("feature_processing") == "vsn"
    ):
        data["feature_processing"] = "dense"
    elif use_vsn and "feature_processing" not in data:
        data["feature_processing"] = "vsn"

    if attention_levels is None:
        attention_levels = data.get("decoder_attention_stack")
    data["decoder_attention_stack"] = (
        _normalize_attention_stack(attention_levels)
    )

    feature_processing = (
        str(data.get("feature_processing", "vsn"))
        .strip()
        .lower()
    )
    if feature_processing not in {"vsn", "dense"}:
        raise ValueError(
            "feature_processing must be 'vsn' or 'dense'."
        )
    data["feature_processing"] = feature_processing

    return BaseAttentiveArchitectureSpec(**data)


def normalize_legacy_runtime_spec(
    **kwargs: Any,
) -> BaseAttentiveRuntimeSpec:
    """Build a structured runtime spec from legacy inputs."""
    data = dict(DEFAULT_BASE_ATTENTIVE_RUNTIME)
    data.update(
        {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
    )

    scales = data.get("scales", ())
    if isinstance(scales, str):
        data["scales"] = scales.lower()
    else:
        data["scales"] = tuple(scales or ())

    return BaseAttentiveRuntimeSpec(**data)


@validate_params(
    _LEGACY_BASE_ATTENTIVE_PARAM_RULES,
    prefer_skip_nested_validation=False,
)
def legacy_base_attentive_to_spec(
    *,
    static_input_dim: int,
    dynamic_input_dim: int,
    future_input_dim: int,
    output_dim: int = 1,
    forecast_horizon: int = 1,
    mode: str | None = None,
    num_encoder_layers: int = 2,
    quantiles: list[float] | tuple[float, ...] | None = None,
    embed_dim: int = 32,
    hidden_units: int = 64,
    lstm_units: int | tuple[int, ...] = 64,
    attention_units: int = 32,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    max_window_size: int = 10,
    memory_size: int = 100,
    scales: list[int] | tuple[int, ...] | str | None = None,
    multi_scale_agg: str = "last",
    final_agg: str = "last",
    activation: str = "relu",
    use_residuals: bool = True,
    use_vsn: bool = True,
    vsn_units: int | None = None,
    use_batch_norm: bool = False,
    apply_dtw: bool = True,
    attention_levels: str | list[str] | None = None,
    objective: str = "hybrid",
    architecture_config: Mapping[str, Any] | None = None,
    backend_name: str | None = None,
    component_overrides: (
        BaseAttentiveComponentSpec | Mapping[str, Any] | None
    ) = None,
    verbose: int = 0,
    extras: Mapping[str, Any] | None = None,
) -> BaseAttentiveSpec:
    """Adapt the legacy constructor payload into ``BaseAttentiveSpec``.

    This function is the compatibility boundary between the permissive
    legacy ``BaseAttentive`` constructor and the strict resolver-driven
    ``BaseAttentiveSpec`` world. Broad type/range checks happen here,
    while final structural validation remains the responsibility of the
    normalized spec validators.
    """
    from .normalize import (
        normalize_base_attentive_spec,
        normalize_component_spec,
    )

    runtime_mode, head_type = _resolve_legacy_head_type(
        mode=mode,
        quantiles=quantiles,
    )

    architecture = normalize_legacy_architecture_spec(
        objective=objective,
        use_vsn=use_vsn,
        attention_levels=attention_levels,
        architecture_config=architecture_config,
    )
    runtime = normalize_legacy_runtime_spec(
        mode=runtime_mode,
        num_encoder_layers=num_encoder_layers,
        max_window_size=max_window_size,
        memory_size=memory_size,
        scales=scales,
        multi_scale_agg=multi_scale_agg,
        final_agg=final_agg,
        use_residuals=use_residuals,
        use_batch_norm=use_batch_norm,
        apply_dtw=apply_dtw,
        verbose=verbose,
    )
    components = normalize_component_spec(component_overrides)

    extras_payload = dict(extras or {})
    extras_payload.update(
        {
            "legacy_mode": mode,
            "legacy_objective": objective,
            "legacy_attention_levels": attention_levels,
            "legacy_architecture_config": dict(
                architecture_config or {}
            ),
            "legacy_component_overrides": dict(
                _coerce_mapping(
                    component_overrides,
                    name="component_overrides",
                )
            ),
        }
    )

    return normalize_base_attentive_spec(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        future_input_dim=future_input_dim,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        quantiles=tuple(quantiles or ()),
        embed_dim=embed_dim,
        hidden_units=hidden_units,
        attention_heads=num_heads,
        dropout_rate=dropout_rate,
        activation=activation,
        backend_name=backend_name,
        head_type=head_type,
        lstm_units=lstm_units,
        attention_units=attention_units,
        vsn_units=vsn_units,
        architecture=architecture,
        runtime=runtime,
        components=components,
        extras=extras_payload,
    )


__all__ = [
    "legacy_base_attentive_to_spec",
    "normalize_legacy_architecture_spec",
    "normalize_legacy_runtime_spec",
]
