"""Compatibility helpers for smooth API version transitions.

This module centralizes parameter migration rules so public entry points
can accept legacy constructor payloads while gradually moving users
toward the canonical API.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


class BaseAttentiveCompatibilityWarning(UserWarning):
    """Base warning for API-compatibility behavior."""


class DeprecatedParameterWarning(
    BaseAttentiveCompatibilityWarning
):
    """Warn when a deprecated parameter name is still used."""


class RemovedParameterWarning(
    BaseAttentiveCompatibilityWarning
):
    """Warn when a removed parameter is still supplied."""


class UnsupportedCompatibilityWarning(
    BaseAttentiveCompatibilityWarning
):
    """Warn when a compatibility knob is accepted but ignored."""


@dataclass(frozen=True)
class ParameterRule:
    """Describe how a public parameter migrates across versions.

    Parameters
    ----------
    old_name : str
        Deprecated or alternate parameter name.
    new_name : str or None, default=None
        Canonical replacement name.  When ``None`` the rule represents a
        no-op or removed parameter.
    since : str or None, default=None
        Version in which the compatibility rule became active.
    remove_in : str or None, default=None
        Planned removal version for the deprecated name.
    behavior : str, default="rename"
        Rule behavior. Supported values are ``rename``,
        ``interchangeable``, ``removed``, and ``noop``.
    message : str or None, default=None
        Optional extra explanatory text appended to warnings.
    transform : callable or None, default=None
        Optional value transformer applied when forwarding the legacy
        value to the canonical parameter.
    implemented : bool, default=True
        Whether the compatibility mapping is implemented.  Useful when a
        keyword is accepted but intentionally ignored for the moment.
    precedence : str, default="new"
        Conflict policy when both the old and new names are supplied.
    """

    old_name: str
    new_name: str | None = None
    since: str | None = None
    remove_in: str | None = None
    behavior: str = "rename"
    message: str | None = None
    transform: Callable[[Any], Any] | None = None
    implemented: bool = True
    precedence: str = "new"


def _rule_warning_prefix(
    component_name: str,
    rule: ParameterRule,
) -> str:
    pieces = [f"{component_name}: '{rule.old_name}'"]
    if (
        rule.behavior in {"rename", "interchangeable"}
        and rule.new_name
    ):
        pieces.append(
            "is deprecated"
            + (f" since {rule.since}" if rule.since else "")
            + (
                f" and will be removed in {rule.remove_in}"
                if rule.remove_in
                else ""
            )
            + f". Use '{rule.new_name}' instead."
        )
    elif rule.behavior == "removed":
        pieces.append("has been removed.")
    elif rule.behavior == "noop":
        pieces.append(
            "is accepted for compatibility but currently has no effect."
        )
    else:
        pieces.append("requires compatibility handling.")
    if rule.message:
        pieces.append(rule.message)
    return " ".join(pieces)


def _warn(
    message: str,
    warning_cls: type[Warning],
) -> None:
    warnings.warn(message, warning_cls, stacklevel=3)


def n_quantiles_to_quantiles(
    value: int | None,
) -> tuple[float, ...] | None:
    """Convert an integer quantile count into evenly spaced quantiles."""
    if value is None:
        return None
    try:
        n_quantiles = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "n_quantiles must be an integer."
        ) from exc
    if n_quantiles <= 0:
        raise ValueError("n_quantiles must be >= 1.")
    if n_quantiles == 1:
        return (0.5,)
    return tuple(
        index / (n_quantiles + 1)
        for index in range(1, n_quantiles + 1)
    )


def resolve_deprecated_kwargs(
    kwargs: Mapping[str, Any],
    rules: Iterable[ParameterRule],
    *,
    component_name: str = "BaseAttentive",
) -> dict[str, Any]:
    """Normalize a mapping using compatibility rules."""
    resolved = dict(kwargs)

    for rule in rules:
        value = resolved.get(rule.old_name)
        old_present = (
            rule.old_name in resolved and value is not None
        )
        new_value = (
            resolved.get(rule.new_name)
            if rule.new_name is not None
            else None
        )
        new_present = (
            rule.new_name is not None
            and rule.new_name in resolved
            and new_value is not None
        )
        if not old_present:
            continue

        if rule.behavior in {"rename", "interchangeable"}:
            if new_present and rule.precedence == "new":
                _warn(
                    (
                        f"{component_name}: both '{rule.old_name}' and "
                        f"'{rule.new_name}' were provided. Using "
                        f"'{rule.new_name}' and ignoring deprecated "
                        f"'{rule.old_name}'."
                    ),
                    DeprecatedParameterWarning,
                )
                resolved.pop(rule.old_name, None)
                continue

            mapped = (
                rule.transform(value)
                if rule.transform is not None
                else value
            )
            if rule.new_name is not None:
                resolved[rule.new_name] = mapped
            _warn(
                _rule_warning_prefix(component_name, rule),
                DeprecatedParameterWarning,
            )
            resolved.pop(rule.old_name, None)
            continue

        if rule.behavior == "removed":
            _warn(
                _rule_warning_prefix(component_name, rule),
                RemovedParameterWarning,
            )
            resolved.pop(rule.old_name, None)
            continue

        if rule.behavior == "noop":
            warning_cls = (
                UnsupportedCompatibilityWarning
                if not rule.implemented
                else DeprecatedParameterWarning
            )
            _warn(
                _rule_warning_prefix(component_name, rule),
                warning_cls,
            )
            resolved.pop(rule.old_name, None)
            continue

        raise ValueError(
            f"Unsupported compatibility behavior: {rule.behavior!r}."
        )

    return resolved


def resolve_deprecated_config(
    config: Mapping[str, Any],
    rules: Iterable[ParameterRule],
    *,
    component_name: str = "config",
) -> dict[str, Any]:
    """Compatibility wrapper for configuration dictionaries."""
    return resolve_deprecated_kwargs(
        config,
        rules,
        component_name=component_name,
    )


def apply_parameter_compatibility(
    rules: Iterable[ParameterRule],
    *,
    component_name: str,
):
    """Decorator applying compatibility normalization to kwargs."""

    def decorator(init):
        @functools.wraps(init)
        def wrapped(self, *args, **kwargs):
            normalized = resolve_deprecated_kwargs(
                kwargs,
                rules,
                component_name=component_name,
            )
            return init(self, *args, **normalized)

        return wrapped

    return decorator


BASE_ATTENTIVE_PARAMETER_RULES = (
    ParameterRule(
        old_name="static_input_dim",
        new_name="static_dim",
        since="2.1.0",
        remove_in="3.0.0",
        behavior="rename",
    ),
    ParameterRule(
        old_name="dynamic_input_dim",
        new_name="dynamic_dim",
        since="2.1.0",
        remove_in="3.0.0",
        behavior="rename",
    ),
    ParameterRule(
        old_name="future_input_dim",
        new_name="future_dim",
        since="2.1.0",
        remove_in="3.0.0",
        behavior="rename",
    ),
    ParameterRule(
        old_name="max_window_size",
        new_name="lookback_window",
        since="2.1.0",
        remove_in="3.0.0",
        behavior="interchangeable",
        message=(
            "'lookback_window' is the canonical public name in the "
            "modern BaseAttentive interface."
        ),
    ),
    ParameterRule(
        old_name="attention_levels",
        new_name="attention_stack",
        since="2.1.0",
        remove_in="3.0.0",
        behavior="rename",
    ),
)


__all__ = [
    "BaseAttentiveCompatibilityWarning",
    "DeprecatedParameterWarning",
    "RemovedParameterWarning",
    "UnsupportedCompatibilityWarning",
    "ParameterRule",
    "BASE_ATTENTIVE_PARAMETER_RULES",
    "apply_parameter_compatibility",
    "n_quantiles_to_quantiles",
    "resolve_deprecated_config",
    "resolve_deprecated_kwargs",
]
