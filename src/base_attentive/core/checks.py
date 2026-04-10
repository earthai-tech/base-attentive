"""Parameter checking utilities for core module."""

from __future__ import annotations

from typing import Any, Type, Union, get_args, get_origin

__all__ = ["validate_nested_param"]


def validate_nested_param(
    value: Any, expected_type: Type, param_name: str
) -> Any:
    """
    Validate nested parameter type.

    Parameters
    ----------
    value : Any
        The value to validate.
    expected_type : Type
        The expected type (e.g., list[int]).
    param_name : str
        Parameter name for error messages.

    Returns
    -------
    Any
        The validated value.

    Raises
    ------
    TypeError
        If the value doesn't match the expected type.
    """
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Handle list[T] types
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(
                f"{param_name} must be a list, got {type(value).__name__}"
            )
        if args:
            expected_element_type = args[0]
            for i, item in enumerate(value):
                if not isinstance(item, expected_element_type):
                    raise TypeError(
                        f"{param_name}[{i}] must be {expected_element_type.__name__}, "
                        f"got {type(item).__name__}"
                    )
    elif not isinstance(value, expected_type):
        raise TypeError(
            f"{param_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    return value
