"""Compatibility layers for validator support."""

from __future__ import annotations

import inspect
import numbers

try:
    from sklearn.utils._param_validation import (
        Interval as sklearn_Interval,
    )
    from sklearn.utils._param_validation import (
        StrOptions,
    )
    from sklearn.utils._param_validation import (
        validate_params as sklearn_validate_params,
    )
except ImportError:
    # Fallback for older scikit-learn versions
    try:
        from sklearn.utils.validation import (
            Interval as sklearn_Interval,
        )
        from sklearn.utils.validation import (
            StrOptions,
        )

        sklearn_validate_params = None
    except ImportError:
        sklearn_Interval = None
        StrOptions = None
        sklearn_validate_params = None

try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:

    def check_is_fitted(estimator, attributes, *, msg=None, all_or_any=all):
        """Simple fallback for check_is_fitted."""
        pass


class Interval:
    """
    Compatibility wrapper for scikit-learn's Interval class to handle
    versions that use different parameter interfaces.

    This wrapper converts `int` type to `numbers.Integral` to match
    scikit-learn's expected parameter types, and handles the `inclusive`
    parameter for newer versions.
    """

    def __new__(cls, type_, left, right, *, closed="right", inclusive=None):
        """
        Create a compatible Interval object based on the scikit-learn version.

        Parameters
        ----------
        type_ : type
            The expected type for the parameter. If `int` is passed, it will be
            converted to `numbers.Integral`.
        left : int or float
            The left bound of the interval.
        right : int or float
            The right bound of the interval.
        closed : str, default="right"
            Which side(s) of the interval are closed.
        inclusive : bool, optional
            Whether the interval is inclusive (only for newer versions).
        """
        if sklearn_Interval is None:
            raise ImportError(
                "scikit-learn is not installed or does not have Interval support"
            )

        # Convert builtin numeric types to the abstract types newer sklearn expects.
        if type_ is int:
            type_ = numbers.Integral
        elif type_ is float:
            type_ = numbers.Real

        # Check if 'inclusive' is a parameter in sklearn_Interval
        signature = inspect.signature(sklearn_Interval.__init__)
        if "inclusive" in signature.parameters:
            # Newer version - use inclusive parameter
            return sklearn_Interval(
                type_, left, right, closed=closed, inclusive=inclusive
            )
        else:
            # Older version - don't use inclusive parameter
            return sklearn_Interval(type_, left, right, closed=closed)


def validate_params(
    params,
    *args,
    prefer_skip_nested_validation=True,
    **kwargs,
):
    """
    Compatibility wrapper for scikit-learn's validate_params decorator
    to handle versions that require the prefer_skip_nested_validation argument.

    Parameters
    ----------
    params : dict
        Parameter validation specification.
    prefer_skip_nested_validation : bool, default=True
        Whether to skip nested validation (for newer sklearn versions).
    *args, **kwargs
        Additional arguments passed to sklearn's validate_params.
    """
    if sklearn_validate_params is None:
        # Fallback - return identity decorator
        def decorator(func):
            return func

        return decorator

    # Check if prefer_skip_nested_validation is required
    sig = inspect.signature(sklearn_validate_params)
    if "prefer_skip_nested_validation" in sig.parameters:
        # Newer version - pass the parameter
        kwargs["prefer_skip_nested_validation"] = prefer_skip_nested_validation

    # Call the actual validate_params
    return sklearn_validate_params(params, *args, **kwargs)


__all__ = [
    "Interval",
    "StrOptions",
    "validate_params",
    "check_is_fitted",
]
