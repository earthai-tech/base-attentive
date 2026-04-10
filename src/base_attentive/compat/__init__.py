"""Compatibility layers for validator support."""

try:
    from sklearn.utils._param_validation import (
        Interval,
        StrOptions,
    )
except ImportError:
    # Fallback for older scikit-learn versions
    from sklearn.utils.validation import (
        Interval,
        StrOptions,
    )

try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:
    def check_is_fitted(estimator, attributes, *, msg=None, all_or_any=all):
        """Simple fallback for check_is_fitted."""
        pass

__all__ = [
    "Interval",
    "StrOptions",
    "check_is_fitted",
]
