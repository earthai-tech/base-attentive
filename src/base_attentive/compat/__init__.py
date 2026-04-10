"""Compatibility layers for validator support."""

try:
    from sklearn.utils._param_validation import (
        Interval,
        StrOptions,
        validate_params,
    )
except ImportError:
    # Fallback for older scikit-learn versions
    try:
        from sklearn.utils.validation import (
            Interval,
            StrOptions,
        )
        def validate_params(params):
            """Fallback decorator for parameter validation."""
            def decorator(func):
                return func
            return decorator
    except ImportError:
        # Even older versions - provide stubs
        class Interval:
            """Placeholder validator for parameter ranges."""
            def __init__(self, type_, left, right, *, closed="right"):
                self.type_ = type_
                self.left = left
                self.right = right
                self.closed = closed

        class StrOptions:
            """Placeholder validator for string options."""
            def __init__(self, options):
                self.options = set(options) if not isinstance(options, set) else options

        def validate_params(params):
            """Fallback decorator for parameter validation."""
            def decorator(func):
                return func
            return decorator

try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:
    def check_is_fitted(estimator, attributes, *, msg=None, all_or_any=all):
        """Simple fallback for check_is_fitted."""
        pass

__all__ = [
    "Interval",
    "StrOptions",
    "validate_params",
    "check_is_fitted",
]
