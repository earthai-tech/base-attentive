"""Handler utilities for parameter management and deprecation."""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "param_deprecated_message",
    "delegate_on_error",
]


def param_deprecated_message(
    conditions_params_mappings: List[Dict[str, Any]],
    warning_category: type = DeprecationWarning,
) -> Callable:
    """
    Decorator to issue deprecation warnings for specific parameters.

    Parameters
    ----------
    conditions_params_mappings : list of dict
        List of mappings with keys:
        - 'param': parameter name
        - 'condition': callable or lambda to check if warning should be issued
        - 'message': deprecation message
    warning_category : type, optional
        Warning category to use. Default is DeprecationWarning.

    Returns
    -------
    Callable
        Decorator function.
    """

    def decorator(func_or_class):
        if inspect.isclass(func_or_class):
            # Wrap __init__ method
            original_init = func_or_class.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                # Check each parameter mapping
                for mapping in conditions_params_mappings:
                    param_name = mapping.get("param")
                    condition = mapping.get("condition")
                    message = mapping.get("message", f"Parameter '{param_name}' is deprecated.")

                    if param_name in kwargs:
                        value = kwargs[param_name]
                        if callable(condition):
                            should_warn = condition(value)
                        else:
                            should_warn = bool(condition)

                        if should_warn:
                            warnings.warn(message, warning_category, stacklevel=2)

                return original_init(self, *args, **kwargs)

            func_or_class.__init__ = wrapped_init
            return func_or_class
        else:
            # Wrap function
            @functools.wraps(func_or_class)
            def wrapper(*args, **kwargs):
                for mapping in conditions_params_mappings:
                    param_name = mapping.get("param")
                    condition = mapping.get("condition")
                    message = mapping.get("message", f"Parameter '{param_name}' is deprecated.")

                    if param_name in kwargs:
                        value = kwargs[param_name]
                        if callable(condition):
                            should_warn = condition(value)
                        else:
                            should_warn = bool(condition)

                        if should_warn:
                            warnings.warn(message, warning_category, stacklevel=2)

                return func_or_class(*args, **kwargs)

            return wrapper

    return decorator


def delegate_on_error(error_handler: Optional[Callable] = None) -> Callable:
    """
    Decorator to handle errors gracefully.

    Parameters
    ----------
    error_handler : Callable, optional
        Function to handle errors.

    Returns
    -------
    Callable
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    return error_handler(e)
                raise

        return wrapper

    return decorator
