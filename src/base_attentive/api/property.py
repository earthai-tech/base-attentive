# SPDX-License-Identifier: Apache-2.0
# Author: BASE-ATTENTIVE Contributors
# Base class for neural network learners

"""Property and base class definitions for NN learners."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any


class NNLearner:
    """Base class for neural network learners.

    Provides parameter management and introspection for NN components.
    """

    @classmethod
    def _get_param_names(cls):
        """Retrieve the names of parameters defined in the constructor."""
        init = getattr(
            cls.__init__, "deprecated_original", cls.__init__
        )
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__} should not have variable positional arguments "
                    f"in the constructor (no *args)."
                )
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters for this learner."""
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if (
                deep
                and hasattr(value, "get_params")
                and not isinstance(value, type)
            ):
                deep_items = value.get_params().items()
                out.update(
                    (key + "__" + k, val) for k, val in deep_items
                )
            out[key] = value
        return out

    def set_params(self, **params: Any) -> NNLearner:
        """Set the parameters of this learner."""
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for {self}. "
                    f"Valid parameters: {sorted(valid_params.keys())}"
                )

            if "__" in key:
                name, sub_name = key.split("__", 1)
                nested_params[name][sub_name] = value
            else:
                setattr(self, key, value)

        for name, sub_params in nested_params.items():
            sub_object = getattr(self, name)
            if hasattr(sub_object, "set_params"):
                sub_object.set_params(**sub_params)

        return self


__all__ = ["NNLearner"]
