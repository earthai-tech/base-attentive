# SPDX-License-Identifier: Apache-2.0
# Author: BASE-ATTENTIVE Contributors
# Base class for neural network learners

"""Property and base class definitions for NN learners."""

from __future__ import annotations

import html
import inspect
from collections import defaultdict
from typing import Any


class NNLearner:
    """Base class for neural network learners.

    Provides parameter management, introspection, and
    a compact pretty-printer for NN components.
    """

    _repr_width = 88
    _repr_indent = 4
    _repr_max_depth = 3
    _repr_max_items = 6
    _repr_max_chars = 1200
    _repr_max_value_chars = 120

    @classmethod
    def _get_param_names(cls):
        """Retrieve constructor parameter names."""
        init = getattr(
            cls.__init__,
            "deprecated_original",
            cls.__init__,
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
                    f"{cls.__name__} should not have "
                    f"variable positional arguments in "
                    f"the constructor (no *args)."
                )

        return sorted(p.name for p in parameters)

    @classmethod
    def _repr_config(cls) -> dict[str, int]:
        """Return representation settings."""
        return {
            "width": cls._repr_width,
            "indent": cls._repr_indent,
            "max_depth": cls._repr_max_depth,
            "max_items": cls._repr_max_items,
            "max_chars": cls._repr_max_chars,
            "max_value_chars": cls._repr_max_value_chars,
        }

    @staticmethod
    def _is_array_like(value: Any) -> bool:
        """Return True for array or tensor-like objects."""
        if isinstance(
            value,
            (str, bytes, bytearray, list, tuple, dict, set),
        ):
            return False

        return hasattr(value, "shape") and hasattr(
            value,
            "dtype",
        )

    @staticmethod
    def _is_learner_like(value: Any) -> bool:
        """Return True for estimator-like objects."""
        return hasattr(
            value, "get_params"
        ) and not inspect.isclass(value)

    @staticmethod
    def _safe_len(value: Any) -> int | None:
        """Safely return len(value)."""
        try:
            return len(value)
        except Exception:
            return None

    @staticmethod
    def _truncate_text(
        text: str,
        *,
        max_chars: int,
    ) -> str:
        """Hard truncate a representation string."""
        text = (
            " ".join(text.split())
            if "\n" not in text
            else text
        )
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _indent_block(
        text: str,
        *,
        spaces: int,
    ) -> str:
        """Indent every line in a block of text."""
        prefix = " " * spaces
        return "\n".join(
            prefix + line for line in text.splitlines()
        )

    @classmethod
    def _safe_shape(cls, value: Any) -> str:
        """Return a stable shape string."""
        shape = getattr(value, "shape", None)

        try:
            if shape is None:
                return "?"
            if isinstance(shape, tuple):
                return str(shape)
            return str(tuple(shape))
        except Exception:
            return str(shape)

    @classmethod
    def _safe_dtype(cls, value: Any) -> str:
        """Return a stable dtype string."""
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            return "?"
        return str(dtype)

    @classmethod
    def _array_summary(cls, value: Any) -> str:
        """Summarize an array or tensor-like object."""
        typ = type(value)
        module = typ.__module__.split(".")[0]
        name = typ.__name__

        if module == "builtins":
            label = name
        else:
            label = f"{module}.{name}"

        shape = cls._safe_shape(value)
        dtype = cls._safe_dtype(value)

        return f"{label}(shape={shape}, dtype={dtype})"

    @classmethod
    def _container_summary(
        cls,
        value: Any,
    ) -> str:
        """Return a short container summary."""
        n_items = cls._safe_len(value)
        label = type(value).__name__

        if n_items is None:
            return f"{label}(...)"
        return f"{label}(len={n_items})"

    @classmethod
    def _callable_name(cls, value: Any) -> str:
        """Return a stable callable name."""
        if inspect.isclass(value):
            return value.__name__

        if hasattr(value, "__name__"):
            return value.__name__

        return type(value).__name__

    @classmethod
    def _iter_items_limited(
        cls,
        value: Any,
        *,
        max_items: int,
    ) -> tuple[list[Any], bool]:
        """Return at most max_items items and truncation flag."""
        items = list(value)
        truncated = len(items) > max_items
        return items[:max_items], truncated

    @classmethod
    def _format_atom(
        cls,
        value: Any,
        *,
        cfg: dict[str, int],
    ) -> str:
        """Format atomic values."""
        if isinstance(value, str):
            text = repr(value)
        elif inspect.isclass(value):
            text = value.__name__
        elif callable(value):
            text = cls._callable_name(value)
        else:
            text = repr(value)

        return cls._truncate_text(
            text,
            max_chars=cfg["max_value_chars"],
        )

    @classmethod
    def _format_sequence(
        cls,
        value: Any,
        *,
        depth: int,
        indent: int,
        visited: set[int],
        cfg: dict[str, int],
    ) -> str:
        """Format lists, tuples, and sets."""
        if depth >= cfg["max_depth"]:
            return cls._container_summary(value)

        if isinstance(value, list):
            left, right = "[", "]"
            seq = value
            preserve_order = True
        elif isinstance(value, tuple):
            left, right = "(", ")"
            seq = value
            preserve_order = True
        else:
            left, right = "{", "}"
            seq = value
            preserve_order = False

        if preserve_order:
            items, truncated = cls._iter_items_limited(
                seq,
                max_items=cfg["max_items"],
            )
        else:
            items, truncated = cls._iter_items_limited(
                sorted(seq, key=repr),
                max_items=cfg["max_items"],
            )

        parts = [
            cls._format_value(
                item,
                depth=depth + 1,
                indent=indent,
                visited=visited,
                cfg=cfg,
            )
            for item in items
        ]

        if truncated:
            parts.append("...")

        inline = f"{left}{', '.join(parts)}{right}"

        if (
            isinstance(value, tuple)
            and len(value) == 1
            and inline == f"({parts[0]})"
        ):
            inline = f"({parts[0]},)"

        if len(inline) <= cfg["width"] - indent:
            return inline

        pad = indent + cfg["indent"]
        body = ",\n".join(
            cls._indent_block(part, spaces=pad)
            for part in parts
        )
        text = f"{left}\n{body}\n{' ' * indent}{right}"

        if (
            isinstance(value, tuple)
            and len(value) == 1
            and not truncated
        ):
            text = text[:-1] + ",)"

        return text

    @classmethod
    def _format_dict(
        cls,
        value: dict[Any, Any],
        *,
        depth: int,
        indent: int,
        visited: set[int],
        cfg: dict[str, int],
    ) -> str:
        """Format dictionaries compactly."""
        if depth >= cfg["max_depth"]:
            return cls._container_summary(value)

        items = list(value.items())
        truncated = len(items) > cfg["max_items"]
        items = items[: cfg["max_items"]]

        parts = []
        for key, val in items:
            key_text = cls._format_value(
                key,
                depth=depth + 1,
                indent=indent,
                visited=visited,
                cfg=cfg,
            )
            val_text = cls._format_value(
                val,
                depth=depth + 1,
                indent=indent,
                visited=visited,
                cfg=cfg,
            )
            parts.append(f"{key_text}: {val_text}")

        if truncated:
            parts.append("...")

        inline = "{" + ", ".join(parts) + "}"
        if len(inline) <= cfg["width"] - indent:
            return inline

        pad = indent + cfg["indent"]
        body = ",\n".join(
            cls._indent_block(part, spaces=pad)
            for part in parts
        )
        return f"{{\n{body}\n{' ' * indent}}}"

    @classmethod
    def _format_learner(
        cls,
        value: Any,
        *,
        depth: int,
        indent: int,
        visited: set[int],
        cfg: dict[str, int],
    ) -> str:
        """Format nested learner-like objects."""
        cls_name = value.__class__.__name__

        if depth >= cfg["max_depth"]:
            return f"{cls_name}(...)"

        try:
            params = value.get_params(deep=False)
        except Exception:
            return f"{cls_name}(...)"

        if hasattr(value, "_get_param_names"):
            try:
                names = value._get_param_names()
            except Exception:
                names = sorted(params)
        else:
            names = sorted(params)

        parts = []
        for name in names:
            val = params.get(name, getattr(value, name, None))
            val_text = cls._format_value(
                val,
                depth=depth + 1,
                indent=indent + cfg["indent"],
                visited=visited,
                cfg=cfg,
            )
            parts.append(f"{name}={val_text}")

        if not parts:
            return f"{cls_name}()"

        inline = f"{cls_name}({', '.join(parts)})"
        if len(inline) <= cfg["width"] - indent:
            return inline

        pad = indent + cfg["indent"]
        body = ",\n".join(
            cls._indent_block(part, spaces=pad)
            for part in parts
        )
        return f"{cls_name}(\n{body}\n{' ' * indent})"

    @classmethod
    def _format_value(
        cls,
        value: Any,
        *,
        depth: int,
        indent: int,
        visited: set[int],
        cfg: dict[str, int],
    ) -> str:
        """Format any value with depth and cycle control."""
        obj_id = id(value)

        if isinstance(
            value,
            (str, bytes, bytearray, int, float, complex),
        ):
            return cls._format_atom(value, cfg=cfg)

        if value is None or isinstance(value, bool):
            return cls._format_atom(value, cfg=cfg)

        if cls._is_array_like(value):
            return cls._array_summary(value)

        if inspect.isclass(value) or callable(value):
            return cls._format_atom(value, cfg=cfg)

        trackable = isinstance(
            value,
            (list, tuple, set, dict),
        ) or cls._is_learner_like(value)

        if trackable and obj_id in visited:
            return "..."

        if trackable:
            visited = set(visited)
            visited.add(obj_id)

        if isinstance(value, dict):
            return cls._format_dict(
                value,
                depth=depth,
                indent=indent,
                visited=visited,
                cfg=cfg,
            )

        if isinstance(value, (list, tuple, set)):
            return cls._format_sequence(
                value,
                depth=depth,
                indent=indent,
                visited=visited,
                cfg=cfg,
            )

        if cls._is_learner_like(value):
            return cls._format_learner(
                value,
                depth=depth,
                indent=indent,
                visited=visited,
                cfg=cfg,
            )

        return cls._truncate_text(
            repr(value),
            max_chars=cfg["max_value_chars"],
        )

    def _repr_text(self) -> str:
        """Return the canonical text representation."""
        cfg = self._repr_config()
        text = self._format_learner(
            self,
            depth=0,
            indent=0,
            visited=set(),
            cfg=cfg,
        )
        return self._truncate_text(
            text,
            max_chars=cfg["max_chars"],
        )

    @staticmethod
    def _repr_html_escape(text: str) -> str:
        """Escape text for safe HTML display."""
        return html.escape(text, quote=True)

    def _repr_html_(self) -> str:
        """Return an HTML-safe notebook representation."""
        text = self._repr_text()
        safe = self._repr_html_escape(text)
        return (
            "<pre style="
            "'white-space:pre-wrap;"
            "word-break:break-word;"
            "margin:0;"
            "font-family:monospace;'>"
            f"{safe}</pre>"
        )

    def __repr__(self) -> str:
        """Return a compact constructor-like repr."""
        return self._repr_text()

    def __str__(self) -> str:
        """Return a readable multi-line summary."""
        cfg = self._repr_config()

        try:
            params = self.get_params(deep=False)
        except Exception:
            return self.__class__.__name__

        names = self._get_param_names()
        if not names:
            return self.__class__.__name__

        lines = [f"{self.__class__.__name__}:"]

        for name in names:
            value = params.get(
                name, getattr(self, name, None)
            )
            text = self._format_value(
                value,
                depth=1,
                indent=cfg["indent"],
                visited=set(),
                cfg=cfg,
            )
            if "\n" in text:
                text = "\n" + self._indent_block(
                    text,
                    spaces=cfg["indent"],
                )
                lines.append(f"  - {name}:{text}")
            else:
                lines.append(f"  - {name}: {text}")

        return "\n".join(lines)

    def get_params(
        self,
        deep: bool = True,
    ) -> dict[str, Any]:
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
                    (key + "__" + k, val)
                    for k, val in deep_items
                )

            out[key] = value

        return out

    def set_params(
        self,
        **params: Any,
    ) -> NNLearner:
        """Set the parameters of this learner."""
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for "
                    f"{self.__class__.__name__}. "
                    f"Valid parameters are: "
                    f"{sorted(valid_params.keys())}"
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
