# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Version checking utilities for backends."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import sys
from typing import Optional, Tuple

__all__ = [
    "get_backend_version",
    "check_tensorflow_compatibility",
    "check_torch_compatibility",
    "parse_version",
    "version_at_least",
]

_logger = logging.getLogger(__name__)


def parse_version(version_string: str) -> Tuple[int, ...]:
    """Parse a version string to a tuple of integers.

    Examples
    --------
    >>> parse_version("2.15.0")
    (2, 15, 0)
    >>> parse_version("3.0.0rc1")
    (3, 0, 0)
    """
    # Remove rc, alpha, beta suffixes
    version_string = version_string.split("-")[0]
    version_string = version_string.split("rc")[0]
    version_string = version_string.split("a")[0]
    version_string = version_string.split("b")[0]

    try:
        return tuple(int(x) for x in version_string.split(".")[:3])
    except (ValueError, IndexError):
        _logger.warning(f"Could not parse version string: {version_string}")
        return (0, 0, 0)


def version_at_least(version: str, min_required: Tuple[int, ...] | str) -> bool:
    """Check if version is at least the minimum required version.

    Parameters
    ----------
    version : str
        The version string to check.
    min_required : tuple or str
        The minimum required version as tuple or string.

    Returns
    -------
    bool
        True if version >= min_required.
    """
    parsed = parse_version(version)
    if isinstance(min_required, str):
        min_required = parse_version(min_required)

    return parsed >= min_required


def get_backend_version(backend_name: str) -> Optional[str]:
    """Get the version of an installed backend.

    Parameters
    ----------
    backend_name : str
        Name of the backend ('tensorflow', 'jax', 'torch').

    Returns
    -------
    str or None
        Version string if available, None otherwise.
    """
    module_map = {
        "tensorflow": "tensorflow",
        "tf": "tensorflow",
        "jax": "jax",
        "torch": "torch",
        "pytorch": "torch",
    }

    module_name = module_map.get(backend_name.lower())
    if not module_name:
        return None

    # Check if module exists
    if importlib.util.find_spec(module_name) is None:
        return None

    loaded_module = sys.modules.get(module_name)
    if loaded_module is not None:
        return getattr(loaded_module, "__version__", None)

    distribution_map = {
        "tensorflow": ("tensorflow", "tensorflow-cpu", "tensorflow-intel"),
        "jax": ("jax",),
        "torch": ("torch",),
    }

    for dist_name in distribution_map.get(module_name, (module_name,)):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def check_tensorflow_compatibility(
    tf_version: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check if installed TensorFlow is compatible with BaseAttentive.

    Parameters
    ----------
    tf_version : str, optional
        TensorFlow version string. If None, will try to detect.

    Returns
    -------
    tuple
        (is_compatible, message)
    """
    if tf_version is None:
        tf_version = get_backend_version("tensorflow")

    if tf_version is None:
        return (False, "TensorFlow not installed")

    # Minimum supported version
    min_version = (2, 10, 0)

    if not version_at_least(tf_version, min_version):
        return (
            False,
            f"TensorFlow {tf_version} is not supported. Minimum required: 2.10.0",
        )

    return (True, f"TensorFlow {tf_version} is compatible")


def check_torch_compatibility(torch_version: Optional[str] = None) -> Tuple[bool, str]:
    """Check if installed PyTorch is compatible with BaseAttentive.

    Parameters
    ----------
    torch_version : str, optional
        PyTorch version string. If None, will try to detect.

    Returns
    -------
    tuple
        (is_compatible, message)

    Notes
    -----
    Minimum supported PyTorch version: 2.0.0
    BaseAttentive uses Keras 3 PyTorch backend which requires PyTorch >= 2.0
    """
    if torch_version is None:
        torch_version = get_backend_version("torch")

    if torch_version is None:
        return (False, "PyTorch not installed")

    # Parse and remove CUDA suffix (e.g., "2.0.1+cu118" -> "2.0.1")
    torch_version_clean = torch_version.split("+")[0]

    # Minimum supported version
    min_version = (2, 0, 0)

    if not version_at_least(torch_version_clean, min_version):
        return (
            False,
            f"PyTorch {torch_version} is not supported. Minimum required: 2.0.0",
        )

    return (True, f"PyTorch {torch_version} is compatible")
