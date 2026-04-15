# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Torch-specific backend utilities and device management."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from typing import Literal, Optional

__all__ = [
    "get_torch_device",
    "torch_is_available",
    "get_torch_version",
    "check_torch_compatibility",
    "TorchDeviceManager",
]

_logger = logging.getLogger(__name__)

_CUDA_DEVICE_RE = re.compile(r"^cuda(?::\d+)?$")


def _get_torch_module():
    """Return the loaded/imported torch module when available."""
    loaded = sys.modules.get("torch")
    if loaded is not None:
        return loaded

    try:
        if importlib.util.find_spec("torch") is None:
            return None
    except (ImportError, ValueError, AttributeError):
        return None

    try:
        import torch

        return torch
    except Exception:
        return None


def _cuda_is_available(torch_module) -> bool:
    """Safely check CUDA availability on a torch-like module."""
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _mps_is_available(torch_module) -> bool:
    """Safely check MPS availability on a torch-like module."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _is_valid_device_string(device: str) -> bool:
    """Validate common torch device string formats without importing torch."""
    return device in {"cpu", "mps"} or bool(
        _CUDA_DEVICE_RE.fullmatch(device)
    )


def torch_is_available() -> bool:
    """Check if PyTorch is installed and importable.

    Returns
    -------
    bool
        True if PyTorch is available.
    """
    return _get_torch_module() is not None


def get_torch_version() -> Optional[str]:
    """Get installed PyTorch version.

    Returns
    -------
    str or None
        Version string (e.g., "2.0.1") or None if not installed.
    """
    if not torch_is_available():
        return None

    torch = _get_torch_module()
    if torch is None:
        return None

    try:
        return str(torch.__version__).split("+")[
            0
        ]  # Remove CUDA suffix if present
    except Exception:
        return None


def check_torch_compatibility(
    torch_version: Optional[str] = None,
) -> tuple[bool, str]:
    """Check if installed PyTorch version is compatible with BaseAttentive.

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
    Current compatibility: PyTorch >= 2.0.0
    """
    if torch_version is None:
        torch_version = get_torch_version()

    if torch_version is None:
        return (False, "PyTorch not installed")

    # Minimum supported version
    try:
        major, minor, patch = map(
            int, torch_version.split(".")[:3]
        )
    except (ValueError, IndexError):
        return (
            False,
            f"Could not parse PyTorch version: {torch_version}",
        )

    if (major, minor, patch) < (2, 0, 0):
        return (
            False,
            f"PyTorch {torch_version} is not supported. Minimum required: 2.0.0",
        )

    return (True, f"PyTorch {torch_version} is compatible")


def get_torch_device(
    prefer: Literal["cuda", "cpu", "mps"] = "cuda",
    verbose: bool = True,
) -> str:
    """Get the best available device for PyTorch computations.

    Parameters
    ----------
    prefer : {'cuda', 'cpu', 'mps'}, default='cuda'
        Preferred device type.
        - 'cuda': NVIDIA GPU (with CUDA support)
        - 'cpu': CPU
        - 'mps': Apple Metal Performance Shaders (macOS)

    verbose : bool, default=True
        Whether to log device selection info.

    Returns
    -------
    str
        Device string for use with PyTorch (e.g., 'cuda:0', 'cpu').

    Examples
    --------
    >>> device = get_torch_device()
    >>> # 'cuda:0' if available, else 'cpu'
    >>> device = get_torch_device(prefer="cpu")
    >>> # 'cpu' always
    """
    if not torch_is_available():
        if verbose:
            _logger.warning(
                "PyTorch not available, using CPU"
            )
        return "cpu"

    torch = _get_torch_module()
    if torch is None:
        if verbose:
            _logger.warning(
                "PyTorch runtime unavailable, using CPU"
            )
        return "cpu"

    # Try preferred device first
    if prefer == "cuda" and _cuda_is_available(torch):
        device_factory = getattr(torch, "device", None)
        device = (
            device_factory("cuda:0")
            if callable(device_factory)
            else "cuda:0"
        )
        if verbose:
            get_name = getattr(
                getattr(torch, "cuda", None),
                "get_device_name",
                None,
            )
            device_name = "cuda:0"
            if callable(get_name):
                try:
                    device_name = get_name(0)
                except Exception:
                    pass
            _logger.info(f"Using CUDA device: {device_name}")
        return str(device)

    if prefer == "mps" and _mps_is_available(torch):
        device_factory = getattr(torch, "device", None)
        device = (
            device_factory("mps")
            if callable(device_factory)
            else "mps"
        )
        if verbose:
            _logger.info("Using MPS device (Apple Metal)")
        return str(device)

    # Fallback to CPU
    if verbose:
        _logger.info("Using CPU device")
    return "cpu"


class TorchDeviceManager:
    """Utility class for managing PyTorch device selection and configuration."""

    def __init__(
        self, prefer: Literal["cuda", "cpu", "mps"] = "cuda"
    ):
        """Initialize device manager.

        Parameters
        ----------
        prefer : {'cuda', 'cpu', 'mps'}, default='cuda'
            Preferred device type.
        """
        self.prefer = prefer
        self._device = None

    @property
    def device(self) -> str:
        """Get the selected device."""
        if self._device is None:
            self._device = get_torch_device(
                self.prefer, verbose=False
            )
        return self._device

    def set_device(
        self, device: str | Literal["cuda", "cpu", "mps"]
    ) -> str:
        """Set the device explicitly.

        Parameters
        ----------
        device : str
            Device string or name.

        Returns
        -------
        str
            The set device string.
        """
        if not torch_is_available():
            raise RuntimeError("PyTorch not available")

        torch = _get_torch_module()
        if torch is None:
            raise RuntimeError("PyTorch runtime unavailable")

        # Validate device
        device_factory = getattr(torch, "device", None)
        try:
            if callable(device_factory):
                device_factory(device)  # Validate
            elif not _is_valid_device_string(device):
                raise ValueError(
                    f"Unsupported device '{device}'"
                )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            raise ValueError(
                f"Invalid device '{device}': {e}"
            ) from e

        self._device = device
        _logger.info(f"Device set to: {device}")
        return self._device

    def get_available_devices(self) -> dict[str, bool]:
        """Get availability of different device types.

        Returns
        -------
        dict
            Mapping of device types to availability.
        """
        if not torch_is_available():
            return {
                "cuda": False,
                "cpu": True,
                "mps": False,
            }

        torch = _get_torch_module()
        if torch is None:
            return {
                "cuda": False,
                "cpu": True,
                "mps": False,
            }

        return {
            "cuda": _cuda_is_available(torch),
            "cpu": True,
            "mps": _mps_is_available(torch),
        }

    def get_device_info(self) -> dict:
        """Get detailed information about available devices.

        Returns
        -------
        dict
            Device information including GPU count, names, memory, etc.
        """
        if not torch_is_available():
            return {
                "available_devices": {"cpu": True},
                "cuda_available": False,
                "current_device": "cpu",
            }

        torch = _get_torch_module()
        if torch is None:
            return {
                "available_devices": {"cpu": True},
                "cuda_available": False,
                "current_device": "cpu",
            }

        cuda_available = _cuda_is_available(torch)

        info = {
            "torch_version": getattr(
                torch, "__version__", None
            ),
            "cuda_available": cuda_available,
            "cudnn_version": None,
            "current_device": self.device,
            "available_devices": self.get_available_devices(),
        }

        backends = getattr(torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None)
        cudnn_version = getattr(cudnn, "version", None)
        if cuda_available and callable(cudnn_version):
            try:
                info["cudnn_version"] = cudnn_version()
            except Exception:
                info["cudnn_version"] = None

        # Add CUDA device details
        if cuda_available:
            cuda = getattr(torch, "cuda", None)
            device_count = getattr(cuda, "device_count", None)
            get_name = getattr(cuda, "get_device_name", None)
            get_props = getattr(
                cuda, "get_device_properties", None
            )

            try:
                count = (
                    int(device_count())
                    if callable(device_count)
                    else 0
                )
            except Exception:
                count = 0

            info["cuda_device_count"] = count
            info["cuda_devices"] = []
            info["cuda_device_memory_mb"] = []

            for i in range(count):
                if callable(get_name):
                    try:
                        info["cuda_devices"].append(
                            get_name(i)
                        )
                    except Exception:
                        info["cuda_devices"].append(
                            f"cuda:{i}"
                        )
                else:
                    info["cuda_devices"].append(f"cuda:{i}")

                if callable(get_props):
                    try:
                        total_memory = get_props(
                            i
                        ).total_memory
                        info["cuda_device_memory_mb"].append(
                            total_memory / 1024 / 1024
                        )
                    except Exception:
                        info["cuda_device_memory_mb"].append(
                            None
                        )
                else:
                    info["cuda_device_memory_mb"].append(None)

        # Add MPS info
        if _mps_is_available(torch):
            info["mps_available"] = True

        return info

    def reset_cache(self) -> None:
        """Clear PyTorch cache to free memory."""
        if not torch_is_available():
            _logger.warning("PyTorch not available")
            return

        torch = _get_torch_module()
        if torch is None:
            _logger.warning("PyTorch runtime unavailable")
            return

        if _cuda_is_available(torch):
            empty_cache = getattr(
                getattr(torch, "cuda", None),
                "empty_cache",
                None,
            )
            if callable(empty_cache):
                empty_cache()
            _logger.info("CUDA cache cleared")

        if hasattr(torch, "mps") and _mps_is_available(torch):
            empty_cache = getattr(
                torch.mps, "empty_cache", None
            )
            if callable(empty_cache):
                empty_cache()
            _logger.info("MPS cache cleared")
