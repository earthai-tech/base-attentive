# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Torch-specific backend utilities and device management."""

from __future__ import annotations

import importlib.util
import logging
from typing import Literal, Optional

__all__ = [
    "get_torch_device",
    "torch_is_available",
    "get_torch_version",
    "check_torch_compatibility",
    "TorchDeviceManager",
]

_logger = logging.getLogger(__name__)


def torch_is_available() -> bool:
    """Check if PyTorch is installed and importable.

    Returns
    -------
    bool
        True if PyTorch is available.
    """
    return importlib.util.find_spec("torch") is not None


def get_torch_version() -> Optional[str]:
    """Get installed PyTorch version.

    Returns
    -------
    str or None
        Version string (e.g., "2.0.1") or None if not installed.
    """
    if not torch_is_available():
        return None

    try:
        import torch

        return torch.__version__.split("+")[0]  # Remove CUDA suffix if present
    except Exception:
        return None


def check_torch_compatibility(torch_version: Optional[str] = None) -> tuple[bool, str]:
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
        major, minor, patch = map(int, torch_version.split(".")[:3])
    except (ValueError, IndexError):
        return (False, f"Could not parse PyTorch version: {torch_version}")

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
            _logger.warning("PyTorch not available, using CPU")
        return "cpu"

    import torch

    # Try preferred device first
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        if verbose:
            _logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return str(device)

    if prefer == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            _logger.info("Using MPS device (Apple Metal)")
        return str(device)

    # Fallback to CPU
    if verbose:
        _logger.info("Using CPU device")
    return "cpu"


class TorchDeviceManager:
    """Utility class for managing PyTorch device selection and configuration."""

    def __init__(self, prefer: Literal["cuda", "cpu", "mps"] = "cuda"):
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
            self._device = get_torch_device(self.prefer, verbose=False)
        return self._device

    def set_device(self, device: str | Literal["cuda", "cpu", "mps"]) -> str:
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

        import torch

        # Validate device
        try:
            torch.device(device)  # Validate
        except RuntimeError as e:
            raise ValueError(f"Invalid device '{device}': {e}") from e

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

        import torch

        return {
            "cuda": torch.cuda.is_available(),
            "cpu": True,
            "mps": torch.backends.mps.is_available(),
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

        import torch

        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None,
            "current_device": self.device,
            "available_devices": self.get_available_devices(),
        }

        # Add CUDA device details
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["cuda_device_memory_mb"] = [
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
                for i in range(torch.cuda.device_count())
            ]

        # Add MPS info
        if torch.backends.mps.is_available():
            info["mps_available"] = True

        return info

    def reset_cache(self) -> None:
        """Clear PyTorch cache to free memory."""
        if not torch_is_available():
            _logger.warning("PyTorch not available")
            return

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _logger.info("CUDA cache cleared")

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            _logger.info("MPS cache cleared")
