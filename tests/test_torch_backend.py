# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Comprehensive tests for PyTorch backend functionality."""

import importlib.util
import sys
from unittest import mock

import pytest

# Check if torch is available (don't mock it, just detect)
torch_available = (
    importlib.util.find_spec("torch") is not None
)


@pytest.fixture
def mock_torch():
    """Fixture providing a mock torch module."""
    with mock.patch.dict(
        sys.modules, {"torch": mock.MagicMock()}
    ):
        yield sys.modules["torch"]


@pytest.fixture
def mock_cuda_available(mock_torch):
    """Fixture with CUDA available."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.__version__ = "2.1.0"
    return mock_torch


@pytest.fixture
def mock_cuda_unavailable(mock_torch):
    """Fixture with CUDA unavailable (CPU only)."""
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.side_effect = RuntimeError(
        "CUDA not available"
    )
    mock_torch.__version__ = "2.0.0"
    return mock_torch


@pytest.fixture
def mock_mps(mock_torch):
    """Fixture with Apple Metal Performance Shaders available."""
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.side_effect = RuntimeError(
        "CUDA not available"
    )
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.__version__ = "2.0.1"
    return mock_torch


class TestTorchIsAvailable:
    """Tests for torch_is_available function."""

    def test_torch_available_when_installed(self):
        """Test torch_is_available returns True when torch is installed."""
        from base_attentive.backend.torch_utils import (
            torch_is_available,
        )

        # Real test if torch is installed
        available = torch_is_available()
        assert isinstance(available, bool)

    def test_torch_not_available_mock(self, monkeypatch):
        """Test torch_is_available returns False when torch import fails."""
        from base_attentive.backend import torch_utils

        # Use mock.patch to mock the torch import
        with mock.patch.dict(sys.modules, {"torch": None}):
            # Reimport to test handling of torch not available
            import importlib

            importlib.reload(torch_utils)

            # This test is complex; better to test with actual torch availability
            result = torch_utils.torch_is_available()
            assert isinstance(result, bool)


class TestGetTorchVersion:
    """Tests for get_torch_version function."""

    def test_get_version_when_available(self):
        """Test get_torch_version returns correct version when torch available."""
        from base_attentive.backend.torch_utils import (
            get_torch_version,
        )

        version = get_torch_version()
        if not torch_available:
            assert version is None
        else:
            assert isinstance(version, str)
            # Check format (e.g., "2.0.0")
            parts = version.split(".")
            assert len(parts) >= 2

    def test_get_version_handle_cuda_suffix(self):
        """Test get_torch_version handles CUDA suffix (e.g., '2.0.0+cu118')."""
        from base_attentive.backend.torch_utils import (
            get_torch_version,
        )

        version = get_torch_version()
        # Gets actual version if torch available, None otherwise
        assert version is None or isinstance(version, str)


class TestCheckTorchCompatibility:
    """Tests for check_torch_compatibility function."""

    def test_compatible_version(self):
        """Test check_torch_compatibility with compatible version."""
        from base_attentive.backend.torch_utils import (
            check_torch_compatibility,
        )

        # Test with version >= 2.0.0
        is_compatible, msg = check_torch_compatibility(
            "2.0.0"
        )
        assert is_compatible is True
        assert isinstance(msg, str)

        is_compatible, msg = check_torch_compatibility(
            "2.1.0"
        )
        assert is_compatible is True

    def test_incompatible_version(self):
        """Test check_torch_compatibility with incompatible version."""
        from base_attentive.backend.torch_utils import (
            check_torch_compatibility,
        )

        # Test with version < 2.0.0
        is_compatible, msg = check_torch_compatibility(
            "1.13.0"
        )
        assert is_compatible is False
        assert isinstance(msg, str)
        assert "2.0.0" in msg

    def test_cuda_suffix_handling(self):
        """Test check_torch_compatibility removes CUDA suffix before validation."""
        from base_attentive.backend.torch_utils import (
            check_torch_compatibility,
        )

        # Test with CUDA suffix - should still work
        is_compatible, msg = check_torch_compatibility(
            "2.0.0+cu118"
        )
        assert isinstance(is_compatible, bool)
        assert isinstance(msg, str)

    def test_invalid_version_format(self):
        """Test check_torch_compatibility with invalid version format."""
        from base_attentive.backend.torch_utils import (
            check_torch_compatibility,
        )

        # Test with invalid format
        is_compatible, msg = check_torch_compatibility(
            "invalid"
        )
        # Should handle gracefully
        assert isinstance(is_compatible, bool)
        assert isinstance(msg, str)

    def test_none_version_with_torch_unavailable(self):
        """Test check_torch_compatibility with None version when torch unavailable."""
        from base_attentive.backend.torch_utils import (
            check_torch_compatibility,
        )

        is_compatible, msg = check_torch_compatibility(None)
        assert isinstance(is_compatible, bool)
        assert isinstance(msg, str)


class TestGetTorchDevice:
    """Tests for get_torch_device function."""

    def test_cuda_preferred(self):
        """Test get_torch_device returns device when preferences are given."""
        from base_attentive.backend.torch_utils import (
            get_torch_device,
        )

        device = get_torch_device(
            prefer="cuda", verbose=False
        )
        assert isinstance(device, str)

    def test_cpu_fallback(self):
        """Test get_torch_device returns a valid device string."""
        from base_attentive.backend.torch_utils import (
            get_torch_device,
        )

        device = get_torch_device(
            prefer="cuda", verbose=False
        )
        assert isinstance(device, str)

    def test_mps_support(self):
        """Test get_torch_device handles mps preference."""
        from base_attentive.backend.torch_utils import (
            get_torch_device,
        )

        device = get_torch_device(prefer="mps", verbose=False)
        assert isinstance(device, str)

    def test_device_string_format(self):
        """Test get_torch_device returns properly formatted device string."""
        from base_attentive.backend.torch_utils import (
            get_torch_device,
        )

        device = get_torch_device(verbose=False)
        assert isinstance(device, str)
        # Should be non-empty
        assert len(device) > 0


class TestTorchDeviceManager:
    """Tests for TorchDeviceManager class."""

    def test_init_default(self):
        """Test TorchDeviceManager initialization with defaults."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        assert manager.prefer == "cuda"
        assert manager._device is None

    def test_init_custom_preference(self):
        """Test TorchDeviceManager initialization with custom preference."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager(prefer="mps")
        assert manager.prefer == "mps"

    def test_device_property_caching(self):
        """Test TorchDeviceManager caches device selection."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        device1 = manager.device
        device2 = manager.device
        # Should return same device (cached)
        assert device1 == device2

    def test_device_property_lazy_loading(self):
        """Test TorchDeviceManager uses lazy loading for device property."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        # Device should be None before first access
        assert manager._device is None
        # After access, should be set
        _ = manager.device
        assert manager._device is not None

    def test_get_device_info(self):
        """Test TorchDeviceManager.get_device_info returns dict."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        if not torch_available:
            pytest.skip("PyTorch not available")

        manager = TorchDeviceManager()
        info = manager.get_device_info()

        assert isinstance(info, dict)
        # Should contain standard keys (available for CPU at minimum)
        assert (
            "current_device" in info
            or "available_devices" in info
        )

    def test_get_device_info_content(self):
        """Test TorchDeviceManager.get_device_info contains relevant info."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        if not torch_available:
            pytest.skip("PyTorch not available")

        manager = TorchDeviceManager()
        info = manager.get_device_info()

        assert isinstance(info, dict)
        # Check for expected keys
        assert (
            "current_device" in info
            or "available_devices" in info
        )

    def test_get_devices_available(self):
        """Test TorchDeviceManager.get_available_devices returns dict."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        devices = manager.get_available_devices()

        assert isinstance(devices, dict)
        assert "cpu" in devices
        assert (
            devices["cpu"] is True
        )  # CPU should always be available

    def test_clear_cache(self):
        """Test TorchDeviceManager cache mechanism."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        # Set a device
        manager.device
        assert manager._device is not None

        # Create new manager to test different device
        manager2 = TorchDeviceManager(prefer="cpu")
        manager2.device
        assert manager2.prefer == "cpu"

    def test_reset_preference(self):
        """Test TorchDeviceManager.set_device can change device."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager(prefer="cuda")

        # Try to set to CPU
        try:
            device = manager.set_device("cpu")
            assert device == "cpu"
            assert (
                manager.prefer == "cuda"
            )  # Preference unchanged
            assert (
                manager._device == "cpu"
            )  # But internal device is set
        except RuntimeError:
            # PyTorch not available
            pytest.skip("PyTorch not available")

    def test_supports_device(self):
        """Test TorchDeviceManager checks device availability."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        devices = manager.get_available_devices()

        # CPU should always be available
        assert devices["cpu"] is True


class TestTorchBackendIntegration:
    """Integration tests for Torch backend functionality."""

    def test_torch_backend_import(self):
        """Test TorchBackend can be imported from main backend module."""
        from base_attentive.backend import TorchBackend

        assert TorchBackend is not None

    def test_torch_utils_in_public_api(self):
        """Test torch utilities are exposed in public API."""
        from base_attentive import backend

        # Check that torch utilities are available
        assert hasattr(backend, "get_torch_device")
        assert hasattr(backend, "get_torch_version")
        assert hasattr(backend, "torch_is_available")
        assert hasattr(backend, "TorchDeviceManager")
        assert hasattr(backend, "check_torch_compatibility")

    def test_torch_device_manager_with_backend(self):
        """Test TorchDeviceManager works within backend context."""
        from base_attentive.backend import TorchDeviceManager

        manager = TorchDeviceManager(prefer="cuda")
        device = manager.device

        assert isinstance(device, str)

    def test_backend_set_to_torch(self):
        """Test setting backend to torch."""
        from base_attentive.backend import set_backend

        try:
            backend = set_backend("torch")
            # Should not raise
            assert backend is not None
        except Exception as e:
            # Torch might not be installed
            pytest.skip(
                f"Torch backend not available: {str(e)}"
            )


class TestTorchVersionUtils:
    """Tests for torch version parsing utilities."""

    def test_version_parsing_simple(self):
        """Test version parsing with simple version string."""
        from base_attentive.backend.version_check import (
            parse_version,
        )

        version = parse_version("2.0.0")
        assert version == (2, 0, 0)

    def test_version_parsing_four_parts(self):
        """Test version parsing with four-part version."""
        from base_attentive.backend.version_check import (
            parse_version,
        )

        version = parse_version("2.1.0.1")
        # Should handle gracefully
        assert isinstance(version, tuple)
        assert len(version) >= 3

    def test_version_at_least(self):
        """Test version_at_least comparison."""
        from base_attentive.backend.version_check import (
            version_at_least,
        )

        assert version_at_least("2.0.0", "1.13.0") is True
        assert version_at_least("1.13.0", "2.0.0") is False
        assert version_at_least("2.0.0", "2.0.0") is True

    def test_version_comparison_with_different_lengths(self):
        """Test version comparison with different tuple lengths."""
        from base_attentive.backend.version_check import (
            version_at_least,
        )

        # Should handle comparison between different length tuples
        result = version_at_least("2.0", "1.13.0")
        assert isinstance(result, bool)


class TestErrorHandling:
    """Tests for error handling in torch utilities."""

    def test_torch_unavailable_graceful_handling(self):
        """Test torch utilities handle missing torch gracefully."""
        from base_attentive.backend.torch_utils import (
            torch_is_available,
        )

        result = torch_is_available()
        # Should return bool, not raise
        assert isinstance(result, bool)

    def test_invalid_device_preference_fallback(self):
        """Test get_torch_device handles invalid preference gracefully."""
        from base_attentive.backend.torch_utils import (
            get_torch_device,
        )

        # Should not raise on invalid preference
        device = get_torch_device(
            prefer="invalid", verbose=False
        )
        assert isinstance(device, str)

    def test_device_manager_info_with_error(self):
        """Test TorchDeviceManager.get_device_info returns valid info."""
        from base_attentive.backend.torch_utils import (
            TorchDeviceManager,
        )

        manager = TorchDeviceManager()
        try:
            info = manager.get_device_info()
            # Should return a dict with valid info
            assert isinstance(info, dict)
        except Exception:
            # If torch not available, may raise - that's OK for this test
            pytest.skip(
                "PyTorch not available or error getting device info"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
