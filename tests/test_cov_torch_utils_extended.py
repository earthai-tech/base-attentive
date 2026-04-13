"""Extended coverage tests for backend/torch_utils.py.

These tests run without needing a GPU; they exercise device selection
logic, version parsing, and TorchDeviceManager via the real torch module
that is already installed in the CI environment.
"""

from __future__ import annotations

import pytest

from base_attentive.backend.torch_utils import (
    TorchDeviceManager,
    _cuda_is_available,
    _get_torch_module,
    _is_valid_device_string,
    _mps_is_available,
    check_torch_compatibility,
    get_torch_device,
    get_torch_version,
    torch_is_available,
)


# ---------------------------------------------------------------------------
# _is_valid_device_string
# ---------------------------------------------------------------------------

class TestIsValidDeviceString:
    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:0", "cuda:1"])
    def test_valid_strings(self, device):
        assert _is_valid_device_string(device) is True

    @pytest.mark.parametrize("device", ["gpu", "tpu", "npu", "cuda:abc", ""])
    def test_invalid_strings(self, device):
        assert _is_valid_device_string(device) is False


# ---------------------------------------------------------------------------
# torch_is_available / _get_torch_module
# ---------------------------------------------------------------------------

class TestTorchIsAvailable:
    def test_returns_bool(self):
        result = torch_is_available()
        assert isinstance(result, bool)

    def test_module_returned_when_available(self):
        if torch_is_available():
            mod = _get_torch_module()
            assert mod is not None
        else:
            assert _get_torch_module() is None


# ---------------------------------------------------------------------------
# get_torch_version
# ---------------------------------------------------------------------------

class TestGetTorchVersion:
    def test_returns_string_or_none(self):
        v = get_torch_version()
        assert v is None or isinstance(v, str)

    def test_version_format_when_available(self):
        v = get_torch_version()
        if v is not None:
            parts = v.split(".")
            assert len(parts) >= 2
            assert parts[0].isdigit()


# ---------------------------------------------------------------------------
# check_torch_compatibility
# ---------------------------------------------------------------------------

class TestCheckTorchCompatibility:
    def test_returns_tuple(self):
        ok, msg = check_torch_compatibility()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_explicit_compatible_version(self):
        ok, msg = check_torch_compatibility("2.0.0")
        assert ok is True
        assert "compatible" in msg.lower()

    def test_explicit_old_version(self):
        ok, msg = check_torch_compatibility("1.9.0")
        assert ok is False
        assert "not supported" in msg.lower()

    def test_explicit_none_version(self):
        # Falls back to auto-detect
        ok, msg = check_torch_compatibility(None)
        assert isinstance(ok, bool)

    def test_unparseable_version(self):
        ok, msg = check_torch_compatibility("x.y.z")
        assert ok is False
        assert "parse" in msg.lower() or "Could not" in msg

    def test_none_when_torch_not_installed(self, monkeypatch):
        import base_attentive.backend.torch_utils as tu
        monkeypatch.setattr(tu, "get_torch_version", lambda: None)
        ok, msg = tu.check_torch_compatibility()
        assert ok is False
        assert "not installed" in msg.lower()


# ---------------------------------------------------------------------------
# _cuda_is_available / _mps_is_available
# ---------------------------------------------------------------------------

class TestDeviceAvailability:
    def test_cuda_check_no_torch(self):
        class FakeTorch:
            cuda = None
        assert _cuda_is_available(FakeTorch()) is False

    def test_mps_check_no_torch(self):
        class FakeTorch:
            backends = None
        assert _mps_is_available(FakeTorch()) is False

    def test_cuda_check_raises(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                raise RuntimeError("no CUDA")
        class FakeTorch:
            cuda = FakeCuda
        assert _cuda_is_available(FakeTorch()) is False

    def test_mps_check_raises(self):
        class FakeMPS:
            @staticmethod
            def is_available():
                raise RuntimeError("no MPS")
        class FakeBackends:
            mps = FakeMPS
        class FakeTorch:
            backends = FakeBackends
        assert _mps_is_available(FakeTorch()) is False

    def test_real_module(self):
        torch = _get_torch_module()
        if torch is not None:
            result = _cuda_is_available(torch)
            assert isinstance(result, bool)
            result = _mps_is_available(torch)
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# get_torch_device
# ---------------------------------------------------------------------------

class TestGetTorchDevice:
    def test_returns_string(self):
        device = get_torch_device(verbose=False)
        assert isinstance(device, str)

    def test_prefer_cpu_always_cpu(self):
        device = get_torch_device(prefer="cpu", verbose=False)
        assert device == "cpu"

    def test_prefer_cuda_falls_back(self):
        device = get_torch_device(prefer="cuda", verbose=False)
        assert isinstance(device, str)

    def test_prefer_mps_returns_string(self):
        device = get_torch_device(prefer="mps", verbose=False)
        assert isinstance(device, str)

    def test_verbose_true_no_error(self):
        # Just verifies no exception is raised with verbose=True
        device = get_torch_device(verbose=True)
        assert isinstance(device, str)

    def test_no_torch_returns_cpu(self, monkeypatch):
        import base_attentive.backend.torch_utils as tu
        monkeypatch.setattr(tu, "torch_is_available", lambda: False)
        assert tu.get_torch_device(verbose=False) == "cpu"


# ---------------------------------------------------------------------------
# TorchDeviceManager
# ---------------------------------------------------------------------------

class TestTorchDeviceManager:
    def test_device_property_returns_string(self):
        mgr = TorchDeviceManager(prefer="cpu")
        assert isinstance(mgr.device, str)

    def test_device_property_cached(self):
        mgr = TorchDeviceManager(prefer="cpu")
        d1 = mgr.device
        d2 = mgr.device
        assert d1 == d2

    def test_get_available_devices_keys(self):
        mgr = TorchDeviceManager()
        avail = mgr.get_available_devices()
        assert "cuda" in avail
        assert "cpu" in avail
        assert "mps" in avail
        assert avail["cpu"] is True

    def test_get_available_devices_no_torch(self, monkeypatch):
        import base_attentive.backend.torch_utils as tu
        monkeypatch.setattr(tu, "torch_is_available", lambda: False)
        mgr = TorchDeviceManager()
        avail = mgr.get_available_devices()
        assert avail["cpu"] is True
        assert avail["cuda"] is False

    def test_set_device_cpu(self):
        mgr = TorchDeviceManager()
        result = mgr.set_device("cpu")
        assert result == "cpu"
        assert mgr.device == "cpu"

    def test_set_device_no_torch_raises(self, monkeypatch):
        import base_attentive.backend.torch_utils as tu
        monkeypatch.setattr(tu, "torch_is_available", lambda: False)
        mgr = TorchDeviceManager()
        with pytest.raises(RuntimeError, match="not available"):
            mgr.set_device("cpu")

    def test_get_device_info_returns_dict(self):
        mgr = TorchDeviceManager()
        if torch_is_available():
            info = mgr.get_device_info()
            assert isinstance(info, dict)

    def test_prefer_stored(self):
        mgr = TorchDeviceManager(prefer="mps")
        assert mgr.prefer == "mps"
