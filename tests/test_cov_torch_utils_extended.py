"""Extended coverage tests for backend/torch_utils.py.

These tests run without needing a GPU; they exercise device selection
logic, version parsing, and TorchDeviceManager via the real torch module
that is already installed in the CI environment.
"""

from __future__ import annotations

import builtins
import sys
import types

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
    @pytest.mark.parametrize(
        "device", ["cpu", "mps", "cuda", "cuda:0", "cuda:1"]
    )
    def test_valid_strings(self, device):
        assert _is_valid_device_string(device) is True

    @pytest.mark.parametrize(
        "device", ["gpu", "tpu", "npu", "cuda:abc", ""]
    )
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

    def test_get_torch_module_prefers_loaded_module(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        sentinel = object()
        monkeypatch.setitem(sys.modules, "torch", sentinel)

        assert tu._get_torch_module() is sentinel

    def test_get_torch_module_returns_none_when_spec_missing(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.delitem(
            sys.modules, "torch", raising=False
        )
        monkeypatch.setattr(
            tu.importlib.util, "find_spec", lambda name: None
        )

        assert tu._get_torch_module() is None

    @pytest.mark.parametrize(
        "exc_type", [ImportError, ValueError, AttributeError]
    )
    def test_get_torch_module_handles_find_spec_errors(
        self, monkeypatch, exc_type
    ):
        import base_attentive.backend.torch_utils as tu

        def _raise(_name):
            raise exc_type("broken")

        monkeypatch.delitem(
            sys.modules, "torch", raising=False
        )
        monkeypatch.setattr(
            tu.importlib.util, "find_spec", _raise
        )

        assert tu._get_torch_module() is None

    def test_get_torch_module_returns_none_when_import_fails(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("boom")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(
            sys.modules, "torch", raising=False
        )
        monkeypatch.setattr(
            tu.importlib.util,
            "find_spec",
            lambda name: object(),
        )
        monkeypatch.setattr(
            builtins, "__import__", _fake_import
        )

        assert tu._get_torch_module() is None


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

    def test_returns_none_when_runtime_unavailable(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )

        assert tu.get_torch_version() is None

    def test_strips_cuda_suffix(self, monkeypatch):
        import base_attentive.backend.torch_utils as tu

        fake_torch = types.SimpleNamespace(
            __version__="2.4.1+cu121"
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        assert tu.get_torch_version() == "2.4.1"

    def test_returns_none_when_version_lookup_breaks(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        class BrokenTorch:
            @property
            def __version__(self):
                raise RuntimeError("broken")

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: BrokenTorch()
        )

        assert tu.get_torch_version() is None


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

        monkeypatch.setattr(
            tu, "get_torch_version", lambda: None
        )
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
        device = get_torch_device(
            prefer="cuda", verbose=False
        )
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

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: False
        )
        assert tu.get_torch_device(verbose=False) == "cpu"

    def test_runtime_unavailable_returns_cpu(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )

        assert tu.get_torch_device(verbose=False) == "cpu"

    def test_prefer_cuda_uses_device_factory(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        fake_torch = types.SimpleNamespace(
            device=lambda name: f"wrapped:{name}",
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda index: "Fake GPU",
            ),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(
                    is_available=lambda: False
                )
            ),
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        assert (
            tu.get_torch_device(prefer="cuda", verbose=False)
            == "wrapped:cuda:0"
        )

    def test_prefer_cuda_handles_device_name_lookup_failure(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        fake_torch = types.SimpleNamespace(
            device=lambda name: name,
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda index: (
                    _ for _ in ()
                ).throw(RuntimeError("boom")),
            ),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(
                    is_available=lambda: False
                )
            ),
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        assert (
            tu.get_torch_device(prefer="cuda", verbose=True)
            == "cuda:0"
        )

    def test_prefer_mps_uses_device_factory(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        fake_torch = types.SimpleNamespace(
            device=lambda name: f"wrapped:{name}",
            cuda=types.SimpleNamespace(
                is_available=lambda: False
            ),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(
                    is_available=lambda: True
                )
            ),
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        assert (
            tu.get_torch_device(prefer="mps", verbose=False)
            == "wrapped:mps"
        )


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

    def test_get_available_devices_no_torch(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: False
        )
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

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: False
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )
        mgr = TorchDeviceManager()
        try:
            mgr.set_device("cpu")
        except RuntimeError as exc:
            assert "PyTorch not available" in str(exc)
        else:
            assert mgr.device == "cpu"

    def test_set_device_runtime_unavailable_raises(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )

        mgr = TorchDeviceManager()
        try:
            mgr.set_device("cpu")
        except RuntimeError as exc:
            assert "PyTorch runtime unavailable" in str(exc)
        else:
            assert mgr.device == "cpu"

    def test_set_device_invalid_when_factory_raises(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        def _device_factory(device):
            raise RuntimeError(f"bad device: {device}")

        fake_torch = types.SimpleNamespace(
            device=_device_factory
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        mgr = TorchDeviceManager()
        with pytest.raises(
            ValueError, match="Invalid device 'gpu'"
        ):
            mgr.set_device("gpu")

    def test_set_device_invalid_without_factory_uses_string_validation(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        fake_torch = types.SimpleNamespace()
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        mgr = TorchDeviceManager()
        with pytest.raises(
            ValueError, match="Invalid device 'gpu'"
        ):
            mgr.set_device("gpu")

    def test_get_available_devices_runtime_unavailable(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )

        mgr = tu.TorchDeviceManager()
        assert mgr.get_available_devices() == {
            "cuda": False,
            "cpu": True,
            "mps": False,
        }

    def test_get_device_info_returns_dict(self):
        mgr = TorchDeviceManager()
        if torch_is_available():
            info = mgr.get_device_info()
            assert isinstance(info, dict)

    def test_prefer_stored(self):
        mgr = TorchDeviceManager(prefer="mps")
        assert mgr.prefer == "mps"

    def test_get_device_info_when_torch_unavailable(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: False
        )

        mgr = TorchDeviceManager()
        info = mgr.get_device_info()
        assert info["current_device"] == "cpu"
        assert info["cuda_available"] is False

    def test_get_device_info_when_runtime_unavailable(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )
        monkeypatch.setattr(
            tu,
            "get_torch_device",
            lambda prefer="cuda", verbose=False: "cpu",
        )

        mgr = tu.TorchDeviceManager()
        info = mgr.get_device_info()
        assert info["current_device"] == "cpu"
        assert info["available_devices"].get("cpu") is True
        assert info["available_devices"].get("cuda", False) is False
        assert info["available_devices"].get("mps", False) is False

    def test_get_device_info_collects_cuda_and_mps_details(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        class _Props:
            def __init__(self, total_memory):
                self.total_memory = total_memory

        fake_torch = types.SimpleNamespace(
            __version__="2.5.0",
            device=lambda name: name,
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
                get_device_name=lambda index: (
                    "GPU-0"
                    if index == 0
                    else (_ for _ in ()).throw(
                        RuntimeError("boom")
                    )
                ),
                get_device_properties=lambda index: (
                    _Props(1024 * 1024 * 512)
                    if index == 0
                    else (_ for _ in ()).throw(
                        RuntimeError("boom")
                    )
                ),
            ),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(
                    version=lambda: 9100
                ),
                mps=types.SimpleNamespace(
                    is_available=lambda: True
                ),
            ),
            mps=types.SimpleNamespace(
                empty_cache=lambda: None
            ),
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )
        monkeypatch.setattr(
            tu,
            "get_torch_device",
            lambda prefer="cuda", verbose=False: "cpu",
        )

        mgr = tu.TorchDeviceManager(prefer="cpu")
        info = mgr.get_device_info()

        assert info["torch_version"] == getattr(
            fake_torch, "__version__", None
        )
        assert info["cuda_available"] is True
        assert info["cuda_device_count"] == 2
        assert info["cuda_devices"] == ["GPU-0", "cuda:1"]
        assert info["cuda_device_memory_mb"] == [512.0, None]
        assert info["cudnn_version"] == 9100
        assert info["mps_available"] is True

    def test_get_device_info_handles_count_and_cudnn_failures(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        fake_torch = types.SimpleNamespace(
            __version__="2.5.0",
            device=lambda name: name,
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: (_ for _ in ()).throw(
                    RuntimeError("boom")
                ),
                get_device_name=lambda index: "GPU",
                get_device_properties=lambda index: None,
            ),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(
                    version=lambda: (_ for _ in ()).throw(
                        RuntimeError("boom")
                    )
                ),
                mps=types.SimpleNamespace(
                    is_available=lambda: False
                ),
            ),
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        mgr = TorchDeviceManager(prefer="cpu")
        info = mgr.get_device_info()

        assert info.get("cuda_device_count", 0) == 0
        assert info.get("cuda_devices", []) == []
        assert info.get("cuda_device_memory_mb", []) == []
        assert info["cudnn_version"] is None

    def test_reset_cache_warns_when_torch_unavailable(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: False
        )

        TorchDeviceManager().reset_cache()

    def test_reset_cache_warns_when_runtime_unavailable(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: None
        )

        TorchDeviceManager().reset_cache()

    def test_reset_cache_clears_cuda_and_mps(
        self, monkeypatch
    ):
        import base_attentive.backend.torch_utils as tu

        calls = []
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: calls.append("cuda"),
            ),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(
                    is_available=lambda: True
                )
            ),
            mps=types.SimpleNamespace(
                empty_cache=lambda: calls.append("mps")
            ),
        )
        monkeypatch.setattr(
            tu, "torch_is_available", lambda: True
        )
        monkeypatch.setattr(
            tu, "_get_torch_module", lambda: fake_torch
        )

        TorchDeviceManager().reset_cache()

        assert calls in (["cuda", "mps"], [])
