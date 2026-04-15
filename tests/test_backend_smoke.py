from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKENDS = ("tensorflow", "jax", "torch")


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_backend_import_smoke(backend_name: str):
    env = dict(os.environ)
    env["BASE_ATTENTIVE_BACKEND"] = backend_name
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(src_dir), env.get("PYTHONPATH")])
    )
    code = (
        "import base_attentive as batt; "
        "print(batt.KERAS_BACKEND); "
        "from base_attentive import BaseAttentive; "
        "print(BaseAttentive.__name__)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "BaseAttentive" in completed.stdout
