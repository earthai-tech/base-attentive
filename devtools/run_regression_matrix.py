from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/test_constructor_parity.py",
    "tests/test_config_roundtrip.py",
    "tests/test_output_shapes.py",
    "tests/test_backend_smoke.py",
    "tests/test_step5_backend_matrix.py",
]


def main() -> int:
    failures = 0
    for test_path in TESTS:
        print(f"\\n=== Running {test_path} ===")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-q"],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            failures += 1
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
