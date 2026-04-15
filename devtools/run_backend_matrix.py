from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_step5_backend_matrix.py"


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(ROOT / "src"), env.get("PYTHONPATH")],
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q"],
        cwd=ROOT,
        env=env,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
