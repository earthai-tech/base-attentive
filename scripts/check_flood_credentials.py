"""Secret-safe credential checks for flood data providers."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / "data" / "flood" / ".env.local"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def check_noaa(timeout: int) -> bool:
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        print("NOAA=missing")
        return False
    request = urllib.request.Request(
        "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets?limit=1",
        headers={"token": token, "User-Agent": "base-attentive-flood-paper/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ok = 200 <= response.status < 300
    except Exception as exc:  # noqa: BLE001
        print(f"NOAA=failed ({type(exc).__name__})")
        return False
    print("NOAA=ok" if ok else "NOAA=failed")
    return ok


def check_cds() -> bool:
    if not os.environ.get("CDSAPI_URL") or not os.environ.get("CDSAPI_KEY"):
        print("CDS=missing")
        return False
    try:
        import cdsapi  # noqa: PLC0415

        cdsapi.Client(
            url=os.environ["CDSAPI_URL"],
            key=os.environ["CDSAPI_KEY"],
            quiet=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"CDS=failed ({type(exc).__name__})")
        return False
    print("CDS=client_ok")
    return True


def check_gee() -> bool:
    try:
        import ee  # noqa: PLC0415

        project = os.environ.get("EARTHENGINE_PROJECT") or "fair-future-496413-f5"
        ee.Initialize(project=project)
        count = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").size().getInfo()
    except Exception as exc:  # noqa: BLE001
        print(f"GEE=failed ({type(exc).__name__})")
        return False
    print(f"GEE=ok events={count}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)

    load_env_file(Path(args.env_file))
    results = [
        check_noaa(args.timeout),
        check_cds(),
        check_gee(),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
