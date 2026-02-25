"""
Fetch real SBDB and NEOCC payloads into testdata for parsing/regression tests.

Run from repo root (or from this directory):

    python src/adam_core/orbits/query/tests/testdata/fetch_real_payloads.py

Requires network. Saves to testdata/sbdb/ and testdata/neocc/. Use the READMEs
in those dirs for object IDs that exercise different response shapes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

_BASE = Path(__file__).resolve().parent
_SBDB_DIR = _BASE / "sbdb"
_NEOCC_DIR = _BASE / "neocc"
_SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
_NEOCC_URL = "https://neo.ssa.esa.int/PSDB-portlet/download"

# SBDB: well-known and lesser-known; numbered/unnumbered; asteroids and comets.
SBDB_IDS = [
    "1", "2", "3", "4", "433", "243", "25143", "99942", "101955",  # numbered asteroids
    "2001VB", "2015AB", "2022AP7", "2023DZ2", "2024YR4", "2022OB5",  # unnumbered NEOs
    "1P", "67P", "73P", "81P", "C/2022 E3", "C/2014 UN271",  # comets
    "Ceres", "Vesta", "Eros", "Bennu", "Apophis", "Halley",  # names
]
# NEOCC: NEO-focused; use space-free designations. Not all may be in NEOCC.
NEOCC_IDS = [
    "433", "99942", "101955", "162173", "65803",  # numbered NEOs
    "2024YR4", "2022OB5", "2023DZ2", "2015AB", "2001VB", "2022AP7",
    "2024BX1", "2023BU", "2012DA14",
]


def _sbdb_fetch(object_id: str) -> None:
    params = {
        "sstr": object_id,
        "cov": "mat",
        "full-prec": "true",
        "phys-par": "true",
    }
    resp = requests.get(_SBDB_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "object" not in data:
        print(f"  sbdb {object_id}: no object (multi-match or not found), skip save")
        return
    safe = re.sub(r"[^a-zA-Z0-9]", "_", object_id).strip("_") or "object"
    out = _SBDB_DIR / f"{safe}_phys.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  sbdb {object_id} -> {out.name}")


def _neocc_fetch(designation: str, orbit_type: str = "ke", epoch: int = 1) -> None:
    file_name = f"{designation.replace(' ', '')}.{orbit_type}{epoch}"
    resp = requests.get(_NEOCC_URL, params={"file": file_name}, timeout=30)
    resp.raise_for_status()
    out = _NEOCC_DIR / file_name
    out.write_text(resp.text, encoding="utf-8")
    print(f"  neocc {file_name} -> {out.name}")


def main() -> None:
    print("Fetching SBDB (with phys-par=1)...")
    for oid in SBDB_IDS:
        try:
            _sbdb_fetch(oid)
        except requests.RequestException as e:
            print(f"  sbdb {oid}: {e}")

    print("Fetching NEOCC OEF...")
    for desig in NEOCC_IDS:
        for epoch in (0, 1):
            try:
                _neocc_fetch(desig, epoch=epoch)
            except requests.RequestException as e:
                print(f"  neocc {desig} ke{epoch}: {e}")


if __name__ == "__main__":
    main()
