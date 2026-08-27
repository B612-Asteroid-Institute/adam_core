"""Generate the space-observatory (custom SPICE kernel) parity fixture.

Correction to the personal-cmy.27 audit: legacy DOES support space-based
observatories through custom SPICE kernels (register_spice_kernel +
get_observer_state("JWST", ...) -> bodn2c -> get_spice_body_state), with the
JWST Horizons kernel vendored in both checkouts. This fixture freezes the
legacy (CSPICE) JWST heliocentric-ecliptic states over an epoch panel so the
migration checkout's spicekit-served path is numerically gated against it.

Run with the LEGACY baseline interpreter:

    .legacy-venv/bin/python migration/scripts/generate_jwst_observer_fixture.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adam_core.coordinates.origin import OriginCodes
from adam_core.observers.state import get_observer_state
from adam_core.time import Timestamp
from adam_core.utils.spice import register_spice_kernel, unregister_spice_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = (
    REPO_ROOT
    / "src"
    / "adam_core"
    / "utils"
    / "tests"
    / "data"
    / "spice"
    / "jwst_horizons_20200101_20240101_v01.bsp"
)

# Epoch panel inside the kernel coverage (2020-01-01 .. 2024-01-01), mixing
# round days and irregular fractions.
EPOCH_MJDS_TDB = [
    59580.0,
    59580.5,
    59674.25,
    59795.125,
    59941.8125,
    60100.03125,
    60250.9,
    60301.55,
]

FRAMES = ["ecliptic", "equatorial"]


def build_fixture() -> dict:
    register_spice_kernel(str(KERNEL_PATH))
    try:
        times = Timestamp.from_mjd(np.asarray(EPOCH_MJDS_TDB), scale="tdb")
        states = {}
        for frame in FRAMES:
            coordinates = get_observer_state(
                "JWST", times, frame=frame, origin=OriginCodes.SUN
            )
            states[frame] = np.asarray(coordinates.values, dtype=np.float64).tolist()
    finally:
        unregister_spice_kernel(str(KERNEL_PATH))
    return {
        "schema": "adam_core.jwst_observer_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_jwst_observer_fixture.py",
        "source_contract": (
            "Legacy adam-core get_observer_state('JWST', ...) via CSPICE with "
            "the vendored JWST Horizons kernel, executed in the untouched "
            "legacy checkout."
        ),
        "kernel": KERNEL_PATH.name,
        "epoch_mjds_tdb": EPOCH_MJDS_TDB,
        "origin": "SUN",
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "migration"
        / "artifacts"
        / "jwst_observer_fixture_2026-07-06.json",
    )
    args = parser.parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
