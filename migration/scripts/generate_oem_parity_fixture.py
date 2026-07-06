"""Generate the OEM writer/parser parity fixture (bead personal-cmy.28).

Run with the LEGACY baseline interpreter (untouched adam-core checkout, which
still writes through the third-party `oem` package):

    .legacy-venv/bin/python migration/scripts/generate_oem_parity_fixture.py

Freezes, for a panel of orbits (with/without covariance, tdb and utc time
systems, single and many epochs):

* the exact legacy `orbit_to_oem` file text (CREATION_DATE line masked by the
  consumer, since it is wall-clock dependent);
* the exact legacy `orbit_from_oem` parse of those files (flat Orbits
  columns), gating the Rust parser's epoch splits and covariance joins.

The migration test asserts the Rust-dispatched writer reproduces the text
byte-for-byte (modulo CREATION_DATE) and the parser reproduces the parse
bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.orbits.oem_io import orbit_from_oem, orbit_to_oem
from adam_core.time import Timestamp


def panel_orbits(seed: int, n: int, scale: str, with_covariance: bool) -> Orbits:
    rng = np.random.default_rng(seed)
    times = Timestamp.from_kwargs(
        days=rng.integers(59000, 61000, size=1).repeat(n) + np.arange(n),
        nanos=rng.integers(0, 86_400_000_000_000, size=n),
        scale=scale,
    )
    covariance = None
    if with_covariance:
        matrices = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            root = rng.normal(0.0, 1e-7, size=(6, 6))
            matrices[i] = root @ root.T
        # Leave one row without covariance to exercise the partial case.
        if n > 2:
            matrices[1] = np.nan
        covariance = CoordinateCovariances.from_matrix(matrices)
    return Orbits.from_kwargs(
        orbit_id=[f"orbit-{i}" for i in range(n)],
        object_id=["FIXTURE OBJECT"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=rng.uniform(-3, 3, n),
            y=rng.uniform(-3, 3, n),
            z=rng.uniform(-1, 1, n),
            vx=rng.uniform(-0.02, 0.02, n),
            vy=rng.uniform(-0.02, 0.02, n),
            vz=rng.uniform(-0.01, 0.01, n),
            time=times,
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN"] * n),
            covariance=covariance,
        ),
    )


def orbits_to_flat(orbits: Orbits) -> dict:
    coordinates = orbits.coordinates
    return {
        "orbit_id": orbits.orbit_id.to_pylist(),
        "object_id": orbits.object_id.to_pylist(),
        "days": coordinates.time.days.to_pylist(),
        "nanos": coordinates.time.nanos.to_pylist(),
        "scale": coordinates.time.scale,
        "frame": coordinates.frame,
        "origin": coordinates.origin.code.to_pylist(),
        "values": np.asarray(coordinates.values, dtype=np.float64).tolist(),
        "covariance": np.asarray(
            coordinates.covariance.to_matrix(), dtype=np.float64
        ).tolist(),
    }


PANELS = [
    {"name": "tdb_plain", "seed": 0, "n": 6, "scale": "tdb", "covariance": False},
    {"name": "tdb_covariance", "seed": 1, "n": 5, "scale": "tdb", "covariance": True},
    {"name": "utc_covariance", "seed": 2, "n": 4, "scale": "utc", "covariance": True},
    {"name": "two_states", "seed": 3, "n": 2, "scale": "tdb", "covariance": False},
]


def build_fixture() -> dict:
    panels = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in PANELS:
            orbits = panel_orbits(
                spec["seed"], spec["n"], spec["scale"], spec["covariance"]
            )
            path = str(Path(tmp) / f"{spec['name']}.oem")
            orbit_to_oem(orbits, path, originator="ADAM CORE PARITY FIXTURE")
            text = Path(path).read_text()
            parsed = orbit_from_oem(path)
            panels.append(
                {
                    "name": spec["name"],
                    "orbits": orbits_to_flat(orbits),
                    "oem_text": text,
                    "parsed": orbits_to_flat(parsed),
                }
            )
    return {
        "schema": "adam_core.oem_parity_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_oem_parity_fixture.py",
        "source_contract": (
            "Legacy adam-core orbit_to_oem/orbit_from_oem via the third-party "
            "oem package, executed in the untouched legacy checkout."
        ),
        "originator": "ADAM CORE PARITY FIXTURE",
        "panels": panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "migration"
        / "artifacts"
        / "oem_parity_fixture_2026-07-06.json",
    )
    args = parser.parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
