"""Generate frozen baseline-main fixtures for ``evaluate_orbits``.

Run with the pristine legacy environment and pinned checkout, for example::

    PYTHONPATH=/Users/aleck/Code/adam-core-legacy-main/src \
      .legacy-venv/bin/python migration/scripts/generate_evaluate_orbits_fixture.py
"""

import json
from pathlib import Path

import numpy as np
import quivr as qv

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.observers.observers import Observers
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.time.time import Timestamp
from adam_core.utils.helpers.orbits import make_real_orbits

OUT = Path("migration/artifacts/evaluate_orbits_fixture_2026-07-12.json")


class FixedEphemerisPropagator:
    def __init__(self, ephemeris):
        self.ephemeris = ephemeris

    def generate_ephemeris(self, orbits, observers, max_processes=1):
        return self.ephemeris


def make_coordinates(count, longitude_offset=0.0):
    time = Timestamp.from_mjd(
        59000 + np.arange(count, dtype=np.float64) / 1440,
        scale="utc",
    )
    covariance = np.tile(np.eye(6, dtype=np.float64), (count, 1, 1))
    return SphericalCoordinates.from_kwargs(
        rho=np.full(count, 1.0),
        lon=np.linspace(10.0, 11.0, count) + longitude_offset,
        lat=np.linspace(-1.0, 1.0, count),
        vrho=np.full(count, 0.01),
        vlon=np.full(count, 0.02),
        vlat=np.full(count, 0.03),
        covariance=CoordinateCovariances.from_matrix(covariance),
        origin=Origin.from_kwargs(code=np.full(count, "500", dtype=object)),
        time=time,
        frame="equatorial",
    )


def make_case(num_orbits, num_observations):
    orbits = make_real_orbits(num_orbits).sort_by(["orbit_id"])
    coordinates = make_coordinates(num_observations)
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i:03d}" for i in range(num_observations)],
        coordinates=coordinates,
        observers=Observers.from_code("500", coordinates.time),
    )
    blocks = []
    orbit_ids = []
    object_ids = []
    for row, orbit_id in enumerate(orbits.orbit_id.to_pylist()):
        blocks.append(make_coordinates(num_observations, 0.001 * (row + 1)))
        orbit_ids.extend([orbit_id] * num_observations)
        object_ids.extend([orbits.object_id[row].as_py()] * num_observations)
    ephemeris = Ephemeris.from_kwargs(
        orbit_id=orbit_ids,
        object_id=object_ids,
        coordinates=qv.concatenate(blocks),
    )
    return orbits, observations, FixedEphemerisPropagator(ephemeris)


def result(case, ignore):
    orbits, observations, propagator = case
    fitted, members = evaluate_orbits(
        orbits, observations, propagator, parameters=6, ignore=ignore
    )
    return {
        "orbit_ids": fitted.orbit_id.to_pylist(),
        "arc_length": fitted.arc_length.to_pylist(),
        "num_obs": fitted.num_obs.to_pylist(),
        "chi2": fitted.chi2.to_pylist(),
        "reduced_chi2": fitted.reduced_chi2.to_pylist(),
        "member_obs_ids": members.obs_id.to_pylist(),
        "member_outlier": members.outlier.to_pylist(),
        "residual_values": members.residuals.values.to_pylist(),
        "residual_chi2": members.residuals.chi2.to_pylist(),
        "residual_dof": members.residuals.dof.to_pylist(),
        "residual_probability": members.residuals.probability.to_pylist(),
    }


def error(case):
    try:
        evaluate_orbits(*case)
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}
    return {"result": "success"}


def main():
    empty = make_case(1, 0)
    ordering_error = make_case(2, 3)
    ordering_error[2].ephemeris = ordering_error[2].ephemeris.take([0, 1, 2, 3, 4])
    payload = {
        "legacy_commit": "4c1fbc4cd1a67b1e8527f20dce0b853b9a4022ac",
        "normal": result(make_case(4, 8), None),
        "ignored": result(make_case(4, 8), ["obs001", "obs006"]),
        "empty": error(empty),
        "ordering_error": error(ordering_error),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
