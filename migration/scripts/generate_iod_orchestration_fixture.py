"""Generate the pinned-main end-to-end IOD orchestration fixture.

Run with the pristine legacy ASSIST environment and pinned checkout::

    PYTHONPATH=/Users/aleck/Code/adam-core-legacy-main/src \
      .legacy-assist-venv/bin/python \
      migration/scripts/generate_iod_orchestration_fixture.py
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from adam_assist import ASSISTPropagator

from adam_core.coordinates import CoordinateCovariances, SphericalCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.iod import iod
from adam_core.utils.helpers.observations import make_observations
from adam_core.utils.helpers.orbits import make_real_orbits

OUT = Path("migration/artifacts/iod_orchestration_fixture_2026-07-12.json")


def make_case():
    exposures, detections, associations = make_observations()
    orbits = make_real_orbits(num_orbits=18)
    object_id = orbits.object_id[-1].as_py()
    associations = associations.select("object_id", object_id)
    detections = detections.apply_mask(
        pc.is_in(detections.id, associations.detection_id)
    )
    exposures = exposures.apply_mask(pc.is_in(exposures.id, detections.exposure_id))
    sigmas = np.full((len(detections), 6), np.nan)
    sigmas[:, 1] = detections.ra_sigma.to_numpy(zero_copy_only=False)
    sigmas[:, 2] = detections.dec_sigma.to_numpy(zero_copy_only=False)
    coordinates = SphericalCoordinates.from_kwargs(
        lon=detections.ra.to_numpy(),
        lat=detections.dec.to_numpy(),
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        origin=Origin.from_kwargs(code=exposures.observatory_code),
        time=exposures.midpoint(),
        frame="equatorial",
    )
    observations = OrbitDeterminationObservations.from_kwargs(
        id=detections.id.to_numpy(zero_copy_only=False),
        coordinates=coordinates,
        observers=Observers.from_codes(
            times=exposures.midpoint(), codes=exposures.observatory_code
        ),
    )[:10]
    fitted, members = iod(
        observations,
        propagator=ASSISTPropagator,
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000,
        observation_selection_method="combinations",
        iterate=False,
        light_time=True,
    )
    return {
        "observation_ids": observations.id.to_pylist(),
        "state": fitted.coordinates.values.tolist(),
        "epoch_mjd": fitted.coordinates.time.mjd().to_pylist(),
        "arc_length": fitted.arc_length.to_pylist(),
        "num_obs": fitted.num_obs.to_pylist(),
        "chi2": fitted.chi2.to_pylist(),
        "reduced_chi2": fitted.reduced_chi2.to_pylist(),
        "member_obs_ids": members.obs_id.to_pylist(),
        "residual_values": members.residuals.to_array().tolist(),
        "residual_chi2": members.residuals.chi2.to_pylist(),
        "residual_dof": members.residuals.dof.to_pylist(),
        "residual_probability": members.residuals.probability.to_pylist(),
        "solution": members.solution.to_pylist(),
        "outlier": members.outlier.to_pylist(),
    }


def main():
    # This downstream ASSIST fixture intentionally uses .legacy-assist-venv;
    # unlike the adam-core oracle, its compatible package pair remains pinned.
    payload = {
        "legacy_commit": "4c1fbc4cd1a67b1e8527f20dce0b853b9a4022ac",
        "case": make_case(),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
