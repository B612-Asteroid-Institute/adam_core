"""Focused IOD qualification against an external propagation capsule."""

import numpy as np
import pyarrow.compute as pc
import pytest

from ...coordinates import CoordinateCovariances, SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...utils.helpers.observations import make_observations
from ...utils.helpers.orbits import make_real_orbits
from ..evaluate import OrbitDeterminationObservations
from ._focused import run_iod_qualified

ASSISTPropagator = pytest.importorskip("adam_assist").ASSISTPropagator


@pytest.fixture
def real_data():
    exposures, detections, associations = make_observations()
    orbits = make_real_orbits(num_orbits=18)
    object_id = orbits.object_id[-1].as_py()
    associations_i = associations.select("object_id", object_id)
    detections_i = detections.apply_mask(
        pc.is_in(detections.id, associations_i.detection_id)
    )
    exposures_i = exposures.apply_mask(pc.is_in(exposures.id, detections_i.exposure_id))
    sigmas = np.full((len(detections_i), 6), np.nan)
    sigmas[:, 1] = detections_i.ra_sigma.to_numpy(zero_copy_only=False)
    sigmas[:, 2] = detections_i.dec_sigma.to_numpy(zero_copy_only=False)
    coordinates = SphericalCoordinates.from_kwargs(
        lon=detections_i.ra,
        lat=detections_i.dec,
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        origin=Origin.from_kwargs(code=exposures_i.observatory_code),
        time=exposures_i.midpoint(),
        frame="equatorial",
    )
    observers = Observers.from_codes(
        times=exposures_i.midpoint(), codes=exposures_i.observatory_code
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=detections_i.id, coordinates=coordinates, observers=observers
    )


def test_focused_iod_returns_candidate_report(real_data) -> None:
    fitted, members, report = run_iod_qualified(
        real_data[:10],
        ASSISTPropagator(),
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000.0,
    )
    assert report.found
    assert len(fitted) == 1
    assert len(members) == 10
    assert report.accepted_candidate is not None


def test_focused_iod_is_deterministic(real_data) -> None:
    first = run_iod_qualified(real_data[:10], ASSISTPropagator(), min_obs=6)
    second = run_iod_qualified(real_data[:10], ASSISTPropagator(), min_obs=6)
    np.testing.assert_array_equal(
        first[0].coordinates.values, second[0].coordinates.values
    )
    assert first[1].solution.to_pylist() == second[1].solution.to_pylist()
