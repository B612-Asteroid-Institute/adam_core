"""Focused IOD qualification against an external propagation capsule."""

import json
from pathlib import Path

import numpy as np
import pytest

from adam_core._rust.arrow import pc

from ...coordinates import CoordinateCovariances, SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...utils.helpers.observations import make_observations
from ...utils.helpers.orbits import make_real_orbits
from ..evaluate import OrbitDeterminationObservations
from ._focused import run_iod_qualified, run_prefit_iod_qualified

ASSISTPropagator = pytest.importorskip("adam_assist").ASSISTPropagator


@pytest.fixture
def real_data() -> OrbitDeterminationObservations:
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


def test_focused_iod_returns_candidate_report(
    real_data: OrbitDeterminationObservations,
) -> None:
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


def test_focused_iod_is_deterministic(
    real_data: OrbitDeterminationObservations,
) -> None:
    first = run_iod_qualified(real_data[:10], ASSISTPropagator(), min_obs=6)
    second = run_iod_qualified(real_data[:10], ASSISTPropagator(), min_obs=6)
    np.testing.assert_array_equal(
        first[0].coordinates.values, second[0].coordinates.values
    )
    assert first[1].solution.to_pylist() == second[1].solution.to_pylist()


def test_prefit_iod_matches_core_0_7_baseline_fixture(
    real_data: OrbitDeterminationObservations,
) -> None:
    """Pin the accepted Core 0.7 single-arc ASSIST prefit products."""

    fixture_document = json.loads(
        (
            Path(__file__).parents[4]
            / "migration/artifacts/iod_prefit_core_0_7_fixture_2026-11-06.json"
        ).read_text()
    )
    assert fixture_document["authority"]["core_commit"] == (
        "0b49938eb0be04214a8d07b206227d31b248394a"
    )
    assert fixture_document["authority"]["assist_commit"] == (
        "824f96e58d59c3f7a16ec62831c084aeac2904b6"
    )
    fixture = fixture_document["case"]
    observations = real_data[:10]
    assert observations.id.to_pylist() == fixture["observation_ids"]

    fitted, members = run_prefit_iod_qualified(
        observations,
        ASSISTPropagator(),
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000.0,
        observation_selection_method="combinations",
        light_time=True,
    )

    # This preliminary orbit remains mildly ill-conditioned with respect to
    # platform libm RA/Dec conversion. Preserve the measured component-specific
    # cross-platform bounds around the accepted Core 0.7 prefit authority; see
    # iod_platform_numerics_2026-07-15.md.
    state_atol = [4e-12, 2e-11, 4e-12, 4e-13, 4e-13, 4e-13]
    for axis, atol in enumerate(state_atol):
        np.testing.assert_allclose(
            fitted.coordinates.values[:, axis],
            np.asarray(fixture["state"])[:, axis],
            rtol=0,
            atol=atol,
        )
    assert fitted.coordinates.time.scale == "utc"
    np.testing.assert_allclose(
        fitted.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        fixture["epoch_mjd"],
        rtol=0,
        atol=1e-11,
    )
    np.testing.assert_allclose(fitted.arc_length, fixture["arc_length"], rtol=0, atol=0)
    np.testing.assert_array_equal(fitted.num_obs, fixture["num_obs"])
    np.testing.assert_allclose(fitted.chi2, fixture["chi2"], rtol=1e-7, atol=2e-5)
    np.testing.assert_allclose(
        fitted.reduced_chi2, fixture["reduced_chi2"], rtol=1e-7, atol=2e-6
    )
    np.testing.assert_allclose(
        members.residuals.to_array(),
        fixture["residual_values"],
        rtol=0,
        atol=3e-11,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        members.residuals.chi2,
        fixture["residual_chi2"],
        rtol=2e-7,
        atol=2e-5,
    )
    assert members.obs_id.to_pylist() == fixture["member_obs_ids"]
    assert members.solution.to_pylist() == fixture["solution"]
    assert members.outlier.to_pylist() == fixture["outlier"]
