import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest
import quivr as qv
from adam_assist import ASSISTPropagator

from ...coordinates import CoordinateCovariances, SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...utils.helpers.observations import make_observations
from ...utils.helpers.orbits import make_real_orbits
from ..evaluate import OrbitDeterminationObservations
from ..iod import initial_orbit_determination, iod


class LinkageMembers(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()


class ComposedASSISTPropagator:
    """Expose only the provider boundary to exercise the compatibility path."""

    def __init__(self, **kwargs):
        self.inner = ASSISTPropagator(**kwargs)

    def generate_ephemeris(self, *args, **kwargs):
        return self.inner.generate_ephemeris(*args, **kwargs)


@pytest.fixture
def real_data():
    # Generate real observations and orbits
    exposures, detections, associations = make_observations()
    orbits = make_real_orbits(num_orbits=18)

    # Select a specific object ID for testing
    object_id = orbits.object_id[-1].as_py()
    orbit = orbits.select("object_id", object_id)

    # Filter observations for the selected object ID
    associations_i = associations.select("object_id", object_id)
    detections_i = detections.apply_mask(
        pc.is_in(detections.id, associations_i.detection_id)
    )

    exposures_i = exposures.apply_mask(pc.is_in(exposures.id, detections_i.exposure_id))

    sigmas = np.full((len(detections_i.ra_sigma), 6), np.nan)
    sigmas[:, 1] = detections_i.ra_sigma.to_numpy(zero_copy_only=False)
    sigmas[:, 2] = detections_i.dec_sigma.to_numpy(zero_copy_only=False)

    coordinates = SphericalCoordinates.from_kwargs(
        lon=detections_i.ra.to_numpy(),
        lat=detections_i.dec.to_numpy(),
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        origin=Origin.from_kwargs(code=exposures_i.observatory_code),
        time=exposures_i.midpoint(),
        frame="equatorial",  # Assuming the frame is equatorial
    )

    # Generate Observers from exposures start_time and observatory codes
    observers = Observers.from_codes(
        times=exposures_i.midpoint(), codes=exposures_i.observatory_code
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=detections_i.id.to_numpy(zero_copy_only=False),
        coordinates=coordinates,
        observers=observers,
    )

    return orbit, observations


def test_iod(real_data):
    orbit, observations = real_data
    # Call the iod function
    fitted_orbits, fitted_orbit_members = iod(
        observations[:10],
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000,
        observation_selection_method="combinations",
        iterate=False,
        light_time=True,
        propagator=ASSISTPropagator,
    )
    # save these out as parquet

    # Assertions
    assert len(fitted_orbits) == 1, "No orbits were fitted"
    assert len(fitted_orbit_members) == 10, "No orbit members were fitted"
    assert fitted_orbits.orbit_id[0].as_py() is not None, "Orbit ID is None"
    assert fitted_orbits.coordinates is not None, "Coordinates are None"


def test_iod_fused_matches_pinned_legacy_fixture(real_data):
    fixture = json.loads(
        (
            Path(__file__).parents[4]
            / "migration/artifacts/iod_orchestration_fixture_2026-07-12.json"
        ).read_text()
    )["case"]
    _, observations = real_data
    observations = observations[:10]
    assert observations.id.to_pylist() == fixture["observation_ids"]
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
    np.testing.assert_allclose(
        fitted.coordinates.values, fixture["state"], rtol=0, atol=4e-12
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
        members.residuals.chi2, fixture["residual_chi2"], rtol=2e-7, atol=2e-5
    )
    assert members.obs_id.to_pylist() == fixture["member_obs_ids"]
    assert members.solution.to_pylist() == fixture["solution"]
    assert members.outlier.to_pylist() == fixture["outlier"]


def test_iod_fused_matches_composed_path(real_data):
    _, observations = real_data
    kwargs = dict(
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000,
        observation_selection_method="combinations",
        iterate=False,
        light_time=True,
    )
    fused, fused_members = iod(observations[:10], propagator=ASSISTPropagator, **kwargs)
    composed, composed_members = iod(
        observations[:10], propagator=ComposedASSISTPropagator, **kwargs
    )
    np.testing.assert_allclose(
        fused.coordinates.values, composed.coordinates.values, rtol=0, atol=4e-12
    )
    np.testing.assert_allclose(
        fused_members.residuals.to_array(),
        composed_members.residuals.to_array(),
        rtol=0,
        atol=2e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(fused.chi2, composed.chi2, rtol=1e-6, atol=1e-3)
    np.testing.assert_array_equal(fused.arc_length, composed.arc_length)
    np.testing.assert_array_equal(fused.num_obs, composed.num_obs)
    assert fused_members.obs_id.to_pylist() == composed_members.obs_id.to_pylist()
    assert fused_members.solution.to_pylist() == composed_members.solution.to_pylist()
    assert fused_members.outlier.to_pylist() == composed_members.outlier.to_pylist()


def test_initial_orbit_determination_fused_batch_deduplicates(real_data):
    _, observations = real_data
    observations = observations[:10]
    obs_ids = observations.id.to_pylist()
    members = LinkageMembers.from_kwargs(
        cluster_id=["b"] * len(obs_ids) + ["a"] * len(obs_ids),
        obs_id=obs_ids + obs_ids,
    )
    orbits, orbit_members = initial_orbit_determination(
        observations,
        members,
        propagator=ASSISTPropagator,
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000,
        contamination_percentage=0.0,
        chunk_size=2,
        max_processes=4,
    )
    # Exact-state duplicates keep the first linkage in member first-appearance
    # order, matching drop_duplicate_orbits(..., keep="first").
    assert orbits.orbit_id.to_pylist() == ["b"]
    assert orbit_members.orbit_id.to_pylist() == ["b"] * len(obs_ids)
    assert orbit_members.obs_id.to_pylist() == obs_ids


def test_initial_orbit_determination_fused_empty(real_data):
    _, observations = real_data
    empty_members = LinkageMembers.empty()
    orbits, members = initial_orbit_determination(
        observations[:0], empty_members, propagator=ASSISTPropagator
    )
    assert len(orbits) == 0
    assert len(members) == 0
