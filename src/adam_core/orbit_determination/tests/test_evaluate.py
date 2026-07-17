import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import quivr as qv

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.observers.observers import Observers
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits
from adam_core.time.time import Timestamp
from adam_core.utils.helpers.orbits import make_real_orbits

try:
    from adam_core.propagator.adam_pyoorb import PYOORBPropagator
except ImportError:
    PYOORBPropagator = None

from ..evaluate import OrbitDeterminationObservations, evaluate_orbits


class FixedEphemerisPropagator:
    def __init__(self, ephemeris: Ephemeris) -> None:
        self.ephemeris = ephemeris

    def generate_ephemeris(
        self,
        orbits: Orbits,
        observers: Observers,
        max_processes: int = 1,
    ) -> Ephemeris:
        return self.ephemeris


def make_spherical_observations(
    num_observations: int, lon_offset: float = 0.0
) -> SphericalCoordinates:
    time = Timestamp.from_mjd(
        59000 + np.arange(num_observations, dtype=np.float64) / 1440,
        scale="utc",
    )
    covariance = np.tile(np.eye(6, dtype=np.float64), (num_observations, 1, 1))
    return SphericalCoordinates.from_kwargs(
        rho=np.full(num_observations, 1.0),
        lon=np.linspace(10.0, 11.0, num_observations) + lon_offset,
        lat=np.linspace(-1.0, 1.0, num_observations),
        vrho=np.full(num_observations, 0.01),
        vlon=np.full(num_observations, 0.02),
        vlat=np.full(num_observations, 0.03),
        covariance=CoordinateCovariances.from_matrix(covariance),
        origin=Origin.from_kwargs(
            code=np.full(num_observations, "500", dtype="object")
        ),
        time=time,
        frame="equatorial",
    )


def make_evaluate_case(
    num_orbits: int, num_observations: int
) -> tuple[Orbits, OrbitDeterminationObservations, FixedEphemerisPropagator]:
    orbits = make_real_orbits(num_orbits).sort_by(["orbit_id"])
    observed_coordinates = make_spherical_observations(num_observations)
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i:03d}" for i in range(num_observations)],
        coordinates=observed_coordinates,
        observers=Observers.from_code("500", observed_coordinates.time),
    )

    predicted_blocks = []
    ephemeris_orbit_ids: list[str] = []
    ephemeris_object_ids: list[str | None] = []
    for orbit_index, orbit_id in enumerate(orbits.orbit_id.to_pylist()):
        predicted_blocks.append(
            make_spherical_observations(
                num_observations,
                lon_offset=0.001 * (orbit_index + 1),
            )
        )
        ephemeris_orbit_ids.extend([orbit_id] * num_observations)
        ephemeris_object_ids.extend(
            [orbits.object_id[orbit_index].as_py()] * num_observations
        )

    ephemeris = Ephemeris.from_kwargs(
        orbit_id=ephemeris_orbit_ids,
        object_id=ephemeris_object_ids,
        coordinates=qv.concatenate(predicted_blocks),
    )
    return orbits, observations, FixedEphemerisPropagator(ephemeris)


def expected_statistics(
    observations: OrbitDeterminationObservations,
    ephemeris: Ephemeris,
    include_mask: np.ndarray,
    num_orbits: int,
    parameters: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_observations = len(observations)
    observation_indices = np.tile(
        np.arange(num_observations, dtype=np.int64), num_orbits
    )
    residuals = Residuals.calculate(
        observations.coordinates.take(observation_indices), ephemeris.coordinates
    )
    chi2_rows = residuals.chi2.to_numpy(zero_copy_only=False).reshape(
        num_orbits, num_observations
    )
    dof_rows = residuals.dof.to_numpy(zero_copy_only=False).reshape(
        num_orbits, num_observations
    )
    chi2 = chi2_rows[:, include_mask].sum(axis=1)
    reduced_chi2 = chi2 / (dof_rows[:, include_mask].sum(axis=1) - parameters)
    return chi2, reduced_chi2


def test_evaluate_orbits_vectorized_statistics() -> None:
    num_orbits = 4
    num_observations = 8
    parameters = 6
    orbits, observations, propagator = make_evaluate_case(num_orbits, num_observations)
    ignore = ["obs001", "obs006"]

    fitted_orbits, fitted_orbit_members = evaluate_orbits(
        orbits,
        observations,
        propagator,
        parameters=parameters,
        ignore=ignore,
    )

    include_mask = np.array(
        [obs_id not in ignore for obs_id in observations.id.to_pylist()],
        dtype=bool,
    )
    expected_chi2, expected_reduced_chi2 = expected_statistics(
        observations,
        propagator.ephemeris,
        include_mask,
        num_orbits,
        parameters,
    )

    assert fitted_orbits.orbit_id.to_pylist() == orbits.orbit_id.to_pylist()
    np.testing.assert_allclose(
        fitted_orbits.chi2.to_numpy(zero_copy_only=False), expected_chi2
    )
    np.testing.assert_allclose(
        fitted_orbits.reduced_chi2.to_numpy(zero_copy_only=False),
        expected_reduced_chi2,
    )
    assert (
        fitted_orbits.num_obs.to_pylist()
        == [num_observations - len(ignore)] * num_orbits
    )

    expected_obs_ids = observations.id.to_pylist() * num_orbits
    expected_outliers = [
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ] * num_orbits
    assert fitted_orbit_members.obs_id.to_pylist() == expected_obs_ids
    assert fitted_orbit_members.outlier.to_pylist() == expected_outliers


@pytest.mark.parametrize(
    ("fixture_case", "ignore"),
    [("normal", None), ("ignored", ["obs001", "obs006"])],
)
def test_evaluate_orbits_frozen_legacy_parity(fixture_case, ignore) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "migration"
        / "artifacts"
        / "evaluate_orbits_fixture_2026-07-12.json"
    )
    expected = json.loads(fixture_path.read_text())[fixture_case]
    orbits, observations, propagator = make_evaluate_case(4, 8)
    fitted, members = evaluate_orbits(
        orbits, observations, propagator, parameters=6, ignore=ignore
    )

    assert fitted.orbit_id.to_pylist() == expected["orbit_ids"]
    assert fitted.num_obs.to_pylist() == expected["num_obs"]
    assert members.obs_id.to_pylist() == expected["member_obs_ids"]
    assert members.outlier.to_pylist() == expected["member_outlier"]
    np.testing.assert_allclose(fitted.arc_length.to_pylist(), expected["arc_length"])
    np.testing.assert_allclose(fitted.chi2.to_pylist(), expected["chi2"], rtol=1e-14)
    np.testing.assert_allclose(
        fitted.reduced_chi2.to_pylist(), expected["reduced_chi2"], rtol=1e-14
    )
    np.testing.assert_allclose(
        members.residuals.to_array(), expected["residual_values"], rtol=1e-14
    )
    np.testing.assert_allclose(
        members.residuals.chi2.to_pylist(), expected["residual_chi2"], rtol=1e-14
    )
    assert members.residuals.dof.to_pylist() == expected["residual_dof"]
    np.testing.assert_allclose(
        members.residuals.probability.to_pylist(),
        expected["residual_probability"],
        rtol=1e-14,
    )


def test_evaluate_orbits_empty_legacy_error() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "migration"
        / "artifacts"
        / "evaluate_orbits_fixture_2026-07-12.json"
    )
    expected = json.loads(fixture_path.read_text())["empty"]
    orbits, observations, propagator = make_evaluate_case(1, 0)
    with pytest.raises(ValueError, match=expected["error"]):
        evaluate_orbits(orbits, observations, propagator)


def test_evaluate_orbits_rust_native_timing() -> None:
    from adam_core import _rust_native

    orbits, observations, propagator = make_evaluate_case(4, 8)
    observed = observations.coordinates
    predicted = propagator.ephemeris.coordinates
    samples = _rust_native.benchmark_evaluate_orbits_numpy(
        orbits.orbit_id.to_pylist(),
        propagator.ephemeris.orbit_id.to_pylist(),
        observations.id.to_pylist(),
        observed.origin.code.to_pylist(),
        predicted.origin.code.to_pylist(),
        observed.frame,
        predicted.frame,
        np.ascontiguousarray(observed.values),
        np.ascontiguousarray(predicted.values),
        np.ascontiguousarray(observed.covariance.to_matrix()),
        np.ascontiguousarray(predicted.covariance.to_matrix()),
        np.ascontiguousarray(observed.time.days.to_numpy()),
        np.ascontiguousarray(observed.time.nanos.to_numpy()),
        6,
        ["obs001", "obs006"],
        2,
        2,
        0,
    )
    assert len(samples) == 2
    assert all(len(trial) == 2 for trial in samples)
    assert all(sample >= 0 for trial in samples for sample in trial)


@pytest.mark.parametrize("indices", [[0, 1, 2, 3, 4], [3, 4, 5, 0, 1, 2]])
def test_evaluate_orbits_requires_grouped_ephemeris_order(indices) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "migration"
        / "artifacts"
        / "evaluate_orbits_fixture_2026-07-12.json"
    )
    baseline_error = json.loads(fixture_path.read_text())["ordering_error"]
    assert baseline_error["error_type"] == "ValueError"

    orbits, observations, propagator = make_evaluate_case(2, 3)
    propagator.ephemeris = propagator.ephemeris.take(indices)
    with pytest.raises(ValueError, match="Ephemeris rows must be grouped"):
        evaluate_orbits(orbits, observations, propagator)


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_evaluate_orbits(pure_iod_orbit):
    # Test that evaluate_orbit correctly calculates residuals and other
    # parameters for an input orbit
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    # Concatenate the orbit three times to test we can handle multiple orbits
    orbits = qv.concatenate([orbit, orbit, orbit])
    orbits = orbits.set_column(
        "orbit_id", pa.array(["orbit01", "orbit02", "orbit03"], type=pa.large_string())
    )

    fitted_orbits, fitted_orbits_members = evaluate_orbits(
        orbits, observations, propagator
    )

    # Check that the returned orbit is the same as the input orbit (this function
    # has merely evaluated the orbit, not changed it)
    assert fitted_orbits.orbit_id.to_pylist() == orbits.orbit_id.to_pylist()
    assert fitted_orbits.object_id.to_pylist() == orbits.object_id.to_pylist()
    assert fitted_orbits.coordinates == orbits.coordinates
    assert fitted_orbits.arc_length.to_pylist() == orbits.arc_length.to_pylist()
    assert fitted_orbits.num_obs.to_pylist() == orbits.num_obs.to_pylist()
    assert fitted_orbits.chi2.to_pylist() == orbits.chi2.to_pylist()
    assert fitted_orbits.reduced_chi2.to_pylist() == orbits.reduced_chi2.to_pylist()

    # Loop through each orbit and check that the returned orbit members are correctly evaluated
    for orbit_id in orbits.orbit_id.to_pylist():

        fitted_orbits_members_i = fitted_orbits_members.select("orbit_id", orbit_id)
        assert len(fitted_orbits_members_i) == len(orbit_members)
        assert fitted_orbits_members_i.orbit_id.tolist() == [
            orbit_id for _ in range(len(orbit_members))
        ]
        assert fitted_orbits_members_i.obs_id.tolist() == orbit_members.obs_id.tolist()
        assert (
            fitted_orbits_members_i.outlier.tolist() == orbit_members.outlier.tolist()
        )
        np.testing.assert_almost_equal(
            fitted_orbits_members_i.residuals.to_array(),
            orbit_members.residuals.to_array(),
        )


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_evaluate_orbits_outliers(pure_iod_orbit):
    # Test that evaluate_orbit correctly calculates residuals and other
    # parameters for an input orbit with outliers defined
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    # Lets remove the last two observations
    outliers = observations.id.tolist()[-2:]

    fitted_orbit, fitted_orbit_members = evaluate_orbits(
        orbit.to_orbits(), observations, propagator, ignore=outliers
    )

    # Check that the returned orbit's ID, object ID and coordinates are the same
    assert fitted_orbit.orbit_id[0].as_py() == orbit.orbit_id[0].as_py()
    assert fitted_orbit.object_id[0].as_py() == orbit.object_id[0].as_py()
    assert fitted_orbit.coordinates == orbit.coordinates

    # Because we marked two observations as outliers we expect that the arc length, number of observations
    # and chi2 values will be different
    assert fitted_orbit.arc_length[0].as_py() < orbit.arc_length[0].as_py()
    assert fitted_orbit.num_obs[0].as_py() == (orbit.num_obs[0].as_py() - 2)
    assert fitted_orbit.chi2[0].as_py() < orbit.chi2[0].as_py()
    assert fitted_orbit.reduced_chi2[0].as_py() < orbit.reduced_chi2[0].as_py()

    # Check that the returned orbit members are correctly evaluated
    assert len(fitted_orbit_members) == len(orbit_members)
    assert fitted_orbit_members.orbit_id.tolist() == orbit_members.orbit_id.tolist()
    assert fitted_orbit_members.obs_id.tolist() == orbit_members.obs_id.tolist()
    assert fitted_orbit_members.outlier.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    np.testing.assert_almost_equal(
        fitted_orbit_members.residuals.to_array(), orbit_members.residuals.to_array()
    )
