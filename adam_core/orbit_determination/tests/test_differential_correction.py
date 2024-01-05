import os
from importlib.resources import files

import numpy as np
import pytest

from ...propagator.pyoorb import PYOORB
from ..differential_correction import (
    OrbitDeterminationObservations,
    evaluate_orbit,
    fit_least_squares,
)
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits


@pytest.fixture
def pure_iod_orbit():
    orbit = FittedOrbits.from_parquet(
        files("adam_core.orbit_determination.tests.data").joinpath(
            "pure_iod_orbit.parquet"
        )
    )
    orbit_members = FittedOrbitMembers.from_parquet(
        files("adam_core.orbit_determination.tests.data").joinpath(
            "pure_iod_orbit_members.parquet"
        )
    )
    observations = OrbitDeterminationObservations.from_parquet(
        files("adam_core.orbit_determination.tests.data").joinpath(
            "pure_iod_orbit_observations.parquet"
        )
    )
    return orbit, orbit_members, observations


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
def test_evaluate_orbit(pure_iod_orbit):
    # Test that evaluate_orbit correctly calculates residuals and other
    # parameters for an input orbit
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORB()

    fitted_orbit, fitted_orbit_members = evaluate_orbit(orbit, observations, propagator)

    # Check that the returned orbit is the same as the input orbit (this function
    # has merely evaluated the orbit, not changed it)
    assert fitted_orbit.orbit_id[0].as_py() == orbit.orbit_id[0].as_py()
    assert fitted_orbit.object_id[0].as_py() == orbit.object_id[0].as_py()
    assert fitted_orbit.coordinates == orbit.coordinates
    assert fitted_orbit.arc_length[0].as_py() == orbit.arc_length[0].as_py()
    assert fitted_orbit.num_obs[0].as_py() == orbit.num_obs[0].as_py()
    assert fitted_orbit.chi2[0].as_py() == orbit.chi2[0].as_py()
    assert fitted_orbit.reduced_chi2[0].as_py() == orbit.reduced_chi2[0].as_py()

    # Check that the returned orbit members are correctly evaluated
    assert len(fitted_orbit_members) == len(orbit_members)
    assert fitted_orbit_members.orbit_id.tolist() == orbit_members.orbit_id.tolist()
    assert fitted_orbit_members.obs_id.tolist() == orbit_members.obs_id.tolist()
    assert fitted_orbit_members.outlier.tolist() == orbit_members.outlier.tolist()
    np.testing.assert_almost_equal(
        fitted_orbit_members.residuals.to_array(), orbit_members.residuals.to_array()
    )


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
def test_evaluate_orbit_outliers(pure_iod_orbit):
    # Test that evaluate_orbit correctly calculates residuals and other
    # parameters for an input orbit with outliers defined
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORB()

    # Lets remove the last two observations
    outliers = observations.id.tolist()[-2:]

    fitted_orbit, fitted_orbit_members = evaluate_orbit(
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


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
def test_fit_least_squares_pure_iod_orbit(pure_iod_orbit):
    # Test that fit_least_squares can fit and improve a pure orbit from an IOD
    # process using least squares

    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORB()

    fitted_orbit, fitted_orbit_members = fit_least_squares(
        orbit, observations, propagator
    )

    assert len(fitted_orbit) == 1
    assert len(fitted_orbit_members) == len(orbit_members) == len(observations)
    assert fitted_orbit.reduced_chi2[0].as_py() < (orbit.reduced_chi2[0].as_py() / 1e4)
    assert fitted_orbit.status_code[0].as_py() > 0
    assert fitted_orbit.iterations[0].as_py() <= 50
