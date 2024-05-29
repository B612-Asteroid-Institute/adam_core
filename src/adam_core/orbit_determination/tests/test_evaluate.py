import os

import numpy as np
import pyarrow as pa
import pytest
import quivr as qv

try:
    from adam_core.propagator.adam_pyoorb import PYOORBPropagator
except ImportError:
    PYOORBPropagator = None

from ..evaluate import evaluate_orbits


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
