import os

import pytest

from ...propagator.pyoorb import PYOORB
from ..differential_correction import fit_least_squares


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
